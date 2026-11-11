"""
Google Drive part: log in, find every video in the folder, work out whose it is,
and download it. Read-only: this code can never change or delete anything on Drive.
"""
import io
import os
import re

from . import config

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_TYPE = "application/vnd.google-apps.folder"


def get_drive():
    """Log in to Google. A browser window opens the first time only."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if config.TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(config.TOKEN_FILE), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not config.CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    "credentials.json not found. Follow README step 4 to create it "
                    "and put it next to app.py."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(config.CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        config.TOKEN_FILE.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds)


def extract_folder_id(link_or_id):
    """Accepts a full Drive link or just the folder ID."""
    text = link_or_id.strip()
    m = re.search(r"folders/([\w-]+)", text) or re.search(r"[?&]id=([\w-]+)", text)
    return m.group(1) if m else text


def _list_children(drive, folder_id):
    items, token = [], None
    while True:
        res = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, mimeType, createdTime)",
            pageSize=200,
            pageToken=token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        items += res.get("files", [])
        token = res.get("nextPageToken")
        if not token:
            return items


def _is_video(item):
    name = item["name"].lower()
    return item["mimeType"].startswith("video/") or name.endswith(config.VIDEO_EXTENSIONS)


def _clean_name(text):
    text = re.sub(r"\s+", " ", text.replace("_", " ")).strip()
    return text.title() if text else "Unknown"


def student_from_filename(filename):
    """'Probin_2026-08-12_video1.mp4' -> 'Probin'   (name = text before the first _ - or digit)"""
    base = os.path.splitext(filename)[0]
    first = re.split(r"[_\-\d]", base)[0]
    return _clean_name(first)


def _video_date(item):
    """Use a date written in the file name (2026-08-12) if there is one, else the upload date."""
    m = re.search(r"(\d{4})[-_.](\d{2})[-_.](\d{2})", item["name"])
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return item["createdTime"][:10]


def scan_folder(drive, folder_id):
    """Return a list of dicts: id, file, student, date  (one per video found)."""
    videos = []

    def walk(fid, folder_name, depth):
        for item in _list_children(drive, fid):
            if item["mimeType"] == FOLDER_TYPE:
                if depth < 3:
                    walk(item["id"], item["name"], depth + 1)
            elif _is_video(item):
                in_subfolder = depth > 0
                use_folder = config.NAME_SOURCE == "folder" or (
                    config.NAME_SOURCE == "auto" and in_subfolder
                )
                student = _clean_name(folder_name) if use_folder and in_subfolder \
                    else student_from_filename(item["name"])
                videos.append({
                    "id": item["id"],
                    "file": item["name"],
                    "student": student,
                    "date": _video_date(item),
                })

    walk(folder_id, "", 0)
    return videos


def download_video(drive, file_id, file_name, student):
    """Download one video into videos/<student>/. Skips it if already there."""
    from googleapiclient.http import MediaIoBaseDownload

    safe_student = re.sub(r"[^\w\- ]", "_", student)
    safe_name = re.sub(r"[^\w.\- ]", "_", file_name)
    folder = config.VIDEOS_DIR / safe_student
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{file_id[:8]}_{safe_name}"
    if path.exists() and path.stat().st_size > 0:
        return path
    request = drive.files().get_media(fileId=file_id, supportsAllDrives=True)
    partial = path.with_name(path.name + ".part")
    with io.FileIO(partial, "wb") as f:
        downloader = MediaIoBaseDownload(f, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    os.replace(partial, path)      # only now is it a "real" finished file
    return path

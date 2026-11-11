"""
FRONTEND - only screens and buttons. All the real work happens in backend/service.py.
Run with: streamlit run app.py
"""
import pandas as pd
import streamlit as st

from backend import service
from backend.config import DEFAULT_DRIVE_FOLDER, METRICS, RAW_LABELS

st.set_page_config(page_title="Presentation Analyzer", page_icon="🎤", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
div[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08);
    padding: 14px 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


def num(x, spec=".0f", empty="—"):
    return empty if pd.isna(x) else format(x, spec)


# ============================================================ Sidebar
with st.sidebar:
    st.title("🎤 Presentation Analyzer")
    st.caption("MLA308 · NLP, Generative AI and GUI")
    page = st.radio("Menu", ["Class overview", "Student report", "Upload videos"],
                    label_visibility="collapsed")
    st.divider()
    st.subheader("Google Drive (optional)")
    folder = st.text_input("Folder link or ID", value=DEFAULT_DRIVE_FOLDER,
                           placeholder="Paste the Drive folder link")
    if st.button("Connect & scan", width="stretch", disabled=not folder):
        try:
            with st.spinner("Scanning the folder (the first time, a Google login opens)..."):
                found = service.scan_drive(folder)
            st.session_state["scan_msg"] = ("ok", f"Found {found} videos")
        except Exception as e:
            st.session_state["scan_msg"] = ("error", f"Couldn't scan the folder: {e}")
    msg = st.session_state.get("scan_msg")
    if msg:
        {"ok": st.success, "error": st.error}[msg[0]](msg[1])

    with st.expander("System check"):
        for name, ok, hint in service.system_check():
            st.write(f"{'✅' if ok else '❌'} {name}" + ("" if ok else f" — {hint}"))


# ============================================================ Page 1: Class overview
if page == "Class overview":
    st.header("Class overview")
    students = service.list_students()
    if students.empty:
        st.info("No students yet. Paste the Drive folder link in the sidebar and click **Connect & scan**, "
                "or add videos from your computer on the **Upload videos** page.")
        c1, c2, c3 = st.columns(3)
        c1.markdown("### 1. Add videos\nConnect the Drive folder, or upload videos from your computer.")
        c2.markdown("### 2. Analyze\nPick a student and the system studies their speech, voice and eye contact.")
        c3.markdown("### 3. Improve\nSee how much they have improved from their first video to now.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", len(students))
    c2.metric("Videos", int(students["Videos"].sum()))
    c3.metric("Videos analyzed", int(students["Analyzed"].sum()))
    avg = pd.to_numeric(students["Improvement"], errors="coerce").mean()
    c4.metric("Average improvement", "—" if pd.isna(avg) else f"{avg:+.1f} pts")

    query = st.text_input("Search student", placeholder="Type a name")
    shown = students[students["Student"].str.contains(query, case=False, na=False)]
    st.dataframe(
        shown, hide_index=True, width="stretch",
        column_config={
            "Overall score": st.column_config.ProgressColumn(
                "Overall score", min_value=0, max_value=100, format="%d"),
            "Improvement": st.column_config.NumberColumn("Improvement", format="%+.1f pts"),
        },
    )

    ranked = students.dropna(subset=["Improvement"])
    if not ranked.empty:
        st.subheader("Who improved the most")
        st.bar_chart(ranked.set_index("Student")["Improvement"])


# ============================================================ Page 2: Upload videos
elif page == "Upload videos":
    st.header("Upload videos")
    st.caption("Add videos from your computer instead of waiting for the Drive download. "
               "Use the same name for the same person and their videos join one report.")

    students = service.list_students()
    existing = students["Student"].tolist() if not students.empty else []
    NEW = "New student"
    choice = st.selectbox("Student", [NEW] + existing)
    name = st.text_input("Student name", placeholder="For example: Probin") if choice == NEW else choice

    files = st.file_uploader("Choose one or more videos", type=service.UPLOAD_TYPES,
                             accept_multiple_files=True)

    edited = None
    if files:
        st.write("Check the date of each video. It decides which video counts as the first and "
                 "which as the latest.")
        table = pd.DataFrame({
            "Video": [f.name for f in files],
            "Date": [pd.to_datetime(service.guess_date(f.name)).date() for f in files],
        })
        edited = st.data_editor(
            table, hide_index=True, width="stretch", disabled=["Video"],
            key="dates_" + str(abs(hash(tuple((f.name, f.size) for f in files)))),
            column_config={"Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD",
                                                               required=True)})

    analyze_now = st.checkbox("Analyze right after saving", value=True)
    ready = bool(files) and bool(name.strip())

    if st.button("Save videos", type="primary", disabled=not ready):
        try:
            uploads = [(f.name, f, str(d)) for f, d in zip(files, edited["Date"])]
            saved_name, ids = service.add_manual_videos(name, uploads)
        except Exception as e:
            st.error(f"Could not save the videos: {e}")
            st.stop()
        st.success(f"Saved {len(ids)} video(s) for {saved_name}.")

        if analyze_now:
            bar = st.progress(0.0, text="Starting...")
            try:
                summary = service.run_analysis(
                    saved_name, lambda p, t: bar.progress(min(p, 1.0), text=t), only_ids=ids)
                bar.empty()
                st.success(f"Analyzed {summary['analyzed']} video(s). Feedback: {summary['feedback']}. "
                           "Open **Student report** to see the results.")
                for file, why in summary["failed"]:
                    st.error(f"Could not analyze {file}: {why}")
                if summary["warnings"]:
                    with st.expander(f"{len(summary['warnings'])} warning(s)"):
                        for w in summary["warnings"]:
                            st.write("• " + w)
            except Exception as e:
                bar.empty()
                st.error(f"Something went wrong while analyzing: {e}")
        else:
            st.info("Saved. Open **Student report** and click **Analyze videos** when you are ready.")


# ============================================================ Page 3: Student report
else:
    st.header("Student report")
    students = service.list_students()
    if students.empty:
        st.info("No students yet. Connect the Drive folder in the sidebar, or use **Upload videos**.")
        st.stop()

    student = st.selectbox("Choose a student", students["Student"])
    row = students[students["Student"] == student].iloc[0]
    st.caption(f"{row['Videos']} videos on Drive · {row['Analyzed']} analyzed")

    force = st.checkbox("Re-analyze videos that were already done")
    local_ids = service.local_video_ids(student)
    only_local = False
    if local_ids:
        only_local = st.checkbox(f"Only analyze the {len(local_ids)} video(s) I uploaded "
                                 "(skip the Drive download)")

    if st.button("▶ Analyze videos", type="primary"):
        bar = st.progress(0.0, text="Starting...")
        try:
            summary = service.run_analysis(student, lambda p, t: bar.progress(min(p, 1.0), text=t),
                                           force=force, only_ids=local_ids if only_local else None)
            bar.empty()
            st.success(f"Analyzed {summary['analyzed']} new video(s), "
                       f"skipped {summary['skipped']} already done. "
                       f"Feedback: {summary['feedback']}.")
            for file, why in summary["failed"]:
                st.error(f"Could not analyze {file}: {why}")
            if summary["warnings"]:
                with st.expander(f"{len(summary['warnings'])} warning(s)"):
                    for w in summary["warnings"]:
                        st.write("• " + w)
        except Exception as e:
            bar.empty()
            st.error(f"Something went wrong: {e}")

    # NEW: recognize the video type for videos analyzed before (uses the saved transcripts, fast)
    if st.button("🏷 Detect video type for analyzed videos"):
        with st.spinner("Reading the saved transcripts..."):
            updated = service.classify_saved_videos(student)
        st.success(f"Updated {updated} video(s).")

    report = service.get_report(student)
    if not report:
        st.info(f"No results for {student} yet. Click **Analyze videos**.")
        st.stop()

    table, videos = report["table"], report["videos"]

    st.subheader("Where they are now")
    cols = st.columns(len(table))
    for col, (_, r) in zip(cols, table.iterrows()):
        delta = None if pd.isna(r["Change"]) else f"{r['Change']:+.0f} since start"
        col.metric(r["Skill"], num(r["Now"]), delta)

    # NEW: which kinds of video this student has made
    if "video_type" in videos:
        kinds = videos["video_type"].dropna().astype(str)
        kinds = kinds[kinds.str.strip() != ""]
        if not kinds.empty:
            st.caption("Video types: " + ", ".join(f"{k} ×{n}" for k, n in kinds.value_counts().items()))

    st.subheader("Progress over time")
    chart = videos.set_index("date")[METRICS + ["Overall"]].astype(float)
    st.line_chart(chart)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Feedback", "Improvement table", "Video details", "Raw measurements",
         f"All videos ({len(videos)})"])

    with tab1:
        st.info(report["feedback"])

    with tab2:
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Start = average of the first 3 videos · Now = average of the last 3 · "
                   "Trend = points gained per week across all videos.")

    with tab3:
        pick = st.selectbox("Choose a video", videos["video"].tolist()[::-1])
        v = videos[videos["video"] == pick].iloc[0]
        c = st.columns(len(METRICS) + 1)
        for col, name in zip(c, METRICS + ["Overall"]):
            col.metric(name, num(v[name]))

        # NEW: what kind of video it is and what it is about
        vtype = v.get("video_type")
        if isinstance(vtype, str) and vtype.strip():
            conf = v.get("type_confidence")
            sure = "" if pd.isna(conf) else f" ({conf:.0f}% sure)"
            st.markdown(f"**Video type:** {vtype}{sure}  \n**Topic:** {v.get('topic', '')}")
            summary_text = v.get("video_summary")
            if isinstance(summary_text, str) and summary_text.strip():
                st.write("**What the video is about:** " + summary_text)

        if isinstance(v["content_comment"], str) and v["content_comment"]:
            st.write("**Content comment:** " + v["content_comment"])
        with st.expander("Transcript"):
            st.write(service.get_transcript(v["id"]) or "No transcript saved.")

    with tab4:
        raw_cols = [c for c in RAW_LABELS if c in videos and videos[c].notna().any()]
        if raw_cols:
            metric = st.selectbox("Measurement", raw_cols, format_func=lambda c: RAW_LABELS[c])
            st.line_chart(videos.set_index("date")[metric].astype(float))
        else:
            st.write("No measurements yet.")

    with tab5:
        show_cols = (["video", "date"] + (["video_type"] if "video_type" in videos else [])
                     + METRICS + ["Overall"])
        show = videos[show_cols].copy()
        show["date"] = show["date"].dt.date
        st.dataframe(show.round(1), hide_index=True, width="stretch")

"""
Analyze videos from the command line, WITHOUT the web app.
Handy to run before the demo, because the first analysis of 20+ videos takes a while
(results are saved, so the app then opens instantly).

    python analyze_all.py                         # every student
    python analyze_all.py --student               # one student
    python analyze_all.py --scan "<drive link>"   # scan Drive first
    python analyze_all.py --force                 # redo videos already analyzed
"""
import argparse

from backend import service

parser = argparse.ArgumentParser()
parser.add_argument("--student", help="only this student")
parser.add_argument("--scan", metavar="LINK", help="scan this Drive folder first")
parser.add_argument("--force", action="store_true", help="re-analyze videos already done")
args = parser.parse_args()

if args.scan:
    print(f"Found {service.scan_drive(args.scan)} videos on Drive.")

students = service.list_students()["Student"].tolist()
if args.student:
    students = [s for s in students if s.lower() == args.student.lower()]
if not students:
    raise SystemExit("No students found. Use --scan with the Drive folder link first.")

for name in students:
    print(f"\n=== {name} ===")
    last = {"text": ""}

    def show(p, text):
        if text != last["text"]:
            print(f"  {p * 100:3.0f}%  {text}")
            last["text"] = text

    summary = service.run_analysis(name, show, force=args.force)
    print(f"  analyzed {summary['analyzed']}, skipped {summary['skipped']}, "
          f"feedback: {summary['feedback']}")
    for file, why in summary["failed"]:
        print(f"  FAILED {file}: {why}")
    for w in summary["warnings"]:
        print(f"  warning: {w}")

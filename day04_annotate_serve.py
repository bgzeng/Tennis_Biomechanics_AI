"""Day 4 — manually label one tennis serve's events and phase windows.

Run from the project root (keep this script there):
    python day04_annotate_serve.py

Keys: a/d = -/+1 frame, j/l = -/+10, z/x = -/+50,
      0 = frame 0, 9 = frame 233, 1..5 = mark events, s = save, q = quit.
"""

import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "data" / "serve.mp4"
OUTPUT = ROOT / "outputs"
EVENTS_CSV = OUTPUT / "serve_events.csv"
PHASES_CSV = OUTPUT / "serve_phases.csv"

EVENTS = [
    ("1", "toss_initiation"),
    ("2", "trophy_position"),
    ("3", "racket_low_point"),
    ("4", "ball_contact"),
    ("5", "follow_through_end"),
]


def read_existing_marks():
    marks = {}
    if EVENTS_CSV.exists():
        with EVENTS_CSV.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["event"] in [name for _, name in EVENTS]:
                    marks[row["event"]] = int(row["frame"])
    return marks


def save_labels(marks, fps, frame_count):
    event_names = [name for _, name in EVENTS]
    if any(name not in marks for name in event_names):
        print("Not saved: please mark all five events (keys 1–5).")
        return False

    event_frames = [marks[name] for name in event_names]
    if not all(a < b for a, b in zip(event_frames, event_frames[1:])):
        print("Not saved: events must be in chronological order (1 < 2 < 3 < 4 < 5).")
        print(dict(zip(event_names, event_frames)))
        return False

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with EVENTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event", "frame", "time_s", "method", "review_status"])
        for name in event_names:
            frame = marks[name]
            writer.writerow([name, frame, f"{frame / fps:.3f}", "manual_visual", "provisional"])

    toss, trophy, racket_low, contact, finish = event_frames
    with PHASES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "phase", "include_in_serve_analysis"])
        for frame in range(frame_count):
            if frame < toss:
                phase = "pre_serve"
            elif frame < trophy:
                phase = "toss_loading"
            elif frame < racket_low:
                phase = "trophy_to_racket_low_point"
            elif frame < contact:
                phase = "forward_swing"
            elif frame <= finish:
                phase = "follow_through"
            else:
                phase = "post_serve"
            include = int(toss <= frame <= finish)
            writer.writerow([frame, f"{frame / fps:.3f}", phase, include])

    print(f"Saved: {EVENTS_CSV}")
    print(f"Saved: {PHASES_CSV}")
    print(f"Serve interval: frames {toss}–{finish} ({toss / fps:.2f}–{finish / fps:.2f}s)")
    print("IMPORTANT: These are provisional visual events, not manual 2D joint-angle references.")
    return True


def main():
    if not VIDEO.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO}")

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError("Video FPS or frame count is invalid.")

    print(f"Video: {fps:.2f} FPS, {frame_count} frames")
    print("Keys: a/d ±1; j/l ±10; z/x ±50; 0 start; 9 frame 233;")
    print("      1=toss, 2=trophy, 3=racket low point, 4=contact, 5=finish; s=save; q=quit")
    print("Frame 233 was identified as rapid racket-drop motion, NOT automatically an error.")

    marks = read_existing_marks()
    current = min(233, frame_count - 1)

    try:
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current)
            ok, frame = cap.read()
            if not ok:
                print(f"Unable to read frame {current}")
                break

            height, width = frame.shape[:2]
            scale = min(1150 / width, 690 / height, 1.0)
            display = cv2.resize(frame, (round(width * scale), round(height * scale)))
            h, w = display.shape[:2]
            canvas = np.zeros((h + 190, w, 3), dtype=np.uint8)
            canvas[:h, :w] = display

            def line(msg, row):
                cv2.putText(canvas, msg, (12, h + 27 + row * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1, cv2.LINE_AA)

            line(f"Frame {current}/{frame_count - 1} | {current / fps:.3f} s | FPS {fps:.2f}", 0)
            line("a/d:1  j/l:10  z/x:50  0:start  9:233  s:save  q:quit", 1)
            for idx, (key, name) in enumerate(EVENTS):
                marked = marks.get(name)
                label = "not marked" if marked is None else f"frame {marked} ({marked / fps:.2f}s)"
                line(f"{key} {name}: {label}", idx + 2)

            cv2.imshow("Day 4 | Manual tennis serve events", canvas)
            key = cv2.waitKey(0) & 0xFF

            steps = {ord("a"): -1, ord("d"): 1, ord("j"): -10,
                     ord("l"): 10, ord("z"): -50, ord("x"): 50}
            if key in steps:
                current = max(0, min(frame_count - 1, current + steps[key]))
            elif key == ord("0"):
                current = 0
            elif key == ord("9"):
                current = min(233, frame_count - 1)
            elif chr(key) in dict(EVENTS):
                name = dict(EVENTS)[chr(key)]
                marks[name] = current
                print(f"Marked {name}: frame {current}, time {current / fps:.3f}s")
            elif key == ord("s"):
                save_labels(marks, fps, frame_count)
            elif key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

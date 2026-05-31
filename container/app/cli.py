import sys
import json
import csv
from pathlib import Path

sys.path.insert(0, "/app/app")
from detector import CatDetector

STUDENT_PATH = Path("/app/STUDENT.json")
MODEL_PATH   = Path("/app/models/best.onnx")
INPUT_DIR    = Path("/data/input")
OUTPUT_DIR   = Path("/data/output")
IMG_EXTS     = {".jpg", ".jpeg", ".png"}


def cmd_info():
    print(STUDENT_PATH.read_text())


def cmd_predict():
    detector = CatDetector(str(MODEL_PATH))

    image_paths = sorted([
        p for p in INPUT_DIR.rglob("*")
        if p.suffix.lower() in IMG_EXTS
    ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "predictions.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "xmin", "ymin", "xmax", "ymax", "confidence", "class"])

        for img_path in image_paths:
            rel_path = img_path.relative_to(INPUT_DIR).as_posix()
            detections = detector.predict(str(img_path))

            if not detections:
                writer.writerow([rel_path, "", "", "", "", "", ""])
            else:
                for d in detections:
                    writer.writerow([
                        rel_path,
                        round(d["xmin"], 2),
                        round(d["ymin"], 2),
                        round(d["xmax"], 2),
                        round(d["ymax"], 2),
                        round(d["confidence"], 4),
                        d["class"],
                    ])

    print(f"predictions.csv written -> {out_csv}  ({len(image_paths)} images processed)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cli.py [info|predict]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "info":
        cmd_info()
    elif cmd == "predict":
        cmd_predict()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

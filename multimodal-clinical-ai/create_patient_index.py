from datasets import load_dataset
from pathlib import Path
import pandas as pd
import re
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
DATASET_NAME = "itsanmolgupta/mimic-cxr-dataset"   # change this
SPLIT = "train"

IMAGE_DIR = Path("data/raw/images")
OUTPUT_CSV = Path("data/metadata/patient_index.csv")

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LABEL DEFINITIONS
# -----------------------------
LABEL_KEYWORDS = {
    "PNEUMOTHORAX": ["pneumothorax"],
    "PNEUMONIA": ["pneumonia", "consolidation", "airspace disease"],
    "EDEMA": ["pulmonary edema", "vascular congestion"],
    "EFFUSION": ["pleural effusion"],
    "CARDIOMEGALY": ["cardiomegaly", "enlarged heart"],
    "NORMAL": [
        "no acute cardiopulmonary",
        "no acute abnormality",
        "no acute disease",
        "normal chest",
        "unremarkable"
    ]
}

PRIORITY = [
    "PNEUMOTHORAX",
    "PNEUMONIA",
    "EDEMA",
    "EFFUSION",
    "CARDIOMEGALY",
    "NORMAL"
]


def assign_label(impression: str) -> str:
    if not isinstance(impression, str):
        return "OTHER"

    text = impression.lower()
    text = re.sub(r"[^\w\s]", " ", text)

    for label in PRIORITY:
        for kw in LABEL_KEYWORDS[label]:
            if kw in text:
                return label

    return "OTHER"


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    print("📥 Loading Hugging Face dataset...")
    dataset = load_dataset(DATASET_NAME, split=SPLIT)

    records = []

    for idx, sample in enumerate(dataset):
        image = sample["image"]
        findings = sample["findings"]
        impression = sample["impression"]

        if image is None or findings is None or impression is None:
            continue

        # Save image locally (important for PyTorch Dataset later)
        image_path = IMAGE_DIR / f"img_{idx}.png"
        if not image_path.exists():
            image.save(image_path)

        label = assign_label(impression)

        records.append({
            "image_path": str(image_path),
            "findings": findings,
            "impression": impression,
            "label": label
        })

        if idx % 1000 == 0:
            print(f"Processed {idx} samples...")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n✅ patient_index.csv created")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()

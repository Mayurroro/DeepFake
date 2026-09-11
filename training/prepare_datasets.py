"""Crawl d:\\VIT\\Datasets and Hugging Face to produce train/test file lists for image and audio."""
import os, glob, random

DATASETS = r"d:\VIT\Datasets"
OUT = os.path.join(os.path.dirname(__file__))
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

HF_DATASET = "saakshigupta/deepfake-detection-dataset-v3"


def _walk_images(root):
    entries = []
    for sub in ("fake", "real"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        label = 2 if sub == "fake" else 0
        for r, _, fs in os.walk(d):
            for f in fs:
                if os.path.splitext(f)[1].lower() in IMG_EXTS:
                    entries.append((os.path.join(r, f), label))
    return entries


def crawl_images():
    base = os.path.join(DATASETS, "DeepFake images")
    train = _walk_images(os.path.join(base, "train-20250112T065955Z-001", "train"))
    # also include sample fakes in train
    sf = os.path.join(base, "Sample_fake_images", "Sample_fake_images", "fake")
    if os.path.isdir(sf):
        for r, _, fs in os.walk(sf):
            for f in fs:
                if os.path.splitext(f)[1].lower() in IMG_EXTS:
                    train.append((os.path.join(r, f), 2))
    test = _walk_images(os.path.join(base, "test-20250112T065939Z-001", "test"))
    for name, data in [("image_train_list.txt", train), ("image_test_list.txt", test)]:
        p = os.path.join(OUT, name)
        with open(p, "w") as fp:
            for path, lbl in data:
                fp.write(f"{path} {lbl}\n")
        print(f"[{name}] {len(data)} entries")


def crawl_audio():
    audio_root = os.path.join(DATASETS, "DeepFake audio")
    all_wavs = []
    for sub in ("fake", "real"):
        d = os.path.join(audio_root, sub)
        if not os.path.isdir(d):
            continue
        label = 2 if sub == "fake" else 0
        for f in os.listdir(d):
            if f.lower().endswith(".wav"):
                all_wavs.append((os.path.join(d, f), label))
    random.seed(42)
    random.shuffle(all_wavs)
    split = int(len(all_wavs) * 0.8)
    for name, data in [("audio_train_list.txt", all_wavs[:split]), ("audio_test_list.txt", all_wavs[split:])]:
        p = os.path.join(OUT, name)
        with open(p, "w") as fp:
            for path, lbl in data:
                fp.write(f"{path} {lbl}\n")
        print(f"[{name}] {len(data)} entries")


def crawl_huggingface(cache_root=None):
    """Download saakshigupta/deepfake-detection-dataset-v3 and write binary (real=0/fake=1) list files.

    The dataset has two splits (train/test) with a two-class `label` feature.
    Images are cached to disk so the existing ListDataset/training flow is reused unchanged.
    """
    from datasets import load_dataset

    if cache_root is None:
        cache_root = os.path.join(DATASETS, "DeepFake-HF-v3")

    ds = load_dataset(HF_DATASET)
    names = list(ds["train"].features["label"].names)
    real_idx = names.index("real")
    fake_idx = names.index("fake")

    print(f"[HF] {HF_DATASET} splits: {list(ds.keys())}, labels: {names}")

    for split, list_name in (("train", "image_train_list.txt"), ("test", "image_test_list.txt")):
        split_dir = os.path.join(cache_root, split)
        os.makedirs(split_dir, exist_ok=True)
        entries = []
        for i, row in enumerate(ds[split]):
            lbl = 1 if row["label"] == fake_idx else 0
            sub = "fake" if lbl else "real"
            d = os.path.join(split_dir, sub)
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, f"{i:05d}.jpg")
            if not os.path.exists(p):
                row["image"].convert("RGB").save(p, "JPEG", quality=95)
            entries.append((p, lbl))
        with open(os.path.join(OUT, list_name), "w") as fp:
            for path, lbl in entries:
                fp.write(f"{path} {lbl}\n")
        print(f"[HF {list_name}] {len(entries)} entries -> {os.path.join(OUT, list_name)}")


if __name__ == "__main__":
    print("=" * 50)
    crawl_images()
    crawl_audio()
    try:
        crawl_huggingface()
        print("Hugging Face dataset lists written "
              "(binary labels: real=0, fake=1).")
    except Exception as e:
        print(f"[WARN] HF crawl failed ({e}); keeping local image lists.")
    print("Done!")

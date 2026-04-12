"""Crawl d:\\VIT\\Datasets to produce train/test file lists for image and audio."""
import os, glob, random

DATASETS = r"d:\VIT\Datasets"
OUT = os.path.join(os.path.dirname(__file__))
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


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


if __name__ == "__main__":
    print("=" * 50)
    crawl_images()
    crawl_audio()
    print("Done!")

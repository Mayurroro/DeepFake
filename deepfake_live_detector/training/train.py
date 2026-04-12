"""
Unified Training & Evaluation Script for Deepfake Live Detector.
Covers: dataset preparation -> training -> evaluation -> model saving.

Usage (from project root):
    python training/train.py --mode image --epochs 30 --batch 32
"""
import os
import sys
import time
import argparse
import traceback
from pathlib import Path

# Configure stdout for UTF-8 and Unbuffered output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Global imports for models and utils (moved out of function to detect import errors early)
try:
    from models.image_detector import ImageDetector
    from models.audio_detector import AudioDetector
    from utils.feature_extractors import preprocess_image, extract_audio_features
except ImportError as e:
    print(f"[FATAL ERROR] Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

WEIGHTS = ROOT / "weights"
WEIGHTS.mkdir(exist_ok=True)


# ===================================================================
# Focal Loss
# ===================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, x, y):
        ce = self.ce(x, y)
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


# ===================================================================
# Dataset
# ===================================================================
class ListDataset(Dataset):
    """Reads a text file with lines: <absolute_path> <label_int>"""

    def __init__(self, list_file, mode="image"):
        self.mode = mode
        self.samples = []
        if not os.path.exists(list_file):
            print(f"  [ERROR] File not found: {list_file}")
            return
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.rsplit(None, 1)
                if len(parts) == 2:
                    self.samples.append((parts[0], int(parts[1])))
        print(f"  Loaded {len(self.samples)} samples from {list_file}", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            if self.mode == "image":
                rgb, freq, edge = preprocess_image(path)
                return rgb, freq, edge, label
            else:
                mel, aux = extract_audio_features(path)
                return mel, aux, label
        except Exception as e:
            # Return zero tensors on error so training doesn't crash
            if self.mode == "image":
                return torch.zeros(3, 512, 512), torch.zeros(1, 512, 512), torch.zeros(1, 512, 512), label
            else:
                return torch.zeros(1, 128, 400), torch.zeros(30), label


# ===================================================================
# Step 1: Prepare Datasets
# ===================================================================
def prepare_datasets():
    """Generate train/test list files from d:\\VIT\\Datasets."""
    try:
        from training.prepare_datasets import crawl_images, crawl_audio
        print("-" * 60, flush=True)
        print("  Step 1: Preparing dataset lists...", flush=True)
        print("-" * 60, flush=True)
        crawl_images()
        crawl_audio()
        print("  Dataset preparation complete.", flush=True)
    except Exception as e:
        print(f"[ERROR] during dataset preparation: {e}", flush=True)
        traceback.print_exc()


# ===================================================================
# Step 2: Train
# ===================================================================
def train_model(list_file, mode="image", epochs=30, batch_size=32, lr=1e-4):
    # Device detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("-" * 60, flush=True)
    print(f"  Step 2: Training -- mode={mode}, device={device}, epochs={epochs}", flush=True)
    print("-" * 60, flush=True)

    ds = ListDataset(list_file, mode)
    if len(ds) == 0:
        print(f"  [ERROR] No samples found in {list_file}. Skipping.", flush=True)
        return

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    print("  Initializing model...", flush=True)
    try:
        if mode == "image":
            model = ImageDetector(pretrained=True).to(device)
            save_name = "image_detector.pth"
        else:
            model = AudioDetector().to(device)
            save_name = "audio_detector.pth"
        print("  Model ready.", flush=True)
    except Exception as e:
        print(f"  [FATAL] Model initialization failed: {e}", flush=True)
        traceback.print_exc()
        return

    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    scaler = None
    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except:
             scaler = None

    best_acc = 0.0
    save_path = WEIGHTS / save_name

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        print(f"\n  Epoch {ep}/{epochs} starting...", flush=True)
        
        for i, batch in enumerate(loader):
            try:
                # DEBUG: Periodic batch check
                if i % 10 == 0:
                    print(f"    Processing batch {i+1}/{len(loader)}...", flush=True)

                if mode == "image":
                    rgb, freq, edge, lbl = [b.to(device) for b in batch]
                else:
                    mel, aux, lbl = batch[0].to(device), batch[1].to(device), batch[2].to(device)

                optimizer.zero_grad()

                if scaler:
                    with torch.amp.autocast("cuda"):
                        logits = model(rgb, freq, edge)[0] if mode == "image" else model(mel, aux)[0]
                        loss = criterion(logits, lbl)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Generic autocast for MPS/CPU if available, or just plain
                    logits = model(rgb, freq, edge)[0] if mode == "image" else model(mel, aux)[0]
                    loss = criterion(logits, lbl)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * lbl.size(0)
                correct += logits.argmax(1).eq(lbl).sum().item()
                total += lbl.size(0)
                
            except Exception as e:
                print(f"\n  [ERROR] Batch {i+1} failed: {e}", flush=True)
                traceback.print_exc()
                # We continue to next batch instead of crashing the whole epoch

        scheduler.step()
        epoch_acc = correct / total if total > 0 else 0
        avg_loss = total_loss / total if total > 0 else 0
        print(f"  Epoch {ep} summary -> Loss: {avg_loss:.4f} Acc: {epoch_acc:.2%}", flush=True)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f"  [SAVED] Best weights -> {save_path}", flush=True)

    final_path = WEIGHTS / f"{mode}_detector_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n  Training {mode} complete. Best Accuracy: {best_acc:.2%}", flush=True)
    return model, save_path


def evaluate_model(test_list, mode="image"):
    # Device detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("-" * 60, flush=True)
    print(f"  Step 3: Evaluating -- mode={mode}", flush=True)
    print("-" * 60, flush=True)

    if not os.path.exists(test_list):
        print(f"  [WARN] Test list not found: {test_list}. Skipping.", flush=True)
        return

    ds = ListDataset(test_list, mode)
    if len(ds) == 0: return
    
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    try:
        if mode == "image":
            model = ImageDetector(pretrained=False).to(device)
            w = WEIGHTS / "image_detector.pth"
        else:
            model = AudioDetector().to(device)
            w = WEIGHTS / "audio_detector.pth"

        if w.exists():
            model.load_state_dict(torch.load(w, map_location=device))
            print(f"  Loaded weights: {w}", flush=True)
        else:
            print(f"  [WARN] No weights found at {w}. Evaluating random init.", flush=True)

        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                if mode == "image":
                    rgb, freq, edge, lbl = [b.to(device) for b in batch]
                    logits = model(rgb, freq, edge)[0]
                else:
                    mel, aux, lbl = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    logits = model(mel, aux)[0]

                preds = logits.argmax(1)
                correct += preds.eq(lbl).sum().item()
                total += lbl.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(lbl.cpu().tolist())

        acc = correct / total if total > 0 else 0
        print(f"\n  Test Accuracy: {acc:.2%} ({correct}/{total})", flush=True)

        classes = ["REAL", "MANIPULATED", "AI_GENERATED"]
        for cls_idx, cls_name in enumerate(classes):
            cls_total = sum(1 for l in all_labels if l == cls_idx)
            cls_correct = sum(1 for p, l in zip(all_preds, all_labels) if l == cls_idx and p == cls_idx)
            if cls_total > 0:
                print(f"  {cls_name}: {cls_correct}/{cls_total} ({cls_correct / cls_total:.2%})", flush=True)

    except Exception as e:
        print(f"  [ERROR] Evaluation failed: {e}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Deepfake Live Detector -- Train & Evaluate")
    p.add_argument("--mode", default="all", choices=["image", "audio", "all"],
                   help="What to train: image, audio, or all (default: all)")
    p.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    p.add_argument("--batch", type=int, default=32, help="Batch size (default: 32)")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    p.add_argument("--skip-prepare", action="store_true", help="Skip dataset preparation")
    p.add_argument("--eval-only", action="store_true", help="Only evaluate, don't train")
    a = p.parse_args()

    here = Path(__file__).resolve().parent

    try:
        if not a.skip_prepare and not a.eval_only:
            prepare_datasets()

        for m in (["image", "audio"] if a.mode == "all" else [a.mode]):
            train_list = here / f"{m}_train_list.txt"
            test_list = here / f"{m}_test_list.txt"

            if not a.eval_only:
                if train_list.exists():
                    train_model(str(train_list), m, a.epochs, a.batch, a.lr)
                else:
                    print(f"  [WARN] {train_list} not found -- skipping {m} training.", flush=True)

            if test_list.exists():
                evaluate_model(str(test_list), m)

        print("\n" + "=" * 60, flush=True)
        print("  ALL TASKS FINISHED!", flush=True)
        print(f"  Check weights directory: {WEIGHTS}", flush=True)
        print("=" * 60, flush=True)
    except Exception as e:
        print(f"\n[FATAL ERROR]: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

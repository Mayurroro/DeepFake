import torch
import sys
from pathlib import Path

ROOT = Path("d:/VIT/deepfake_live_detector").resolve()
sys.path.insert(0, str(ROOT))

try:
    print("Testing ImageDetector loading...")
    from models.image_detector import ImageDetector
    model = ImageDetector(pretrained=True).to("cpu")
    print("Model loaded.")

    print("Testing dataset sample...")
    from training.train import ListDataset
    train_list = ROOT / "training/image_train_list.txt"
    ds = ListDataset(str(train_list), mode="image")
    
    if len(ds) == 0:
        print("Dataset is empty!")
        sys.exit(1)
        
    sample = ds[0]
    rgb, freq, edge, label = sample
    print(f"Sample shapes: rgb={rgb.shape}, freq={freq.shape}, edge={edge.shape}, label={label}")

    print("Testing forward pass with batch size 4...")
    rgb_batch = rgb.unsqueeze(0).repeat(4, 1, 1, 1)
    freq_batch = freq.unsqueeze(0).repeat(4, 1, 1, 1)
    edge_batch = edge.unsqueeze(0).repeat(4, 1, 1, 1)
    
    logits, _ = model(rgb_batch, freq_batch, edge_batch)
    print(f"Logits shape: {logits.shape}")

    print("Testing loss and backward...")
    from training.train import FocalLoss
    import torch.optim as optim
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    loss = criterion(logits, torch.tensor([label]))
    print(f"Loss: {loss.item()}")
    
    loss.backward()
    print("Backward pass successful.")
    
    optimizer.step()
    print("Optimization step successful.")
    
    print("Success!")

except Exception as e:
    import traceback
    print("\nCRASH DETECTED:")
    traceback.print_exc()
    sys.exit(1)

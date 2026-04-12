"""Visualization helpers — Grad-CAM heatmaps and confidence gauge rendering."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def grad_cam(model, input_tensor, target_layer, class_idx=1):
    """Generate a Grad-CAM heatmap (H×W float32 in [0,1])."""
    grads, acts = [], []
    target_layer.register_forward_hook(lambda m, i, o: acts.append(o.detach()))
    target_layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0].detach()))

    model.eval()
    model.zero_grad()
    input_tensor.requires_grad_(True)
    out = model(input_tensor)
    if isinstance(out, tuple):
        out = out[0]
    out[0, class_idx].backward()

    if not grads or not acts:
        return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

    w = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts[0]).sum(1)).squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))


def overlay_heatmap(image_np, heatmap, alpha=0.45):
    """Overlay a [0,1] heatmap on an RGB image."""
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_np, 1 - alpha, colored, alpha, 0)


def draw_confidence_bar(image_np, confidence, label, color=(0, 255, 120)):
    """Draw a confidence bar and label at the top of an image (in-place)."""
    h, w = image_np.shape[:2]
    bar_h = 36
    cv2.rectangle(image_np, (0, 0), (w, bar_h), (20, 20, 20), -1)
    bar_w = int(w * confidence)
    cv2.rectangle(image_np, (0, 0), (bar_w, bar_h), color, -1)
    text = f"{label}  {confidence:.1%}"
    cv2.putText(image_np, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image_np

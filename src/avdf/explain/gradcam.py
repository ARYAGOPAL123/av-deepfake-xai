"""Grad-CAM through the fused model: gradients of the fused 'fake' logit w.r.t. the visual target layer."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .. import FAKE


def gradcam_frames(model, faces, wav, target=FAKE):
    """faces: (1,T,3,H,W) normalised, wav: (1,N). Returns (T,H,W) heatmaps in [0,1] and the fake probability."""
    assert model.visual is not None, "Grad-CAM needs a visual branch"
    layer = model.visual.target_layer()
    store = {}

    def hook(module, inputs, output):
        # Hand a clone downstream so in-place activations after this layer (e.g. MobileNetV3's
        # hardswish) cannot corrupt the stored activation; its gradient equals d(logit)/d(output).
        out = output.clone()
        store["act"] = output.detach()
        out.register_hook(lambda g: store.__setitem__("grad", g))
        return out

    h = layer.register_forward_hook(hook)
    model.eval()
    faces = faces.clone().requires_grad_(True)       # ensures gradients flow even when early blocks are frozen
    try:
        with torch.enable_grad():
            logits, _ = model(faces, wav)
            model.zero_grad()
            logits[0, target].backward()
    finally:
        h.remove()
    act, grad = store["act"].detach(), store["grad"].detach()
    if act.dim() == 3:                               # ViT tokens (T, 1+N, C) -> (T, C, h, w)
        n = int((act.shape[1] - 1) ** 0.5)
        act = act[:, 1:].reshape(act.shape[0], n, n, -1).permute(0, 3, 1, 2)
        grad = grad[:, 1:].reshape(grad.shape[0], n, n, -1).permute(0, 3, 1, 2)
    w = grad.mean(dim=(2, 3), keepdim=True)          # alpha_k (Eq. 4.2)
    cam = F.relu((w * act).sum(1, keepdim=True))     # L = ReLU(sum alpha_k A^k) (Eq. 4.3)
    cam = F.interpolate(cam, size=faces.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return cam.cpu().numpy(), float(logits.detach().softmax(-1)[0, FAKE])


def denorm(face):
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    img = face.permute(1, 2, 0).cpu().numpy() * std + mean
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def overlay(face, cam, alpha=0.45):
    """face: (3,H,W) normalised tensor, cam: (H,W). Returns RGB uint8 overlay."""
    img = denorm(face)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return (alpha * heat + (1 - alpha) * img).astype(np.uint8)


def save_gradcam_grid(faces, cams, path, k=6):
    """Save the k frames with highest activation side by side."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = np.argsort(-cams.reshape(len(cams), -1).mean(1))[:k]
    fig, ax = plt.subplots(1, len(idx), figsize=(2.2 * len(idx), 2.4))
    for a, i in zip(np.atleast_1d(ax), sorted(idx)):
        a.imshow(overlay(faces[0, i], cams[i])); a.set_title(f"frame {i}", fontsize=9); a.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path

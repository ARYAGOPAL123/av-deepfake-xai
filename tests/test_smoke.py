import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avdf.explain import gradcam_frames  # noqa: E402
from avdf.metrics import classification_metrics, ece  # noqa: E402
from avdf.models import AVDeepfakeDetector  # noqa: E402
from avdf.uncertainty import evidential_probs, predict_with_uncertainty, route  # noqa: E402


def tiny(mode="fusion"):
    return AVDeepfakeDetector(mode=mode, visual="efficientnet_b0", visual_pretrained=False, audio="cnn", d=64, heads=4).eval()


def batch(B=2, T=4):
    return torch.randn(B, T, 3, 224, 224), torch.randn(B, 16000 * 2)


def test_forward_all_modes():
    f, w = batch()
    for mode in ["fusion", "concat", "video_only", "audio_only"]:
        logits, _ = tiny(mode)(f, w)
        assert logits.shape == (2, 2)


def test_mc_dropout_and_routing():
    f, w = batch()
    o = predict_with_uncertainty(tiny(), f, w, "mc_dropout", T=4)
    assert o["prob"].shape == (2, 2) and torch.allclose(o["prob"].sum(-1), torch.ones(2), atol=1e-5)
    pred, conf, review = route(o["prob"], o["uncertainty"], tau=0.99, u_max=0.0)
    assert review.all()          # impossible thresholds -> everything goes to review


def test_evidential():
    p, u = evidential_probs(torch.tensor([[0.0, 0.0], [10.0, -10.0]]))
    assert u[0] > u[1]           # more evidence -> less uncertainty


def test_gradcam():
    f, w = batch(1)
    cam, p = gradcam_frames(tiny(), f, w)
    assert cam.shape == (4, 224, 224) and 0 <= p <= 1


def test_metrics():
    y = np.array([0, 1, 1, 0]); p = np.array([0.1, 0.9, 0.8, 0.3])
    m = classification_metrics(y, p)
    assert m["accuracy"] == 1.0 and m["auc"] == 1.0
    assert ece(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.0

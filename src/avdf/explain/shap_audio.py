"""SHAP for the audio stream: KernelSHAP over (Mel-scale band x time segment) groups.

Groups are masked in the STFT domain (magnitude set to zero, original phase kept) and resynthesised,
so the same procedure works for SSL (waveform) and log-Mel CNN encoders.
"""
import numpy as np
import torch

from .. import FAKE

N_FFT, HOP = 400, 160


def _band_edges(n_bins, n_bands, sr=16000):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges_hz = imel(np.linspace(mel(0), mel(sr / 2), n_bands + 1))
    return np.clip(np.round(edges_hz / (sr / 2) * (n_bins - 1)).astype(int), 0, n_bins)


def shap_audio(model, faces, wav, n_bands=8, n_segs=10, nsamples=300, batch=16):
    """faces (1,T,3,H,W), wav (1,N). Returns phi (n_bands, n_segs), base value, band edges (Hz)."""
    import shap
    dev = wav.device
    win = torch.hann_window(N_FFT, device=dev)
    spec = torch.stft(wav[0], N_FFT, HOP, window=win, return_complex=True)      # (F, Tf)
    Fb, Tf = spec.shape
    be = _band_edges(Fb, n_bands)
    te = np.linspace(0, Tf, n_segs + 1).astype(int)
    model.eval()
    with torch.no_grad():
        V = model.visual(faces) if model.visual is not None else None

    def f(Z):
        outs = []
        for s in range(0, len(Z), batch):
            wavs = []
            for z in Z[s:s + batch]:
                m = torch.ones(Fb, Tf, device=dev)
                for g, keep in enumerate(z):
                    if keep < 0.5:
                        b, t = divmod(g, n_segs)
                        m[be[b]:be[b + 1], te[t]:te[t + 1]] = 0
                wavs.append(torch.istft(spec * m, N_FFT, HOP, window=win, length=wav.shape[1]))
            W = torch.stack(wavs)
            W = (W - W.mean(1, keepdim=True)) / (W.std(1, keepdim=True) + 1e-6)
            with torch.no_grad():
                A = model.audio(W)
                Vb = V.expand(len(W), -1, -1) if V is not None else None
                logits, _ = model.fuse(Vb, A)
            outs.append(logits.float().softmax(-1)[:, FAKE].cpu().numpy())
        return np.concatenate(outs)

    G = n_bands * n_segs
    expl = shap.KernelExplainer(f, np.zeros((1, G)))
    phi = np.asarray(expl.shap_values(np.ones((1, G)), nsamples=nsamples, silent=True)).reshape(n_bands, n_segs)
    hz = be / (Fb - 1) * 8000
    return phi, float(np.ravel(expl.expected_value)[0]), hz


def plot_shap(phi, hz, path, clip_sec=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1, 1.6]})
    labels = [f"{int(hz[i])}-{int(hz[i+1])} Hz" for i in range(len(hz) - 1)]
    band = phi.sum(1)
    ax[0].barh(labels, band, color=["#D94A4A" if v > 0 else "#0E9F8E" for v in band])
    ax[0].set_title("Band importance (+ pushes to FAKE)", fontsize=10); ax[0].invert_yaxis()
    lim = np.abs(phi).max() + 1e-9
    im = ax[1].imshow(phi, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim, origin="upper")
    ax[1].set_yticks(range(len(labels))); ax[1].set_yticklabels(labels, fontsize=7)
    ax[1].set_xlabel("time segment"); ax[1].set_title("Band x time attribution", fontsize=10)
    plt.colorbar(im, ax=ax[1]); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path

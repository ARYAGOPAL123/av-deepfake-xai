import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalise_faces(faces_u8):
    return (faces_u8.float() / 255.0 - MEAN) / STD


def fix_length(wav, n):
    if len(wav) >= n:
        s = (len(wav) - n) // 2
        return wav[s:s + n]
    return np.pad(wav, (0, n - len(wav)))


def normalise_wav(wav):
    wav = torch.as_tensor(wav, dtype=torch.float32)
    return (wav - wav.mean()) / (wav.std() + 1e-6)


class AVDataset(Dataset):
    """Returns (faces [T,3,H,W] float, wav [N] float, label int, category str, clip_id str)."""

    def __init__(self, manifest, split, cache_dir, n_frames=32, sample_rate=16000, max_audio_sec=4.0):
        df = pd.read_csv(manifest)
        self.df = df[df.split == split].reset_index(drop=True) if split else df
        self.cache, self.T = cache_dir, n_frames
        self.n_samples = int(sample_rate * max_audio_sec)

    def __len__(self):
        return len(self.df)

    def labels(self):
        return (self.df.label == "fake").astype(int).values

    def __getitem__(self, i):
        r = self.df.iloc[i]
        faces = torch.load(f"{self.cache}/{r.clip_id}_faces.pt")
        if faces.shape[0] != self.T:
            idx = np.linspace(0, faces.shape[0] - 1, self.T).astype(int)
            faces = faces[idx]
        wav, _ = sf.read(f"{self.cache}/{r.clip_id}.wav", dtype="float32")
        wav = normalise_wav(fix_length(wav, self.n_samples))
        return normalise_faces(faces), wav, int(r.label == "fake"), r.category, r.clip_id


def balanced_sampler(ds):
    y = ds.labels()
    counts = np.bincount(y, minlength=2)
    w = 1.0 / np.maximum(counts[y], 1)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(y), replacement=True)

import torch
import torch.nn as nn
import torchaudio


class SSLAudioEncoder(nn.Module):
    """WavLM-Base+ / XLS-R with Sensitive Layer Selection (learned softmax weights over hidden layers)."""

    def __init__(self, name="microsoft/wavlm-base-plus", d=512, freeze_ssl=True):
        super().__init__()
        from transformers import AutoModel
        self.ssl = AutoModel.from_pretrained(name, output_hidden_states=True)
        self.frozen = freeze_ssl
        if freeze_ssl:
            for p in self.ssl.parameters():
                p.requires_grad = False
        L = self.ssl.config.num_hidden_layers + 1
        self.layer_w = nn.Parameter(torch.zeros(L))
        self.proj = nn.Linear(self.ssl.config.hidden_size, d)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:                       # frozen SSL model: no dropout / masking on its features
            self.ssl.eval()
        return self

    def forward(self, wav):                                     # (B,N) 16 kHz
        hs = torch.stack(self.ssl(wav).hidden_states)          # (L,B,Ta,H)
        w = self.layer_w.softmax(0)[:, None, None, None]
        return self.proj((w * hs).sum(0))                      # (B,Ta,d)


class MelCNNAudioEncoder(nn.Module):
    """Light CNN over 80-bin log-Mel spectrogram (offline / classical baseline)."""

    def __init__(self, d=512, sr=16000, n_mels=80):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=400, hop_length=160, n_mels=n_mels)
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(), nn.MaxPool2d((2, 2)))
        self.cnn = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        self.proj = nn.Linear(256 * (n_mels // 16), d)

    def forward(self, wav):                                     # (B,N)
        x = torch.log(self.mel(wav) + 1e-6).unsqueeze(1)       # (B,1,80,F)
        x = self.cnn(x)                                        # (B,256,5,F/16)
        x = x.permute(0, 3, 1, 2).flatten(2)                   # (B,Ta,256*5)
        return self.proj(x)


def build_audio_encoder(name, d=512, freeze_ssl=True):
    if name == "cnn":
        return MelCNNAudioEncoder(d)
    return SSLAudioEncoder(name, d, freeze_ssl)

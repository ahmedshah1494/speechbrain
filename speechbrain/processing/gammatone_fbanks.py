from typing import Optional, Tuple

import numpy as np

from spafe.utils.converters import hz2erb, ErbConversionApproach
from spafe.utils.exceptions import ParameterError, ErrorMsgs
from spafe.utils.filters import scale_fbank, ScaleType
from spafe.fbanks.gammatone_fbanks import generate_center_frequencies, EarQ, minBW, compute_gain
from speechbrain.processing.features import (
    STFT,
    spectral_magnitude,
)
import torch
import torchaudio

def gammatone_filter_banks(
    nfilts: int = 24,
    nfft: int = 512,
    fs: int = 16000,
    low_freq: float = 0,
    high_freq: Optional[float] = None,
    scale: ScaleType = "constant",
    order: int = 4,
    width: float = 1.,
    conversion_approach: ErbConversionApproach = "Glasberg",
):
    # init freqs
    high_freq = high_freq or fs / 2

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])

    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])

    # define custom difference func
    def Dif(u, a):
        return u - a.reshape(nfilts, 1)

    # init vars
    fbank = np.zeros([nfilts, nfft])
    maxlen = nfft // 2 + 1
    T = 1 / fs
    n = 4
    u = np.exp(1j * 2 * np.pi * np.array(range(nfft // 2 + 1)) / nfft)
    idx = range(nfft // 2 + 1)

    # computer center frequencies, convert to ERB scale and compute bandwidths
    fcs = generate_center_frequencies(low_freq, high_freq, nfilts)
    ERB = width * ((fcs / EarQ) ** order + minBW**order) ** (1 / order)
    B = 1.019 * 2 * np.pi * ERB

    # compute input vars
    wT = 2 * fcs * np.pi * T
    pole = np.exp(1j * wT) / np.exp(B * T)

    # compute gain and A matrix
    A, Gain = compute_gain(fcs, B, wT, T)

    # compute fbank
    fbank[:, idx] = (
        (T**4 / Gain.reshape(nfilts, 1))
        * np.abs(Dif(u, A[0]) * Dif(u, A[1]) * Dif(u, A[2]) * Dif(u, A[3]))
        * np.abs(Dif(u, pole) * Dif(u, pole.conj())) ** (-n)
    )

    # make sure all filters has max value = 1.0
    try:
        fbank = np.array([f / np.max(f) for f in fbank[:, range(maxlen)]])

    except BaseException:
        fbank = fbank[:, idx]

    # compute scaling
    scaling = scale_fbank(scale=scale, nfilts=nfilts)
    fbank = fbank * scaling
    return fbank, np.array([hz2erb(freq, conversion_approach) for freq in fcs])

class GammatoneFbank(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int = 25,
        hop_length: int = 10,
        low_freq=0,
        high_freq=8000,
        n_filts=40,
        scale='constant',
        conversion_approach='Glasberg',
        normalize_fbank: bool = True,
        preemphasize: bool = True,
        width: int = 1,
        power: float = 0.3,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.n_filts = n_filts
        self.scale = scale
        self.conversion_approach = conversion_approach
        self.power = power

        self.preemphasize = torchaudio.transforms.Preemphasis(coeff=0.97)
        self.compute_STFT = STFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad_mode="constant"
        )

        fbank = gammatone_filter_banks(
            nfilts=n_filts,
            nfft=n_fft,
            fs=sample_rate,
            low_freq=low_freq,
            high_freq=high_freq,
            scale=scale,
            conversion_approach=conversion_approach,
            width=width
        )

        self.register_buffer('fbank', torch.tensor(fbank[0], dtype=torch.float32))
        self.register_buffer('fbank_freqs', torch.tensor(fbank[1], dtype=torch.float32))

        if normalize_fbank:
            self.fbank = self.fbank / self.fbank.sum(1, keepdim=True)

    def forward(self, x):
        if self.preemphasize:
            x = self.preemphasize(x)

        spec = self.compute_STFT(x).transpose(1, 2)
        spec = spectral_magnitude(spec)
        P = torch.matmul(spec.transpose(1, 2), self.fbank.T.to(x.device))
        P = P ** self.power
        return P

class DifferenceOfGammatoneFbank(GammatoneFbank):
    def __init__(self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int = 25,
        hop_length: int = 10,
        low_freq=0,
        high_freq=8000,
        n_filts=40,
        scale='constant',
        conversion_approach='Glasberg',
        preemphasize: bool = True,
        width: int = 1,
        width_ratio: float = 1.25,
        power: float = 0.3,
        normalize_fbank: bool = True
        ):
        super().__init__(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            low_freq=low_freq,
            high_freq=high_freq,
            n_filts=n_filts,
            scale=scale,
            conversion_approach=conversion_approach,
            normalize_fbank=normalize_fbank,
            preemphasize=preemphasize,
            width=width,
            power=power
        )
        self.width_ratio = width_ratio

        fbank2 = gammatone_filter_banks(
            nfilts=n_filts,
            nfft=n_fft,
            fs=sample_rate,
            low_freq=low_freq,
            high_freq=high_freq,
            scale=scale,
            conversion_approach=conversion_approach,
            width=width*width_ratio
        )
        fbank2 = torch.tensor(fbank2[0], dtype=torch.float32)
        fbank2 = fbank2 / fbank2.sum(1, keepdim=True)
        self.fbank = self.fbank - fbank2
        self.fbank = self.fbank / torch.relu(self.fbank).sum(1, keepdim=True)

    def forward(self, x):
        if (x.dim() == 2) or (x.shape[1] == 1):
            if self.preemphasize:
                x = self.preemphasize(x)
            spec = self.compute_STFT(x).transpose(1, 2)
            spec = spectral_magnitude(spec)
        else:
            spec = x.transpose(1, 2)
        P = torch.matmul(spec.transpose(1, 2), self.fbank.T.to(x.device))
        P = torch.relu(P)
        P = P ** self.power
        return P
        
import torch
import torchaudio

from speechbrain.processing.features import (
    STFT,
    spectral_magnitude,
)

from speechbrain.processing.gammatone_fbanks import gammatone_filter_banks

def freq2ath(f):
    return 3.64 * (f / 1000)**-0.8 - 6.5 * torch.exp(-0.6 * (f / 1000 - 3.3)**2) + 0.001 * (f / 1000)**4

def freq2bark(f):
    return 13 * torch.arctan(0.76 * f / 1000) + 3.5 * torch.arctan((f / 7500)**2)

def apply_freq_masking(spec, fs, freqs=None):
    psd = 10 * torch.log10((spec/spec.shape[1])**2)
    norm_psd = 96 - psd.max(2, keepdim=True)[0] + psd

    norm_psd_ = (10**(norm_psd/10))
    norm_psd_unfold = norm_psd_.unfold(2, 3, 1)
    smooth_psd = 10*torch.log10(norm_psd_unfold.sum(3))

    if freqs is None:
        freqs = torch.tensor([i * fs / 2048 for i in range(1,spec.shape[-1]-1)], dtype=spec.dtype, device=spec.device)
    barks = freq2bark(freqs)
    ath = freq2ath(freqs)

    dbij = (barks.unsqueeze(0) - barks.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    G = -27 + 0.37 * torch.relu(smooth_psd - 40)
    G = G.unsqueeze(-1)
    SF = torch.where(
        dbij < 0,
        27*dbij,
        G * dbij
    )
    dm = -6.025-0.275*barks
    dm = dm.reshape(1,1,-1,1)
    T = SF + dm + smooth_psd.unsqueeze(-1)

    theta = 10*torch.log10((10**(T/10)).sum(2) + 10**(ath.reshape(1,1,-1)/10))

    # mask = (smooth_psd > theta).float()
    spec_ = spec[...,1:-1]
    # masked_spec = spec_ * mask.clone().detach()
    masked_spec = torch.where(
        smooth_psd > theta,
        spec_,
        torch.zeros_like(spec_)
    )
    return masked_spec

class FrequencyMasking(torch.nn.Module):
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
        fbank_type=None,
        conversion_approach='Glasberg',
        normalize_fbank: bool = True,
        preemphasize: bool = True,
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
        self.fbank_type = fbank_type

        self.preemphasize = torchaudio.transforms.Preemphasis(coeff=0.97) if preemphasize else None
        self.compute_STFT = STFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad_mode="constant"
        )

        if fbank_type is not None:
            if fbank_type == 'gammatone':
                fbank, fbank_freqs_erb = gammatone_filter_banks(
                    nfilts=n_filts,
                    nfft=n_fft,
                    fs=sample_rate,
                    low_freq=low_freq,
                    high_freq=high_freq,
                    scale=scale,
                    conversion_approach=conversion_approach
                )

                from spafe.utils.converters import erb2hz
                fbank_freqs = [erb2hz(freq, conversion_approach) for freq in fbank_freqs_erb]
            else:
                raise ValueError(f"Unknown fbank type: {fbank_type}")

            self.register_buffer('fbank', torch.tensor(fbank, dtype=torch.float32))
            self.register_buffer('fbank_freqs', torch.tensor(fbank_freqs, dtype=torch.float32))

            if normalize_fbank:
                self.fbank = self.fbank / self.fbank.sum(1, keepdim=True)
        else:
            self.fbank = None
            self.fbank_freqs = None

    def forward(self, x):
        if self.preemphasize:
            x = self.preemphasize(x)

        spec = self.compute_STFT(x).transpose(1, 2)
        P = spectral_magnitude(spec).transpose(1, 2)
        if self.fbank_type is not None:
            P = torch.matmul(P, self.fbank.T.to(x.device))
            fbank_freqs = self.fbank_freqs[1:-1].to(x.device)
        else:
            fbank_freqs = None
        P = apply_freq_masking(P, self.sample_rate, fbank_freqs)
        P = P ** 0.3
        # print(P.shape)
        return P

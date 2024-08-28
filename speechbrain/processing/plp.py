from spafe.fbanks.bark_fbanks import bark_filter_banks
from speechbrain.processing.features import STFT, spectral_magnitude
import numpy as np
from scipy import signal
import torch
import torchaudio


def toeplitz(r):
    tpz = torch.zeros(*(r.shape), r.shape[-1], dtype=r.dtype, device=r.device)
    n = r.shape[-1]
    for i in range(n):
        tpz[..., i, i:] = r[..., :r.shape[-1]-i]
        tpz[..., i:, i] = r[..., :r.shape[-1]-i]
    return tpz

def m_lpc_helper(frame, order):
    '''
    Compute LPC coefficients and error from a frame
    input:
        frame: torch.Tensor, shape (batch, frame_len, channels)
        order: int, number of LPC coefficients to compute
    return:
        a: torch.Tensor, shape (batch, frame_len, order+1), LPC coefficients
        e: torch.Tensor, shape (batch, frame_len), error
    '''
    def _autocorrelate(x, padding):
        bs, fl, ch = x.shape
        padded_frame = torch.nn.functional.pad(x, padding).unsqueeze(1)
        unfolded = torch.nn.functional.unfold(
            padded_frame, 
            (1, ch))
        unfolded = unfolded.transpose(1,2).reshape(bs, fl, -1, ch)
        ac = torch.matmul(unfolded, x.unsqueeze(3)).squeeze(-1)
        return ac
    
    bs, fl, ch = frame.shape
    p = order + 1
    r = torch.zeros(frame.shape[0], frame.shape[1], p, dtype=frame.dtype, device=frame.device)
    # Number of non zero values in autocorrelation one needs for p LPC coefficients
    nx = np.min([p, ch])

    r[..., :nx] = _autocorrelate(frame, (0, nx-1))[..., : nx]
    # print('r', torch.isfinite(r).all())

    r_shape = r.shape
    r = r.reshape(-1, p)
    nz_r = (r > 1e-36).any(1)
    tpz = toeplitz(r[nz_r, :-1])
    inv_tpz_ = torch.linalg.inv(tpz)
    # print('tpz', torch.isfinite(tpz).all())
    # print('inv_tpz_', torch.isfinite(inv_tpz_).all())

    inv_tpz = torch.zeros(bs*fl, p-1, p-1, dtype=frame.dtype, device=frame.device)
    inv_tpz[nz_r] = inv_tpz_
    inv_tpz = inv_tpz.reshape(r_shape[0], r_shape[1], p-1, p-1)
    r = r.reshape(*r_shape)
    
    phi = (inv_tpz * -r[..., 1:].unsqueeze(2)).sum(3)
    # print('phi', torch.isfinite(phi).all())
    a = torch.ones(bs, fl, 1, dtype=frame.dtype, device=frame.device)
    a = torch.cat((a,phi), 2)
    # print('a', torch.isfinite(a).all())

    auto_corr = _autocorrelate(frame, (ch-1,2))
    e = auto_corr[..., 0] + (auto_corr[..., 1: a.shape[-1]+1]*a).sum(2)
    # print('e', torch.isfinite(e).all())
    return a, e.abs()

def m_lpc2lpcc(a, e, nceps):
    '''
    Compute LPCC coefficients from LPC coefficients
    input:
        a: torch.Tensor, shape (batch, frame_len, order), LPC coefficients
        e: torch.Tensor, shape (batch, frame_len), error
        nceps: int, number of LPCC coefficients to compute
    return:
        c: torch.Tensor, shape (batch, frame_len, nceps), LPCC coefficients
    '''
    bs, fl, nfilt = a.shape
    p = nfilt
    c = torch.zeros(bs, fl, nceps, dtype=a.dtype, device=a.device)

    e[e == 0] = 1e-38
    c[..., 0] = torch.log(e)
    for m in range(1, nceps):
        c[..., m] = a[..., m]
        if m > 1:
            r = torch.arange(1, m, dtype=a.dtype, device=a.device).reshape(1,1,-1)/m
            # print(c[..., 1:m])
            c[..., m] = (r*c[..., 1:m]*a[..., 1:m].flip(-1)).sum(-1)
    if nceps > p:
        c[..., p:nceps] = (torch.arange(1,m).reshape(1,1,-1)*c[..., 1:m]*a[..., 1:m].flip(1)).sum(-1)/m
    return c

class PLP(torch.nn.Module):
    def __init__(
            self,
            sample_rate: int = 16000,
            order: int = 13,
            n_fft: int = 400,
            win_length: int = 25,
            hop_length: int = 10,
            low_freq=0,
            high_freq=8000,
            n_filts=24,
            rasta=False,
            pre_emph: bool = False,
            normalize: bool = False,
    ):
        super(PLP, self).__init__()
        self.sample_rate = sample_rate
        self.order = order
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.n_filts = n_filts
        self.rasta = rasta
        self.pre_emph = pre_emph
        self.normalize = normalize


        fbank, fbank_freqs = bark_filter_banks(
            nfilts=n_filts,
            nfft=n_fft,
            fs=sample_rate,
            low_freq=low_freq,
            high_freq=high_freq,
            scale="constant",
            conversion_approach="Wang"
        )
        self.register_buffer('fbank', torch.tensor(fbank, dtype=torch.float32))
        self.register_buffer('fbank_freqs', torch.tensor(fbank_freqs, dtype=torch.float32))

        self.preemphasize = torchaudio.transforms.Preemphasis(coeff=0.97)
        self.compute_STFT = STFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad_mode="constant"
        )
    
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if self.pre_emph:
            wav = self.preemphasize(wav)
        spec = self.compute_STFT(wav).transpose(1, 2)
        spec = spectral_magnitude(spec)
        # print(spec.shape)
        P = torch.matmul(spec.transpose(1, 2), self.fbank.T.to(wav.device))
        P[P == 0] = 1e-38
        logP = torch.log(P)
        # print('logP', torch.isfinite(logP).all())

        logE = lambda logw: torch.logaddexp(6*logw, 4*logw+torch.log(torch.tensor(56.8e6, dtype=logw.dtype, device=logw.device))) - (
            2*torch.logaddexp(2*logw, torch.log(torch.tensor(6.3e6, dtype=logw.dtype, device=logw.device))) + 
            torch.logaddexp(2*logw, torch.log(torch.tensor(0.38e9, dtype=logw.dtype, device=logw.device))) +
            torch.logaddexp(6*logw, torch.log(torch.tensor(9.58e26, dtype=logw.dtype, device=logw.device)))
        )
        logY = logE(logP)
        # intensity loudness compression
        L = torch.exp(logY / 3)
        # print('L', torch.isfinite(L).all())
        # L = np.abs(Y) ** (1 / 3)
        # ifft
        inverse_fourrier_transform = torch.fft.ifft(L, self.n_fft).abs()
        # print('inverse_fourrier_transform', torch.isfinite(inverse_fourrier_transform).all())
        # compute lpcs and lpccs
        lpcs, e = m_lpc_helper(inverse_fourrier_transform, self.order-1)
        # print('lpcs', torch.isfinite(lpcs).all(), 'e', torch.isfinite(e).all())
        lpccs = m_lpc2lpcc(lpcs, e, self.order)
        # print('lpccs', torch.isfinite(lpccs).all())
        # lpcs = torch.zeros(L.shape[0], L.shape[1], self.order, dtype=L.dtype, device=L.device)
        # lpccs = torch.zeros(L.shape[0], L.shape[1], self.order, dtype=L.dtype, device=L.device)
        # for i in range(L.shape[1]):
        #     frames = inverse_fourrier_transform[:, i]
        #     nz_frames = (frames >= 1e-19).any(1)
        #     a, e = m_lpc_helper(frames[nz_frames], self.order - 1)
        #     lpcs[nz_frames, i] = a
        #     lpcc_coeffs = m_lpc2lpcc(a, e, self.order)
        #     lpccs[nz_frames, i] = lpcc_coeffs
        
        if self.normalize:
            lpccs = (lpccs - lpccs.mean(1, keepdim=True)) / lpccs.std(1, keepdim=True)
        return lpccs

import torch
import torchaudio
from speechbrain.processing.features import (
    STFT,
    spectral_magnitude,
    DCT,
)
from spafe.fbanks.gammatone_fbanks import gammatone_filter_banks

def medium_time_power(x, M):
    """
    Compute the medium time power of a signal
    """
    
    x = torch.nn.functional.pad(x, (0, 0, M, M), mode='constant', value=0)
    x = x.unfold(1, 2*M+1, 1)
    x = x.mean(-1)
    return x

def assymetric_lowpass_filtering(Qin, lm_a, lm_b, lm_0):
    """
    Apply an assymetric lowpass filtering to a signal
    """
    Qout = torch.zeros_like(Qin, dtype=Qin.dtype, device=Qin.device)
    Qout[:, 0] = lm_0 * Qin[:, 0]
    for m in range(Qin.size(1)):
        Q1 = lm_a * Qout[:, m-1] + (1-lm_a) * Qin[:, m]
        Q2 = lm_b * Qout[:, m-1] + (1-lm_b) * Qin[:, m]
        Qout[:, m] = torch.where(Qin[:, m] > Qout[:, m-1], Q1, Q2)
    return Qout

def temporal_masking(Q0, lm_t=0.85, mu_t=0.2):
    Qp = torch.zeros_like(Q0, dtype=Q0.dtype, device=Q0.device)
    Rsp = torch.zeros_like(Q0, dtype=Q0.dtype, device=Q0.device)

    Qp[:, 0] = Q0[:, 0]
    Rsp[:, 0] = Q0[:, 0]
    for m in range(1, Q0.shape[1]):
        Qp[:, m] = torch.maximum(lm_t * Qp[:, m-1], Q0[:, m])
        Rsp[:, m] = torch.where(
            Q0[:, m] >= lm_t * Qp[:, m - 1],
            Q0[:, m],
            mu_t * Qp[:, m-1]
        )
    return Rsp

def asymmetric_noise_suppression_with_temporal_masking(Qin, lm_a, lm_b, lm_t, mu_t, c):
    Qle = assymetric_lowpass_filtering(Qin, lm_a, lm_b, 0.9)
    Q0 = torch.relu(Qin - Qle)
    Qf = assymetric_lowpass_filtering(Q0, lm_a, lm_b, 0.9)
    Qt = temporal_masking(Q0, lm_t, mu_t)
    R = torch.where(Qin >= c * Qle, Qt, Qf)
    return R

def weight_smoothing(
    R: torch.Tensor, Q: torch.Tensor, N: int = 4) -> torch.Tensor:
    """
    Apply spectral weight smoothing according to [Kim]_.

    Args:
        R (torch.Tensor) :
        Q (torch.Tensor) : medium time power
        N                 (int) :

    Returns:
        (torch.Tensor) : time-averaged frequency-averaged transfer function.

    Note:
        .. math::
            \\tilde{S}[m, l]=(\\frac{1}{l_{2}-l_{1}+1} \\sum_{l^{\\prime}=l_{1}}^{l_{2}} \\frac{\\tilde{R}[m, l^{\\prime}]}{\\tilde{Q}[m, l^{\\prime}]})

        where :math:`l_{2}=\\min (l+N, L)` and :math:`l_{1}=\\max (l-N, 1)`, and :math:`L` is the total number of channels,
        and :math:`\\tilde{R}` is the output of the asymmetric noise suppression and temporal masking modules
        and :math:`\\tilde{S}` is the time-averaged, frequency-averaged transfer function.
    """
    D = torch.zeros(R.shape[2], dtype=R.dtype, device=R.device) + 2*N+1
    D[:N] = torch.arange(N, dtype=R.dtype, device=R.device) + 1 + N
    D[-N:] = torch.arange(N, 0, -1, dtype=R.dtype, device=R.device) + N

    R_ = torch.nn.functional.pad(R, (N, N), mode='constant', value=0)
    R_ = R_.unfold(2, 2*N+1, 1)

    Q_ = torch.nn.functional.pad(Q, (N, N), mode='constant', value=1e-8)
    Q_ = Q_.unfold(2, 2*N+1, 1)

    S = R_ / Q_
    S = S.sum(3) / D
    return S

def mean_power_normalization(T, lm_mu=0.999, k=1):
    """
    Apply mean power normalization to a signal
    """
    L = T.shape[2]
    N = T.shape[1]
    mu = torch.zeros(T.shape[0], T.shape[1], 1, dtype=T.dtype, device=T.device)
    mu[:,0] = 0.0001

    T_favg = T.mean(2)
    for m in range(1, N):
        mu[:, m] = lm_mu * mu[:, m-1] + (1-lm_mu) * T_favg[:, m]
    
    U = k * T / mu
    return U

class PNCC(torch.nn.Module):
    def __init__(self,
                sample_rate: int = 16000,
                n_fft: int = 1024,
                win_length: int = 25,
                hop_length: int = 10,
                low_freq=200,
                high_freq=8000,
                n_filts=40,
                lm_a=0.999,
                lm_b=0.5,
                lm_t=0.85,
                lm_mu=0.999,
                mu_t=0.2,
                c=2,
                weight_smoothing_window_size = 4,
                medium_time_power_window_size = 2,
                normalize_fbank=True,
                clip_fbank=False,
                return_cepstral=True,
                normalize_ceps=False,
                n_ceps=40
                 ) -> None:
        super().__init__()
        self.lm_a = lm_a
        self.lm_b = lm_b
        self.lm_t = lm_t
        self.lm_mu = lm_mu
        self.mu_t = mu_t
        self.c = c
        self.weight_smoothing_window_size = weight_smoothing_window_size
        self.medium_time_power_window_size = medium_time_power_window_size
        self.return_cepstral = return_cepstral
        self.normalize_ceps = normalize_ceps
        self.n_ceps = n_ceps
        
        self.preemphasize = torchaudio.transforms.Preemphasis(coeff=0.97)
        # self.compute_STFT = torchaudio.transforms.Spectrogram(
        #     n_fft=n_fft,
        #     win_length=win_length*sample_rate//1000,
        #     hop_length=hop_length*sample_rate//1000,
        #     pad_mode='constant'
        # )
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
            scale='constant',
            conversion_approach='Glasberg'
        )
        self.register_buffer('fbank', torch.tensor(fbank[0], dtype=torch.float32))
        self.register_buffer('fbank_freqs', torch.tensor(fbank[1], dtype=torch.float32))
        if clip_fbank:
            self.fbank = torch.relu(self.fbank - 0.5*(self.fbank.max(1, keepdim=True)[0]))
        self.fbank = self.fbank.pow(2)
        
        if normalize_fbank:
            self.fbank = self.fbank / self.fbank.sum(1, keepdim=True)
        
        self.dct = DCT(n_filts, n_ceps)
        # dct_mat = torchaudio.functional.create_dct(nfilts, nfilts, 'ortho')
        # self.register_buffer('dct_mat', dct_mat)

    def forward(self, x):
        x = self.preemphasize(x)
        spec = self.compute_STFT(x).transpose(1, 2)
        spec = spectral_magnitude(spec)
        P = torch.matmul(spec.transpose(1, 2), self.fbank.T)
        Qt = medium_time_power(P, self.medium_time_power_window_size)
        Rt = asymmetric_noise_suppression_with_temporal_masking(
            P, self.lm_a, self.lm_b, self.lm_t, self.mu_t, self.c
        )
        St = weight_smoothing(Rt, Qt, self.weight_smoothing_window_size)
        T = P*St
        U = mean_power_normalization(T, self.lm_mu)
        V = U.pow(1/15)

        if self.return_cepstral:
            # cepstral = torch.matmul(V, self.dct_mat.T)
            cepstral = self.dct(V)
            if self.normalize_ceps:
                cepstral = (cepstral - cepstral.mean(1, keepdim=True)) / cepstral.std(1, keepdim=True)
            return cepstral[..., :self.n_ceps]
        else:
            return V
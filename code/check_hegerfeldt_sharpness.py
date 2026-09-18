"""Sharpness of the Hegerfeldt inequality (Subsection 7.2, Proposition 7.7) at large lambda0.

Extremal data u = q - q(.-h), with q the Gaussian packet of Lemma 5.1 (delta = lambda0^{-1/4})
transported to an interval of length 2r'. The negative-frequency energy is bounded by E_- <= 4 mu,
with the leak mu = int m(l) |ghat(l - lambda0)|^2 dl computed exactly from the Mellin formula
(Corollary 3.4) at 60 digits; E and M_+ are computed from F = F(phi) by FFT. The proposition
predicts log(E/E_-) / (pi r omega_bar) -> 1 and pi r omega_bar / (2 pi lambda0) -> 1.
"""
import numpy as np
from mpmath import mp, quad, exp, sqrt, pi, log, inf

mp.dps = 60
r = 1.0
print(" lam   log(E/4mu)/(pi r omega_bar)   pi r omega_bar/(2 pi lam)")
for lam in [4, 8, 16, 32, 64, 128]:
    d = lam ** -0.25
    mu = quad(lambda l: 1 / (1 + exp(2 * pi * l)) * sqrt(2 * d * d / pi) * exp(-2 * d * d * (l - lam) ** 2),
              [-inf, lam - 20 / d, lam, inf])
    N = 2 ** 20
    s = (np.arange(N) + 0.5) / N
    rho = np.log(s / (1 - s))
    phi = (2 * np.pi * d * d) ** -0.25 * np.exp(-rho ** 2 / (4 * d * d)) * np.exp(-1j * lam * rho) / np.sqrt(s * (1 - s))
    # |F(t)| = |int phi e^{2 pi i s t} ds| on the grid t_j = j/P (zero padding by a factor P)
    P = 8
    buf = np.zeros(P * N, complex)
    buf[:N] = phi
    F = np.fft.ifft(buf) * P
    t = np.fft.fftfreq(P * N) * N
    dt = 1 / P
    F2 = np.abs(F) ** 2
    tbar = (lam / np.pi) * (1 + np.exp(d * d / 2))
    L = 2 * r * (1 - 1 / lam)
    wt = 4 * np.sin(np.pi * t / (2 * tbar)) ** 2          # 4 sin^2(k h / 2), k = 2 pi t / L, h = L / (2 tbar)
    E = np.sum(wt * F2) * dt
    Mp = np.sum(np.where(t > 0, 2 * np.pi * t / L * wt * F2, 0)) * dt
    om_bar = Mp / E
    lhs = float(log(E / (4 * mu)))
    print(f"{lam:4d}   {lhs / (np.pi * r * om_bar):.4f}                  {np.pi * r * om_bar / (2 * np.pi * lam):.4f}")

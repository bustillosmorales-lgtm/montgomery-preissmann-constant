"""Numerical check of the transfer lemma (Section sec:transfer) on the actual minimisers.

p(xi) = sum_{k<n} u_k e^{2 pi i k xi},  u = bottom eigenvector of T_n(xi),  x = n lambda_min.
  mu_p = int_{1/2}^1 |p|^2,   m_p = n int_0^{1/2} xi |p|^2       (so mu_p <= 2x/n, m_p <= x)
Window: omega(t) = w0(t/L), w0(t) = (sin(pi t/2)/(pi t/2))^4, spectrum in [-1/L, 1/L], 0 <= w0 <= 1.
F(t) = n^{-1/2} p(t/n) omega(t). Lemma claims:
  (T1) leak  mu[F] <= mu_p + tail_mu
  (T2) D[F]       <= m_p  + tail_D
  (T3) ||F||^2    >= (min_{[0,T]} w0(t/L)^2) (1 - m_p/T - mu_p)     for every T > 0
and then (local theorem after shift + dilation by a = 1 + 4/L):
  (T4) mu[F]/||F||^2 >= 0.37847 exp(-pi^2 a D[F]/||F||^2)
  (T5) final bound:   n lambda_min >= (log n - log log n)/pi^2
"""
import sys
import numpy as np
sys.path.insert(0, "code")
from tradeoff_leak_moment import entries_np

PI = np.pi
C0 = 0.5 * np.exp(-PI * 0.08863802)
print(f"{'n':>5} {'L':>6} {'x=n*lam':>8} {'mu_p':>10} {'mu[F]':>10} {'m_p':>8} {'D[F]':>8} {'||F||^2':>8} {'T3 bound':>8} "
      f"{'T4 lhs/rhs':>10} {'(logn-loglogn)/pi^2':>20}")
for n in (256, 1024, 4096):
    A, B = entries_np(n)
    T = A + B / n
    # T_n(xi) itself: diag 1/2, off-diagonal i/(2 pi (j-k))
    j = np.arange(n); d = j[:, None] - j[None, :]
    Tn = np.where(d == 0, 0.5, 1j / (2 * PI * np.where(d == 0, 1, d)))
    w, V = np.linalg.eigh(Tn)
    u = V[:, 0]; x = n * w[0]
    mu_p = np.real(u.conj() @ A @ u); m_p = np.real(u.conj() @ B @ u)
    for L in (np.sqrt(n), n / 16):
        q = 16                                    # t-step 1/q
        per = n * q
        buf = np.zeros(per, dtype=complex); buf[:n] = u
        pt = per * np.fft.ifft(buf)               # p(t/n) at t = k/q, k = 0..per-1 (one period in t)
        reps = 12
        f = np.tile(abs(pt) ** 2, 2 * reps) / n   # |p(t/n)|^2 / n on t in [-reps n, reps n)
        t = (np.arange(2 * reps * per) - reps * per) / q
        z = PI * t / (2 * L)
        w0 = np.where(z == 0, 1.0, (np.sin(z) / np.where(z == 0, 1, z))) ** 4
        g = f * w0 ** 2
        h = 1 / q
        i0 = reps * per
        d1 = (-g[i0 + 2] + 8 * g[i0 + 1] - 8 * g[i0 - 1] + g[i0 - 2]) / (12 * h)
        muF = h * (g[:i0].sum() + g[i0] / 2) - h * h / 12 * d1
        DF = h * (t[i0 + 1:] * g[i0 + 1:]).sum() + h * h / 12 * g[i0]
        NF = h * g.sum()
        Tc = n ** 0.25
        zc = PI * Tc / (2 * L)
        T3 = (np.sin(zc) / zc) ** 8 * (1 - m_p / Tc - mu_p)
        a = 1 + 4 / L
        t4 = (muF / NF) / (C0 * np.exp(-PI ** 2 * a * DF / NF))
        print(f"{n:5d} {L:6.0f} {x:8.4f} {mu_p:10.3e} {muF:10.3e} {m_p:8.4f} {DF:8.4f} {NF:8.5f} {T3:8.5f} "
              f"{t4:10.3g} {(np.log(n)-np.log(np.log(n)))/PI**2:20.4f}")
        assert muF <= mu_p * (1 + 1e-6) + 1e-12 and DF <= m_p * (1 + 1e-6) + 1e-9 and NF >= T3 - 1e-9 and t4 >= 1
print("\nall inequalities (T1)-(T5) hold")

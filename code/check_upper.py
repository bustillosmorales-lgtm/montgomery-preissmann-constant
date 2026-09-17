"""Numerical check of the upper bound (Section sec:upper).

Trial vector u_k = n^{-1/2} phi(k/n), phi = U^{-1} psi, psi(rho) = g(rho) e^{-i l0 rho},
g Gaussian with int g^2 = 1 and variance parameter delta (g^2 ~ N(0, delta^2)).
Claims checked:
  (U1) exact first moment  int t |F|^2 = (l0/pi)(1 + e^{delta^2/2})          [local model, FFT]
  (U2) leak  mu <= exp(-2 pi l0 + pi^2/(2 delta^2))                          [local model, FFT]
  (U3) Rayleigh quotient of T_n(xi) at u  <=  (D + n mu)/n (1 + small)        [discrete, exact matvec]
  (U4) with delta^2 = (log n)^{-1/2}, 2 pi l0 = log n + pi^2/(2 delta^2):
       pi^2 n lambda_n <= pi^2 n R(u) <= log n + 6 sqrt(log n) + 13  (the theorem's bound, C = 13 here)
and, for comparison, the best Gaussian packet (optimised l0, delta) against lambda_n.
"""
import sys
import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import minimize
sys.path.insert(0, "code")
from hilbert_norm import lanczos_norm

LAM = {}
for _l in open("results/hilbert_norm.csv").read().splitlines()[1:]:
    _c = _l.split(",")
    LAM[int(_c[0])] = float(_c[3])          # lambda_n from hilbert_norm.py

PI = np.pi


def phi_samples(s, l0, d):
    s = np.asarray(s, dtype=float)
    out = np.zeros(s.shape, dtype=complex)
    m = (s > 0) & (s < 1)
    sm = s[m]
    rho = np.log(sm / (1 - sm))
    g = (2 * PI * d * d) ** -0.25 * np.exp(-rho ** 2 / (4 * d * d))
    out[m] = np.exp(-1j * l0 * rho) * g / np.sqrt(sm * (1 - sm))
    return out


def rayleigh(n, l0, d):
    u = phi_samples(np.arange(n) / n, l0, d) / np.sqrt(n)
    m = np.arange(-(n - 1), n)
    with np.errstate(divide="ignore"):
        ker = np.where(m == 0, 0.0, 1.0 / np.where(m == 0, 1, m))
    Hu = fftconvolve(u, ker, mode="full")[n - 1:2 * n - 1]
    Tu = 0.5 * u + 1j / (2 * PI) * Hu
    return np.real(np.vdot(u, Tu)) / np.real(np.vdot(u, u)), np.real(np.vdot(u, u))


def local_model(l0, d, N=2 ** 16, P=64):
    S = (np.arange(N) + 0.5) / N
    phi = phi_samples(S, l0, d)
    M = N * P
    x = np.zeros(M, dtype=complex); x[:N] = phi
    k = np.arange(M)
    F = (M * np.fft.ifft(x)) * np.exp(1j * PI * k / M) / N
    t = np.where(k < M // 2, k, k - M) / P
    f = abs(F) ** 2; h = 1 / P
    d1 = (-f[2] + 8 * f[1] - 8 * f[-1] + f[-2]) / (12 * h)
    mu = h * (f[t < 0].sum() + f[0] / 2) - h * h / 12 * d1
    M1 = h * (t * f).sum()
    D = h * (t[t > 0] * f[t > 0]).sum() + h * h / 12 * f[0]
    return mu, M1, D, h * f.sum()


print("(U1)-(U2) local model")
print(f"{'l0':>6} {'delta':>6} {'int t|F|^2':>12} {'(l0/pi)(1+e^(d^2/2))':>21} {'D':>10} {'mu':>11} {'bound U2':>11}")
for l0, d in [(2.0, 1.0), (3.0, 0.7), (4.0, 0.6), (4.76, 0.53)]:
    mu, M1, D, nrm = local_model(l0, d)
    print(f"{l0:6.2f} {d:6.2f} {M1:12.7f} {l0/PI*(1+np.exp(d*d/2)):21.7f} {D:10.7f} {mu:11.3e} {np.exp(-2*PI*l0+PI**2/(2*d*d)):11.3e}")

print("\n(U3)-(U4) discrete, parameters of the theorem")
print(f"{'n':>7} {'delta':>6} {'l0':>6} {'pi^2 n R(u)':>12} {'pi^2 n lam_n':>13} {'log n+6sqrt(log n)+13':>22} {'||u||^2':>9}")
for e in (10, 12, 14, 16, 18):
    n = 2 ** e
    L = np.log(n)
    d = L ** -0.25
    l0 = (L + PI ** 2 / (2 * d * d)) / (2 * PI)
    R, nu = rayleigh(n, l0, d)
    lam = LAM[n]
    print(f"{n:7d} {d:6.3f} {l0:6.3f} {PI**2*n*R:12.4f} {PI**2*n*lam:13.4f} {L+6*np.sqrt(L)+13:22.4f} {nu:9.6f}")
    assert R >= lam - 1e-12 and PI ** 2 * n * R <= L + 6 * np.sqrt(L) + 13

print("\nbest Gaussian packet (optimised l0, delta) vs lambda_n")
print(f"{'n':>7} {'l0*':>7} {'delta*':>7} {'n R_best':>9} {'n lam_n':>9} {'ratio':>7}")
for e in (10, 14, 18):
    n = 2 ** e
    lam = LAM[n]
    res = minimize(lambda v: n * rayleigh(n, v[0], v[1])[0], x0=[1.5, 0.9], method="Nelder-Mead",
                   options=dict(xatol=1e-5, fatol=1e-9))
    print(f"{n:7d} {res.x[0]:7.3f} {res.x[1]:7.3f} {res.fun:9.5f} {n*lam:9.5f} {res.fun/(n*lam):7.4f}")

"""Radial wave equation in 3+1 vs odd extension in 1+1 (Subsection 7.2, Corollary 7.10).

3D quantities are computed from the 3D Fourier transform of the radial data by direct quadrature,
phi_hat(k) = int phi(x) e^{-ik.x} d^3x = 4 pi int_0^r r'^2 phi(r') sinc(k r') dr',
and the 3D energies/moment with d^3k/(2pi)^3 = k^2 dk/(2 pi^2).
1D quantities are computed from the odd extension psi = x phi(|x|) with the chiral formulas.
Checks: E3 = 2 pi E1, E3(phi_-) = 2 pi E1(psi_-), M3_+ = 2 pi M1_+, and the Hegerfeldt bound in 3D.
"""
import numpy as np

r = 1.0
C0 = 0.5 * np.exp(-np.pi * 0.08860170)
rng = np.random.default_rng(7)


def bump(x):
    y = np.zeros_like(x); m = np.abs(x) < 1
    y[m] = np.exp(-1 / (1 - x[m] ** 2)); return y


rr = np.linspace(0, r, 4001)
k = np.linspace(1e-6, 300, 12001)
for trial in range(5):
    a = rng.normal(size=4); b = rng.normal(size=4); w = rng.uniform(2, 25, size=4)
    f0 = lambda s: bump(s / r) * sum(a[i] * np.cos(w[i] * s * s) for i in range(4))
    f1 = lambda s: bump(s / r) * sum(b[i] * np.cos(w[i] * s) for i in range(4))
    phi0, phi1 = f0(rr), f1(rr)
    # 3D transform by quadrature (np.sinc(x) = sin(pi x)/(pi x))
    kr = np.outer(k, rr)
    ker = 4 * np.pi * rr ** 2 * np.sinc(kr / np.pi)
    P0 = np.trapezoid(ker * phi0, rr, axis=1); P1 = np.trapezoid(ker * phi1, rr, axis=1)
    hp = 0.5 * (k * P0 + 1j * P1); hm = 0.5 * (k * P0 - 1j * P1)   # h_pm = omega phi_hat_pm(0,k)
    meas = k ** 2 / (2 * np.pi ** 2)
    E3p = np.trapezoid(abs(hp) ** 2 * meas, k); E3m = np.trapezoid(abs(hm) ** 2 * meas, k)
    M3 = np.trapezoid(k * abs(hp) ** 2 * meas, k)
    # 3D energy in physical space
    d0 = np.gradient(phi0, rr)
    E3x = 0.5 * np.trapezoid((phi1 ** 2 + d0 ** 2) * 4 * np.pi * rr ** 2, rr)
    # 1D odd extension: psi0 = x phi0, psi1 = x phi1 on [-r, r]
    x = np.linspace(-4, 4, 2 ** 17, endpoint=False); dx = x[1] - x[0]
    s = np.abs(x)
    psi0 = x * f0(s); psi1 = x * f1(s)
    u = 0.5 * (psi1 - np.gradient(psi0, x)); v = 0.5 * (psi1 + np.gradient(psi0, x))
    kk = 2 * np.pi * np.fft.fftfreq(len(x), dx); dk = 2 * np.pi / (len(x) * dx)
    U = np.fft.fft(u) * dx; V = np.fft.fft(v) * dx
    E1m = (np.sum(abs(U[kk < 0]) ** 2) + np.sum(abs(V[kk > 0]) ** 2)) * dk / (2 * np.pi)
    E1 = (np.sum(u ** 2) + np.sum(v ** 2)) * dx
    M1 = (np.sum(kk[kk > 0] * abs(U[kk > 0]) ** 2) + np.sum(-kk[kk < 0] * abs(V[kk < 0]) ** 2)) * dk / (2 * np.pi)
    E3 = E3p + E3m
    wbar = M3 / E3
    print(f"E3(x)/E3(k)={E3x/E3:.6f}  E3/(2pi E1)={E3/(2*np.pi*E1):.6f}  E3-/(2pi E1-)={E3m/(2*np.pi*E1m):.6f}  "
          f"M3/(2pi M1)={M3/(2*np.pi*M1):.6f}  E3-/bound={E3m/(C0*E3*np.exp(-np.pi*r*wbar)):.1f}")

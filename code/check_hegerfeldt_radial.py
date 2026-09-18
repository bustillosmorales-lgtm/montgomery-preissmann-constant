"""Radial wave equation in 3+1 vs odd extension in 1+1 (Subsection 7.2, Corollary 7.10).

3D quantities are computed from the 3D Fourier transform of the radial data by direct quadrature,
phi_hat(k) = int phi(x) e^{-ik.x} d^3x = 4 pi int_0^r r'^2 phi(r') sinc(k r') dr',
and the 3D energies/moment with d^3k/(2pi)^3 = k^2 dk/(2 pi^2).
1D quantities are computed from the odd extension psi = x phi(|x|) with the chiral formulas.
Checks: E3 = 2 pi E1, E3(phi_-) = 2 pi E1(psi_-), M3_+ = 2 pi M1_+, and the Hegerfeldt bound in 3D.
The data are complex (for real data E_- = E/2 identically and the bound check would be empty):
  (i)  random complex radial data;
  (ii) data dominated by positive frequencies, built as in the proof of Corollary 7.10:
       u = d/dy [bump(y - c) e^{i kappa y}] (zero integral), v(y) = -u(-y),
       psi_1 = u + v, psi_0 = int_{-inf}^y (v - u), phi = psi / |x|.
"""
import numpy as np

r = 1.0
C0 = 0.5 * np.exp(-np.pi * 0.08860170)
rng = np.random.default_rng(7)


def bump(x):
    y = np.zeros(np.shape(x)); m = np.abs(x) < 1
    y[m] = np.exp(-1 / (1 - x[m] ** 2)); return y


rr = np.linspace(0, r, 40001)
k = np.linspace(1e-6, 300, 12001)
x = np.linspace(-4, 4, 2 ** 17, endpoint=False); dx = x[1] - x[0]
kk = 2 * np.pi * np.fft.fftfreq(len(x), dx); dk = 2 * np.pi / (len(x) * dx)


def check(f0, f1, label):
    """f0, f1: radial profiles as functions of s = |x| >= 0."""
    phi0, phi1 = f0(rr), f1(rr)
    P0 = np.empty(len(k), complex); P1 = np.empty(len(k), complex)
    for j in range(0, len(k), 500):                           # blocks keep memory small
        ker = 4 * np.pi * rr ** 2 * np.sinc(np.outer(k[j:j + 500], rr) / np.pi)   # sinc(z) = sin(pi z)/(pi z)
        P0[j:j + 500] = np.trapezoid(ker * phi0, rr, axis=1); P1[j:j + 500] = np.trapezoid(ker * phi1, rr, axis=1)
    hp = 0.5 * (k * P0 + 1j * P1); hm = 0.5 * (k * P0 - 1j * P1)   # h_pm = omega phi_hat_pm(0,k)
    meas = k ** 2 / (2 * np.pi ** 2)
    E3p = np.trapezoid(abs(hp) ** 2 * meas, k); E3m = np.trapezoid(abs(hm) ** 2 * meas, k)
    M3 = np.trapezoid(k * abs(hp) ** 2 * meas, k)
    d0 = np.gradient(phi0, rr)
    E3x = 0.5 * np.trapezoid((abs(phi1) ** 2 + abs(d0) ** 2) * 4 * np.pi * rr ** 2, rr)
    s = np.abs(x)
    psi0 = x * f0(s); psi1 = x * f1(s)
    u = 0.5 * (psi1 - np.gradient(psi0, x)); v = 0.5 * (psi1 + np.gradient(psi0, x))
    U = np.fft.fft(u) * dx; V = np.fft.fft(v) * dx
    E1m = (np.sum(abs(U[kk < 0]) ** 2) + np.sum(abs(V[kk > 0]) ** 2)) * dk / (2 * np.pi)
    E1 = (np.sum(abs(u) ** 2) + np.sum(abs(v) ** 2)) * dx
    M1 = (np.sum(kk[kk > 0] * abs(U[kk > 0]) ** 2) + np.sum(-kk[kk < 0] * abs(V[kk < 0]) ** 2)) * dk / (2 * np.pi)
    E3 = E3p + E3m
    wbar = M3 / E3
    print(f"{label}  E3(x)/E3(k)={E3x/E3:.6f}  E3/(2pi E1)={E3/(2*np.pi*E1):.6f}  E3-/(2pi E1-)={E3m/(2*np.pi*E1m):.6f}  "
          f"M3/(2pi M1)={M3/(2*np.pi*M1):.6f}  E3-/E3={E3m/E3:.2e}  E3-/bound={E3m/(C0*E3*np.exp(-np.pi*r*wbar)):.3g}")


# (i) random complex radial data
for trial in range(5):
    a = rng.normal(size=4) + 1j * rng.normal(size=4); b = rng.normal(size=4) + 1j * rng.normal(size=4)
    w = rng.uniform(2, 25, size=4)
    f0 = lambda s, a=a, w=w: bump(s / r) * sum(a[i] * np.exp(1j * w[i] * s * s) for i in range(4))
    f1 = lambda s, b=b, w=w: bump(s / r) * sum(b[i] * np.exp(1j * w[i] * s * s) for i in range(4))
    check(f0, f1, "(i)            ")

# (ii) positive-frequency dominated radial data (construction of Corollary 7.10)
for kappa in (8.0, 16.0, 24.0):
    c, wd = -0.3, 0.55
    # B(y) = bump((y-c)/wd) e^{i kappa y}; u = B', v(y) = -u(-y), so in closed form
    # psi_0(y) = int_{-inf}^y (v - u) = B(-y) - B(y) and psi_1(y) = u + v = B'(y) - B'(-y).
    B = lambda y, kappa=kappa: bump((y - c) / wd) * np.exp(1j * kappa * y)

    def dB(y, kappa=kappa):
        z = (y - c) / wd
        out = np.zeros(np.shape(y), complex); m = np.abs(z) < 1
        zm = z[m]; b = np.exp(-1 / (1 - zm ** 2))
        out[m] = (b * (-2 * zm / (1 - zm ** 2) ** 2) / wd + 1j * kappa * b) * np.exp(1j * kappa * y[m])
        return out

    B0 = -2 * dB(np.zeros(1))[0]                          # limit of (B(-s) - B(s))/s at s = 0
    f0 = lambda s, B=B, B0=B0: np.where(s > 0, (B(-s) - B(s)) / np.where(s > 0, s, 1), B0)
    f1 = lambda s, dB=dB: np.where(s > 0, (dB(s) - dB(-s)) / np.where(s > 0, s, 1), 0)   # value at 0 has weight 0
    check(f0, f1, f"(ii) kappa={kappa:4.1f}")

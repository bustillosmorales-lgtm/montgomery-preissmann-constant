"""Trigamma Gram symbol sigma(xi) = 4 sin^2(pi xi) psi_1(xi): check of the Fourier coefficients,
of lambda_max = 4 pi^2 - lambda_min, and of lambda_min(T_n(sigma)) ~ (8/3) log^2 n / n^2
(= (2 pi^4/3) * (2 log n/(pi^2 n))^2), against gamma * lambda_min(T_n(xi^2)) with gamma = 2 pi^4/3."""
import numpy as np, mpmath as mp
from scipy.linalg import eigvalsh
PI = np.pi
mp.mp.dps = 20
sig = lambda x: 4 * mp.sin(mp.pi * x) ** 2 * mp.polygamma(1, x)
d = lambda r: (r + 1) * mp.log(r + 1) + ((r - 1) * mp.log(r - 1) if r > 1 else 0) - 2 * r * mp.log(r)
for r in (0, 1, 2, 5):
    q = mp.quad(lambda x: sig(x) * mp.e ** (-2j * mp.pi * r * x), [0, 0.25, 0.5, 0.75, 1])
    cf = 2 * mp.pi ** 2 if r == 0 else -2j * mp.pi * d(r)
    print(f"sigma-hat({r}): quad {mp.nstr(q, 10)}   closed form {mp.nstr(cf, 10)}")
gam = 2 * PI ** 4 / 3
print(f"sigma(1-x)/x^2 at x=1e-3: {float(sig(1 - mp.mpf('1e-3')) / mp.mpf('1e-6')):.4f}  vs 2pi^4/3 = {gam:.4f}")
nmax = 4096
r = np.arange(1, nmax); dr = (r + 1) * np.log(r + 1) + np.where(r > 1, (r - 1) * np.log(np.maximum(r - 1, 1)), 0) - 2 * r * np.log(r)
print(f"\n{'n':>6} {'n^2 lam_min':>12} {'lam_min+lam_max-4pi^2':>22} {'ratio to (8/3)log^2 n':>22} {'ratio to gam*lam(xi^2)':>23}")
xi2 = {128: 4.11775, 256: 4.83780, 512: 5.60029, 1024: 6.40674, 2048: 7.25818, 4096: 8.15529}
for n in (128, 256, 512, 1024, 2048, 4096):
    j = np.arange(n); m = j[None, :] - j[:, None]          # m = k - j ; T_{jk} = sigma-hat(j-k)
    T = np.where(m == 0, 2 * PI ** 2, 0).astype(complex)
    am = np.abs(m); am1 = np.maximum(am, 1)
    val = dr[am1 - 1]
    T += np.where(m < 0, -2j * PI * val, 0) + np.where(m > 0, 2j * PI * val, 0)
    w = eigvalsh(T)
    y = n * n * w[0]
    print(f"{n:6d} {y:12.5f} {w[0] + w[-1] - 4 * PI ** 2:22.2e} {y / (8 / 3 * np.log(n) ** 2):22.4f} {y / (gam * xi2[n]):23.4f}")

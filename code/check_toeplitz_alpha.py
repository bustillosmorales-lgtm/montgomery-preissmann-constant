"""lambda_min(T_n(xi^alpha)) for alpha = 1/2 and 2 against the predicted (alpha log n/(pi^2 n))^alpha.
Reports y_n = n^alpha lambda_min and the ratio y_n / (alpha log n/pi^2)^alpha (-> 1, slowly),
and the local exponent of y_n in log n."""
import numpy as np, mpmath as mp
from scipy.linalg import eigvalsh
PI = np.pi
mp.mp.dps = 25
for alpha in (0.5, 2.0):
    s = alpha + 1
    nmax = 4096
    c = np.zeros(nmax, dtype=complex)
    c[0] = 1 / s
    for m in range(1, nmax):
        p = -2j * mp.pi * m
        c[m] = complex(p ** (-s) * mp.gammainc(s, 0, p))      # int_0^1 xi^alpha e^{2 pi i m xi}
    prev = None
    print(f"\nalpha={alpha}: prediction (alpha/pi^2)^alpha = {(alpha/PI**2)**alpha:.5f}")
    print(f"{'n':>6} {'n^a lam':>10} {'ratio':>8} {'d log y/d log log n':>20}")
    for n in (128, 256, 512, 1024, 2048, 4096):
        j = np.arange(n); d = j[None, :] - j[:, None]
        T = np.where(d >= 0, c[np.abs(d)], np.conj(c[np.abs(d)]))
        lam = eigvalsh(T, subset_by_index=[0, 0])[0]
        y = n ** alpha * lam
        pred = (alpha * np.log(n) / PI ** 2) ** alpha
        sl = (np.log(y) - np.log(prev[0])) / (np.log(np.log(n)) - np.log(np.log(prev[1]))) if prev else float('nan')
        print(f"{n:6d} {y:10.5f} {y/pred:8.4f} {sl:20.3f}")
        prev = (y, n)

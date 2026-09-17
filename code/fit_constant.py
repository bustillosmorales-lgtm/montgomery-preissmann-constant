"""Fits of x_n = lambda_min(T_n(xi)) * n = (pi - ||H_n||) n / (2 pi) against log n.

Conjecture: x_n = (1/pi^2) log n + lower order, i.e. (pi - ||H_n||) n / log n -> 2/pi.
Models (least squares on n >= n_min, lanczos rows, one row per n):
  M1  x = a + b L                       (free slope, no correction)
  M2  x = a + b L + c sqrt(L)           (free slope, sqrt(log) correction)
  M3  x = a + L/pi^2 + c sqrt(L)        (slope fixed at 1/pi^2)
  M4  x = a + b L + c log L             (free slope, log log correction)
with L = log n. Also the local slope dx/dL between consecutive doublings.
"""
import numpy as np

PI = np.pi
rows = {}
for line in open("results/hilbert_norm.csv").read().splitlines()[1:]:
    n, method, gap, lam, *_ = line.split(",")
    rows[int(n)] = float(lam)            # dense and lanczos agree; last one wins
n = np.array(sorted(rows))
lam = np.array([rows[k] for k in n])
x = lam * n
L = np.log(n)

print(f"{'n':>7} {'lam*n':>10} {'(pi-|H|)n/log n':>16} {'slope dx/dL':>12}")
for i in range(len(n)):
    s = (x[i] - x[i - 1]) / (L[i] - L[i - 1]) if i else float("nan")
    print(f"{n[i]:7d} {x[i]:10.6f} {2*PI*x[i]/L[i]:16.6f} {s:12.6f}")
print(f"\n1/pi^2 = {1/PI**2:.6f}    2/pi = {2/PI:.6f}")


def fit(cols, y):
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    return coef, np.sqrt(np.mean(res ** 2)), np.abs(res).max()


for nmin in (256, 1024, 4096):
    m = n >= nmin
    Lm, xm = L[m], x[m]
    one = np.ones_like(Lm)
    print(f"\n--- fits on n >= {nmin} ({m.sum()} points) ---")
    c, rms, mx = fit([one, Lm], xm)
    print(f"M1 a+bL          : b={c[1]:.5f} (b*pi^2={c[1]*PI**2:.3f})            rms={rms:.1e} max={mx:.1e}")
    c, rms, mx = fit([one, Lm, np.sqrt(Lm)], xm)
    print(f"M2 a+bL+c sqrtL  : b={c[1]:.5f} (b*pi^2={c[1]*PI**2:.3f}) c={c[2]:+.4f}  rms={rms:.1e} max={mx:.1e}")
    c, rms, mx = fit([one, np.sqrt(Lm)], xm - Lm / PI ** 2)
    print(f"M3 a+L/pi^2+c sqrtL: a={c[0]:+.4f} c={c[1]:+.4f}                  rms={rms:.1e} max={mx:.1e}")
    c, rms, mx = fit([one, Lm, np.log(Lm)], xm)
    print(f"M4 a+bL+c logL   : b={c[1]:.5f} (b*pi^2={c[1]*PI**2:.3f}) c={c[2]:+.4f}  rms={rms:.1e} max={mx:.1e}")


# ------------------------------------------------------------------ Legendre cross-check
# Local-model duality: x_n = lambda_min n ~ min_ell [ n mu(ell) + ell ],  dx/dlog n = n mu = 1/C_loc(ell*).
# mu(ell) from the arbitrary-precision curve (results/tradeoff_mp.csv, n = 240), interpolated in log mu.
print("\n--- Legendre cross-check against the leak-moment curve (mp, n=240) ---")
cur = [l.split(",") for l in open("results/tradeoff_mp.csv").read().splitlines()[1:]]
cur = sorted([(float(r[4]), float(r[3])) for r in cur if r[0] == "240"])
ell_c = np.array([c[0] for c in cur]); lmu_c = np.log([c[1] for c in cur])
eg = np.linspace(ell_c[0], ell_c[-1], 20001)
lmu = np.interp(eg, ell_c, lmu_c)


def x_pred(logn):
    v = np.exp(logn + lmu) + eg
    i = v.argmin()
    return v[i], eg[i]


print(f"{'n':>9} {'x_n (Lanczos)':>14} {'x_n (Legendre)':>15} {'ell*':>7}")
for k, xv in zip(n, x):
    if k >= 256:
        xp, es = x_pred(np.log(k))
        print(f"{k:9d} {xv:14.5f} {xp:15.5f} {es:7.3f}")
print(f"\n{'log n':>6} {'x (Legendre)':>13} {'ell*':>7} {'dx/dlog n':>10} {'x/log n':>9}   (1/pi^2 = {1/PI**2:.5f})")
prev = None
for ln in (10, 20, 30, 40, 50, 60, 70, 80):
    xp, es = x_pred(ln)
    sl = (xp - prev[0]) / (ln - prev[1]) if prev else float("nan")
    print(f"{ln:6d} {xp:13.4f} {es:7.3f} {sl:10.5f} {xp/ln:9.5f}")
    prev = (xp, ln)

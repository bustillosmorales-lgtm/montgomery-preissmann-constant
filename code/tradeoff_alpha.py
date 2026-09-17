"""Gate 0 killer test for the generalisation to moments of order alpha.

Conjecture (alpha > 0):  leak mu >= c exp(-pi^2 * D_alpha^{1/alpha}),  D_alpha = int_{t>0} t^alpha |F|^2.
For alpha >= 1 it follows from alpha = 1 by Jensen, so the informative case is alpha < 1.
Discrete local model as in tradeoff_leak_moment.py:
   mu = int_{1/2}^1 |p|^2,   ell_alpha = n^alpha int_0^{1/2} xi^alpha |p|^2.
Lower envelope of (ell_alpha, mu) via bottom eigenvectors of A + t B_alpha (python-flint, 320 bits).
Report s = ell_alpha^{1/alpha} and the local slope d log(1/mu) / d s  (conjecture: -> pi^2 = 9.87, never
substantially above). Output: results/tradeoff_alpha.txt / .csv
"""
import sys
import numpy as np
import mpmath as mp
from multiprocessing import Pool

PI = np.pi
PREC = 320


def moment_entries(n, alpha, dps=110):
    """I[m] = n^alpha int_0^{1/2} xi^alpha e^{2 pi i m xi} d xi for m = -(n-1)..(n-1), as mpc strings."""
    mp.mp.dps = dps
    a = mp.mpf(1) / 2
    s = mp.mpf(alpha) + 1
    out = {}
    for m in range(0, n):
        if m == 0:
            val = a ** s / s
        else:
            p = -2j * mp.pi * m                      # int_0^a xi^{s-1} e^{-p xi} = p^{-s} gamma(s, 0, p a)
            val = p ** (-s) * mp.gammainc(s, 0, p * a)
        val = val * mp.mpf(n) ** alpha
        out[m] = val
        out[-m] = mp.conj(val)
    return {k: (mp.nstr(v.real, 100), mp.nstr(v.imag, 100)) for k, v in out.items()}


def point(args):
    n, alpha, t_exp, ent = args
    import flint
    flint.ctx.prec = PREC
    acb, acb_mat, arb = flint.acb, flint.acb_mat, flint.arb
    pi = arb.pi()
    t = arb(10) ** t_exp
    RA, RM = [], []
    for j in range(n):
        ra, rm = [], []
        for k in range(n):
            m = k - j
            if m == 0:
                a = acb(arb(1) / 2)
            elif m % 2 == 0:
                a = acb(0)
            else:
                a = acb(0, -1) / (pi * m)
            re, im = ent[m]
            b = acb(arb(re), arb(im))
            ra.append(a)
            rm.append(a + t * b)
        RA.append(ra); RM.append(rm)
    A = acb_mat(RA); M = acb_mat(RM)
    Minv = M.inv()
    Minv = acb_mat([[Minv[i, k].mid() for k in range(n)] for i in range(n)])

    def normalise(v):
        s = sum((abs(v[i, 0]) ** 2 for i in range(n)), arb(0))
        v = v * acb(1 / s.sqrt())
        return acb_mat([[v[i, 0].mid()] for i in range(n)])

    u = normalise(acb_mat([[acb(1.0 / (1 + k))] for k in range(n)]))
    prev = None
    for it in range(40000):
        u = normalise(Minv * u)
        if it % 50 == 49 and it >= 599:
            uh = u.conjugate().transpose()
            mu = (uh * A * u)[0, 0].real.mid()
            if prev is not None and abs(float((mu - prev) / mu)) < 1e-12:
                break
            prev = mu
    uh = u.conjugate().transpose()
    mu = (uh * A * u)[0, 0].real
    lam = (uh * M * u)[0, 0].real
    ell = (lam - mu) / t
    return (n, alpha, t_exp, float(mu.mid()), float(ell.mid()))


if __name__ == "__main__":
    # sanity of the closed form against quadrature
    mp.mp.dps = 30
    for alpha in (0.5, 2.0):
        m = 7
        q = mp.quad(lambda x: x ** alpha * mp.e ** (2j * mp.pi * m * x), [0, 0.125, 0.25, 0.375, 0.5])
        cf = mp.mpc(*[mp.mpf(z) for z in moment_entries(1 + m, alpha, dps=30)[m]]) / mp.mpf(1 + m) ** alpha
        print(f"closed form check alpha={alpha}: |diff| = {mp.nstr(abs(q - cf), 3)}")

    rows = []
    with Pool(20) as pool:
        for alpha in (0.5, 2.0):
            for n in (120, 180):
                ent = moment_entries(n, alpha)
                exps = [x / 2 for x in range(-2, -73, -3)]
                rows += pool.map(point, [(n, alpha, e, ent) for e in exps])
    rows.sort()
    with open("results/tradeoff_alpha.csv", "w") as fh:
        fh.write("n,alpha,log10_t,mu,ell_alpha\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    for alpha in (0.5, 2.0):
        for n in (120, 180):
            rr = sorted([r for r in rows if r[0] == n and r[1] == alpha], key=lambda r: r[4])
            print(f"\nalpha={alpha} n={n}:  s = ell^(1/alpha),  log(1/mu),  ratio log(1/mu)/(pi^2 s),  local slope d log(1/mu)/ds")
            prev = None
            for r in rr:
                s = r[4] ** (1 / alpha); lg = -np.log(r[3])
                sl = (lg - prev[1]) / (s - prev[0]) if prev and s - prev[0] > 1e-9 else float("nan")
                print(f"   s={s:9.4f}  log(1/mu)={lg:9.3f}  ratio={lg/(PI**2*s):6.3f}  slope={sl:7.3f}")
                prev = (s, lg)

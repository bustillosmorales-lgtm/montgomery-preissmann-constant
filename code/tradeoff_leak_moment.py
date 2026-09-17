"""Leak-moment trade-off curve for polynomials p(xi) = sum_{k<n} u_k e^{2 pi i k xi}, ||p||_2 = 1.

   leak    mu  = int_{1/2}^{1} |p|^2           (the side t<0 of the local model, t = n xi)
   moment  ell = n int_0^{1/2} xi |p|^2        (the side t>0)

For t > 0 the minimiser of mu + t*ell is the bottom eigenvector of A + tB. Sweeping t traces the lower
envelope log(1/mu) as a function of ell. Local-model predictions:
   * the curve does not depend on n (it is the local model at scale 1/n);
   * Theorem (sharp):  log(1/mu) <= pi^2 ell + log 2 + pi r0   (in the local model);
   * conjectured slope  d log(1/mu) / d ell -> pi^2 = 9.8696.
Double precision (numpy) for n = 300, 600, 1200 down to mu ~ 1e-12;
arbitrary precision (python-flint, 320 bits) for n = 120, 180, 240 down to mu ~ 1e-36.
Output: results/tradeoff_double.csv, results/tradeoff_mp.csv
"""
import sys
import numpy as np
from multiprocessing import Pool

PI = np.pi
R0 = 0.08863802
CONST = np.log(2) + PI * R0


def entries_np(n):
    j = np.arange(n)
    m = (j[None, :] - j[:, None]).astype(float)      # m = k - j
    mi = np.where(m == 0, 1.0, m)
    sgn = np.where(np.abs(m) % 2 == 1, -1.0, 1.0)     # (-1)^m
    A = np.where(m == 0, 0.5, (1 - sgn) / (2j * PI * mi))
    c = 2 * PI * mi
    B = np.where(m == 0, n / 8.0, n * (0.5 * sgn / (1j * c) + (sgn - 1) / c ** 2))
    return A, B


def double_curve(n, ts):
    A, B = entries_np(n)
    out = []
    for t in ts:
        w, V = np.linalg.eigh(A + t * B)
        u = V[:, 0]
        mu = np.real(np.conj(u) @ A @ u)
        ell = np.real(np.conj(u) @ B @ u)
        out.append((n, t, w[0], mu, ell))
    return out


def mp_point(args):
    n, t_exp, prec = args
    import flint
    flint.ctx.prec = prec
    acb, acb_mat, arb = flint.acb, flint.acb_mat, flint.arb
    pi = arb.pi()
    t = arb(10) ** t_exp
    rows_A, rows_M = [], []
    for j in range(n):
        ra, rm = [], []
        for k in range(n):
            m = k - j
            if m == 0:
                a = acb(arb(1) / 2)
                b = acb(arb(n) / 8)
            else:
                s = 1 if m % 2 == 0 else -1
                a = acb(0) if s == 1 else acb(0, -1) / (pi * m)
                c = 2 * pi * m
                b = n * (acb(arb(s) / 2) / acb(0, c) + acb((s - 1) / c ** 2))
            ra.append(a)
            rm.append(a + t * b)
        rows_A.append(ra)
        rows_M.append(rm)
    A = acb_mat(rows_A)
    M = acb_mat(rows_M)
    Minv = M.inv()
    # work with ball midpoints: interval radii would otherwise grow with every iteration
    Minv = acb_mat([[Minv[i, k].mid() for k in range(n)] for i in range(n)])
    u = acb_mat([[acb(1.0 / (1 + k))] for k in range(n)])

    def normalise(v):
        s = sum((abs(v[i, 0]) ** 2 for i in range(n)), arb(0))
        v = v * acb(1 / s.sqrt())
        return acb_mat([[v[i, 0].mid()] for i in range(n)])

    u = normalise(u)
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
    return (n, t_exp, float(lam.mid()), float(mu.mid()), float(ell.mid()), it + 1)


def report(rows, label):
    print(f"\n[{label}]  {'n':>5} {'t':>9} {'mu':>11} {'ell':>9} {'log(1/mu)':>10} {'pi^2 ell+C':>11} {'margin':>8}")
    for r in rows:
        n, t, lam, mu, ell = r[:5]
        lg = -np.log(mu)
        tt = t if label == "double" else 10.0 ** t
        print(f"         {n:5d} {tt:9.1e} {mu:11.4e} {ell:9.5f} {lg:10.4f} {PI**2*ell+CONST:11.4f} {PI**2*ell+CONST-lg:8.4f}")


if __name__ == "__main__":
    ts = 10.0 ** np.arange(-1, -12.5, -0.5)
    dbl = []
    for n in (300, 600, 1200):
        dbl += double_curve(n, ts)
    report(dbl, "double")
    with open("results/tradeoff_double.csv", "w") as fh:
        fh.write("n,t,lambda,mu,ell\n")
        for r in dbl:
            fh.write(",".join(f"{x:.15e}" if isinstance(x, float) else str(x) for x in r) + "\n")

    exps = [x / 2 for x in range(-2, -73, -3)]      # t = 10^-1 ... 10^-36
    jobs = [(n, e, 320) for n in (120, 180, 240) for e in exps]
    with Pool(20) as pool:
        mpr = pool.map(mp_point, jobs)
    mpr.sort()
    report(mpr, "mp")
    with open("results/tradeoff_mp.csv", "w") as fh:
        fh.write("n,log10_t,lambda,mu,ell,iterations\n")
        for r in mpr:
            fh.write(",".join(f"{x:.15e}" if isinstance(x, float) else str(x) for x in r) + "\n")

    # local slopes of the envelope (mp data): d log(1/mu) / d ell
    print("\nlocal slope d log(1/mu)/d ell  (mp)")
    for n in (120, 180, 240):
        rr = sorted([r for r in mpr if r[0] == n], key=lambda r: r[4])
        for a, b in zip(rr[:-1], rr[1:]):
            if b[4] - a[4] > 1e-9:
                print(f"  n={n:4d}  ell {a[4]:7.3f}->{b[4]:7.3f}  slope {(np.log(a[3]) - np.log(b[3])) / (b[4] - a[4]):7.3f}")

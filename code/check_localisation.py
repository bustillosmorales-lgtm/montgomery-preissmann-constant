"""Check of the localisation lemma, computed entirely in the Mellin variable rho.
If the Lambda-spectrum of phi lies in [L, inf), then
      P(t < r) = int_{t<r}|F|^2 = <G psi, m(Lambda) G psi>,  G = exp(2 pi i r x(rho)),  x = 1/(1+e^{-rho}),
and the lemma claims  P(t < r) <= exp(-4 L h(theta)),  theta = pi r/(2L) < 1,
h(theta) = arccos(sqrt(theta)) - sqrt(theta(1-theta)).
psi-hat = smooth bump on [L, L+W] times a random smooth phase; no s-grid is involved.
Also checks the identity P(t<0) = <psi, m(Lambda) psi> against e^{-2 pi L}.
"""
import numpy as np

PI = np.pi
rng = np.random.default_rng(7)
RHO = np.linspace(-150, 150, 30001); DR = RHO[1] - RHO[0]


def bump(x):
    out = np.zeros_like(x); m = (x > 0) & (x < 1)
    out[m] = np.exp(-1 / (x[m] * (1 - x[m])))
    return out


def ft(vals, grid_from, dgrid, grid_to, sign):
    out = np.empty(grid_to.size, dtype=complex)
    for i in range(0, grid_to.size, 500):
        out[i:i + 500] = (vals[None, :] * np.exp(sign * 1j * grid_to[i:i + 500, None] * grid_from[None, :])).sum(1) * dgrid
    return out / np.sqrt(2 * PI)


def h(th):
    return np.arccos(np.sqrt(th)) - np.sqrt(th * (1 - th))


print(f"{'L':>4} {'W':>4} {'theta':>6} {'r':>7} {'P(t<r)':>11} {'bound':>11} {'ok':>3}")
worst = 0
for L, W in [(1.0, 1.0), (2.0, 1.5), (3.0, 2.0), (4.0, 1.0), (5.0, 3.0)]:
    lam = np.linspace(L, L + W, 801); dl = lam[1] - lam[0]
    ph = rng.normal(size=4)
    coef = bump((lam - L) / W) * np.exp(1j * (ph[0] * (lam - L) + ph[1] * (lam - L) ** 2 + ph[2] * np.sin(3 * (lam - L))))
    coef /= np.sqrt((abs(coef) ** 2).sum() * dl)
    psi = ft(coef, lam, dl, RHO, -1)                      # psi(rho) = (2pi)^{-1/2} int coef e^{-i lam rho}
    nrm = (abs(psi) ** 2).sum() * DR
    LAM = np.linspace(-25, L + W + 6, 6001); DL = LAM[1] - LAM[0]
    m = 1 / (1 + np.exp(2 * PI * LAM))
    for th in (0.0, 0.3, 0.6, 0.8, 0.9, 0.95):
        r = 2 * L * th / PI
        G = np.exp(2j * PI * r / (1 + np.exp(-RHO)))
        gh = ft(G * psi, RHO, DR, LAM, +1)
        tot = (abs(gh) ** 2).sum() * DL
        Pr = (m * abs(gh) ** 2).sum() * DL / nrm
        b = np.exp(-4 * L * h(th)) if th > 0 else np.exp(-2 * PI * L)
        worst = max(worst, Pr / b)
        print(f"{L:4.1f} {W:4.1f} {th:6.2f} {r:7.3f} {Pr:11.3e} {b:11.3e} {'y' if Pr <= b else 'NO':>3}   (captured mass {tot/nrm:.6f})")
print(f"\nmax P/bound = {worst:.3g}  (must be <= 1)")

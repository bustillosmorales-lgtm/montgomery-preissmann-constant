"""Numerical checks of Section 'The local model' (section_local_model.tex).

Checks, each against an independent computation:
  1. formula (eq:sinh):  PV int e^{i w x}/sinh x dx = i pi tanh(pi w/2)
  2. r0 = sup_{l>=0} 2l/(e^{2 pi l}+1) and its argmax
  3. Lemma KP: H_[0,1] = -tanh(pi Lambda), by direct principal-value quadrature
  4. Corollary leak: mu[phi] computed from F(t) (FFT) equals <m(Lambda)>
  5. Key identity: D[phi] - (2/pi)<Lambda+> = D[c phi] - <R>/pi
  6. Theorem sharp: mu >= (1/2) e^{-pi r0} e^{-pi^2 D}   (constant 0.378)
  7. Remark (Gaussian packets): leak ~ e^{-2 pi l0 + 2 pi^2 s^2}, D ~ (l0/pi)(1+e^{d^2/2})
Test functions: phi = U^{-1} psi, psi a sum of Gaussian packets in rho = log(s/(1-s)).
"""
import numpy as np
import mpmath as mp

PI = np.pi
rng = np.random.default_rng(20260917)


def header(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ---------------------------------------------------------------- 1. sinh pair
header("1. PV int e^{iwx}/sinh(x) dx  =  i pi tanh(pi w/2)")
mp.mp.dps = 30
worst = 0
for w in [0.1, 0.5, 1.0, 2.0, 3.7]:
    # odd kernel: PV integral = 2i int_0^inf sin(wx)/sinh(x) dx (integrand regular at 0)
    val = 2 * mp.quad(lambda x: mp.sin(w * x) / mp.sinh(x), [0, 1, 5, 20, 60])
    ref = mp.pi * mp.tanh(mp.pi * w / 2)
    worst = max(worst, abs(val - ref))
    print(f"  w={w:4.1f}  numeric/i={mp.nstr(val, 16)}  pi*tanh(pi w/2)={mp.nstr(ref, 16)}")
print(f"  max |diff| = {mp.nstr(worst, 3)}")

# ---------------------------------------------------------------- 2. r0
header("2. r0 = sup 2l/(e^{2 pi l}+1)")
f = lambda l: 2 * l / (mp.e ** (2 * mp.pi * l) + 1)
lstar = mp.findroot(lambda l: mp.diff(f, l), 0.2)
r0 = float(f(lstar))
print(f"  argmax = {mp.nstr(lstar, 8)}   r0 = {r0:.8f}   r0/pi = {r0/PI:.8f}")
print(f"  (1/2)e^(-pi r0) = {0.5*np.exp(-PI*r0):.6f}    (1/2)e^(-2 r0) = {0.5*np.exp(-2*r0):.6f}")

# ---------------------------------------------------------------- packets
RHO = np.linspace(-40, 40, 16001)
DRHO = RHO[1] - RHO[0]
LAM = np.linspace(-12, 12, 4801)
DLAM = LAM[1] - LAM[0]


def psi_of(packets, rho):
    out = np.zeros_like(rho, dtype=complex)
    for (A, r0_, d, l0) in packets:
        # |psi|^2 has variance d^2 in rho; spectral measure centred at l0 (e_lambda = e^{-i lambda rho})
        out += A * np.exp(-(rho - r0_) ** 2 / (4 * d * d)) * np.exp(-1j * l0 * rho)
    return out


def psi_hat(psi_vals):
    out = np.empty(LAM.size, dtype=complex)
    for i in range(0, LAM.size, 400):
        L = LAM[i:i + 400, None]
        out[i:i + 400] = (np.exp(1j * L * RHO[None, :]) * psi_vals[None, :]).sum(1) * DRHO
    return out / np.sqrt(2 * PI)


def spectral(packets):
    ps = psi_of(packets, RHO)
    nrm = np.sqrt((abs(ps) ** 2).sum() * DRHO)
    ph = psi_hat(ps) / nrm
    w = abs(ph) ** 2
    m = 1 / (1 + np.exp(2 * PI * LAM))
    r = abs(LAM) * (1 - np.tanh(PI * abs(LAM)))
    return dict(norm_hat=w.sum() * DLAM, mu=(m * w).sum() * DLAM,
                Lp=(np.maximum(LAM, 0) * w).sum() * DLAM, R=(r * w).sum() * DLAM,
                nrm=nrm, ph=ph)


N = 2 ** 15          # samples of phi on (0,1)
P = 64               # t-step 1/P
S = (np.arange(N) + 0.5) / N


def phi_on_grid(packets, nrm):
    a = S * (1 - S)
    rho = np.log(S / (1 - S))
    return psi_of(packets, rho) / np.sqrt(a) / nrm


def F_of(phi):
    M = N * P
    x = np.zeros(M, dtype=complex)
    x[:N] = phi
    k = np.arange(M)
    F = (M * np.fft.ifft(x)) * np.exp(1j * PI * k / M) / N
    t = np.where(k < M // 2, k, k - M) / P
    return t, F


def time_side(phi):
    # |F|^2 is smooth, but the leak integral is cut at t=0 where |F(0)|^2 >> mu:
    # trapezoid with Euler-Maclaurin endpoint corrections (O(h^4)) at t=0.
    t, F = F_of(phi)
    h = 1 / P
    f = abs(F) ** 2
    i0 = 0                       # index of t=0
    fm1, f1, f2, fm2 = f[-1], f[1], f[2], f[-2]
    d1 = (-f2 + 8 * f1 - 8 * fm1 + fm2) / (12 * h)       # f'(0)
    mu = h * (f[t < 0].sum() + f[i0] / 2) - h * h / 12 * d1
    D = h * (t[t > 0] * f[t > 0]).sum() + h * h / 12 * f[i0]   # (t f)'(0) = f(0)
    return dict(norm=h * f.sum(), mu=mu, D=D)


# ---------------------------------------------------------------- 3. Lemma KP directly
header("3. Lemma KP: (Hg)(s) = (1/(i pi)) PV int_0^1 g(s')/(s'-s) ds'  vs  U^{-1}[-tanh(pi Lambda)] U g")
pk = [(1.0, 0.3, 0.8, 0.6), (0.7, -1.0, 0.6, -0.4)]
sp = spectral(pk)
Ns = 200000
ss = (np.arange(Ns) + 0.5) / Ns
a_s = ss * (1 - ss)
g = psi_of(pk, np.log(ss / (1 - ss))) / np.sqrt(a_s) / sp["nrm"]
gp = np.gradient(g, ss)
evals = np.linspace(0.08, 0.92, 15)
maxerr = 0
for s0 in evals:
    g0 = np.interp(s0, ss, g.real) + 1j * np.interp(s0, ss, g.imag)
    d = ss - s0
    q = np.where(abs(d) < 0.5 / Ns, np.interp(s0, ss, gp.real) + 1j * np.interp(s0, ss, gp.imag), (g - g0) / np.where(d == 0, 1, d))
    pv = q.sum() / Ns + g0 * np.log((1 - s0) / s0)
    Hg = pv / (1j * PI)
    rho0 = np.log(s0 / (1 - s0))
    psiH = ((-np.tanh(PI * LAM)) * sp["ph"] * np.exp(-1j * LAM * rho0)).sum() * DLAM / np.sqrt(2 * PI)
    spec = psiH / np.sqrt(s0 * (1 - s0))
    maxerr = max(maxerr, abs(Hg - spec) / max(abs(spec), 1e-12))
    wrong = -spec  # opposite sign
    if s0 in evals[[0, 7, 14]]:
        print(f"  s={s0:.2f}  PV={Hg: .6f}  -tanh={spec: .6f}  (+tanh would give {wrong: .6f})")
print(f"  max relative error over 15 points = {maxerr:.2e}")

# ---------------------------------------------------------------- 4-6 on many packets
header("4-6. Leak (Cor.), key identity (Thm key), sharp inequality (Thm sharp)")
tests = [
    [(1, 0.0, 0.7, 0.0)],
    [(1, 0.0, 0.5, 1.0)],
    [(1, 0.0, 0.8, 2.0)],
    [(1, 0.0, 1.2, 3.0)],
    [(1, 1.5, 0.6, 1.5)],
    [(1, -2.0, 0.6, -1.0)],
    [(1, 0.0, 2.0, 0.2)],
    [(1, 0.0, 0.3, 0.2)],
]
for _ in range(8):
    k = rng.integers(1, 4)
    tests.append([(rng.normal() + 1j * rng.normal(), rng.uniform(-2, 2), rng.uniform(0.35, 1.2), rng.uniform(-1.5, 3.0)) for _ in range(k)])

bound_c = 0.5 * np.exp(-PI * r0)
print(f"  {'#':>2} {'mu(time)':>11} {'mu(spec)':>11} {'relerr':>8} | {'LHS key':>10} {'RHS key':>10} {'diff':>9} | {'mu/bound':>9}")
worst_mu = worst_key = 0
min_ratio = np.inf
for i, pk in enumerate(tests):
    sp = spectral(pk)
    phi = phi_on_grid(pk, sp["nrm"])
    ts = time_side(phi)
    ts_c = time_side((1 - 2 * S) * phi)
    lhs = ts["D"] - 2 / PI * sp["Lp"]
    rhs = ts_c["D"] - sp["R"] / PI
    rel = abs(ts["mu"] - sp["mu"]) / sp["mu"]
    ratio = ts["mu"] / (bound_c * np.exp(-PI ** 2 * ts["D"]))
    worst_mu = max(worst_mu, rel); worst_key = max(worst_key, abs(lhs - rhs)); min_ratio = min(min_ratio, ratio)
    print(f"  {i:2d} {ts['mu']:11.4e} {sp['mu']:11.4e} {rel:8.1e} | {lhs:10.6f} {rhs:10.6f} {lhs-rhs:9.1e} | {ratio:9.3g}")
    assert abs(ts["norm"] - 1) < 1e-6 and abs(sp["norm_hat"] - 1) < 1e-6
print(f"  worst leak relerr = {worst_mu:.1e};  worst |key identity| = {worst_key:.1e};  min mu/bound = {min_ratio:.3g} (must be >= 1)")
print(f"  key RHS >= -r0/pi = {-r0/PI:.5f} in all tests: see column RHS")

# ---------------------------------------------------------------- 7 Remark packets
header("7. Remark sharpness: Gaussian packet at rho=0, |psi|^2 variance d^2, sigma^2 = 1/(4 d^2)")
print(f"  {'l0':>4} {'d':>4} {'mu':>11} {'e^(-2pi l0+2pi^2 s^2)':>22} {'D':>9} {'(l0/pi)(1+e^(d^2/2))':>21} {'log(1/mu)/(pi^2 D)':>19}")
for l0, d in [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (3.0, 0.7), (3.0, 1.5), (4.0, 1.0)]:
    pk = [(1, 0.0, d, l0)]
    sp = spectral(pk)
    ts = time_side(phi_on_grid(pk, sp["nrm"]))
    s2 = 1 / (4 * d * d)
    print(f"  {l0:4.1f} {d:4.1f} {ts['mu']:11.4e} {np.exp(-2*PI*l0+2*PI**2*s2):22.4e} {ts['D']:9.5f} {l0/PI*(1+np.exp(d*d/2)):21.5f} {np.log(1/ts['mu'])/(PI**2*ts['D']):19.4f}")

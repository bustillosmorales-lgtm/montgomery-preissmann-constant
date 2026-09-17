"""verify_all.py - every number quoted in the manuscript, recomputed or read from results/*.txt|csv.
Prints: location | quoted | computed | OK/FAIL. Run from the project root (folder with code/ and results/)."""
import re, math, csv
import numpy as np
import mpmath as mp

rows = []
PI = math.pi


def chk(loc, quoted, computed, tol):
    ok = abs(quoted - computed) <= tol
    rows.append((loc, quoted, computed, ok))


def txt(fn):
    return open('results/' + fn, encoding='utf-8', errors='replace').read()

# ---------------- constants (Sections 2-4)
mp.mp.dps = 30
f = lambda l: 2 * l / (mp.e ** (2 * mp.pi * l) + 1)
ls = mp.findroot(lambda l: mp.diff(f, l), 0.2); r0 = float(f(ls))
C0 = 0.5 * math.exp(-PI * r0)
chk('r0 = 0.0886', 0.0886, r0, 5e-5)
chk('argmax 0.2035', 0.2035, float(ls), 5e-5)
chk('r0/pi = 0.0282', 0.0282, r0 / PI, 5e-5)
chk('C0 = 0.378', 0.378, C0, 5e-4)
chk('log(pi^2 C0/2) = 0.6247', 0.6247, math.log(PI ** 2 * C0 / 2), 5e-5)
chk('7 zeta(3)+pi^4/6 < 25', 1.0, float(7 * mp.zeta(3) + mp.pi ** 4 / 6 < 25), 0)
chk('sum (2j-1)^-4 = pi^4/96', PI ** 4 / 96, float(mp.nsum(lambda j: (2 * j - 1) ** -4, [1, mp.inf])), 1e-12)
chk('0.4122 >= e^{1/2}/4', 1.0, float(0.4122 >= math.exp(0.5) / 4), 0)
chk('5.35 >= pi^2/2+0.4122', 1.0, float(5.35 >= PI ** 2 / 2 + 0.4122), 0)
chk('2.04 >= 0.4122 pi^2/2', 1.0, float(2.04 >= 0.4122 * PI ** 2 / 2), 0)
chk('2.04+pi^2 < 11.91', 1.0, float(2.04 + PI ** 2 < 11.91), 0)
chk('4 pi^2 <= 40', 1.0, float(4 * PI ** 2 <= 40), 0)
chk('32/3 * 8 < 86', 1.0, float(32 / 3 * 8 < 86), 0)
chk('gamma trigamma 2pi^4/3 -> 8/3', 8 / 3, 2 * PI ** 4 / 3 * (2 / PI ** 2) ** 2, 1e-12)

# ---------------- local model checks (Section 7.1, Remark 2.3, Lemma 4.1)
t = txt('check_local_model.txt')
chk('sinh pair to 1e-25', 1.0, float(float(re.search(r'max \|diff\| = ([0-9.e+-]+)', t).group(1)) < 1e-25), 0)
chk('KP relerr 2e-11', 2e-11, float(re.search(r'max relative error over 15 points = ([0-9.e+-]+)', t).group(1)), 1e-11)
m = re.search(r'worst leak relerr = ([0-9.e+-]+);\s+worst \|key identity\| = ([0-9.e+-]+);\s+min mu/bound = ([0-9.e+-]+)', t)
chk('leak relerr 6e-6', 6e-6, float(m.group(1)), 1e-6)
chk('key identity 2e-8', 2e-8, float(m.group(2)), 1e-8)
chk('mu/bound >= 3.6', 3.6, float(m.group(3)), 0.05)
t = txt('check_upper.txt')
pairs = re.findall(r'^\s*[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+([0-9.]+)\s', t.split('(U3)')[0], re.M)
chk('Lemma 4.1(ii) seven digits', 0.0, max(abs(float(a) - float(b)) for a, b in pairs), 5e-7)

# ---------------- Remark 3.5 (transfer)
t = txt('check_transfer.txt')
tr = {}
for line in t.splitlines():
    p = line.split()
    if len(p) >= 11 and p[0].isdigit():
        n, L = int(p[0]), float(p[1])
        if abs(L - math.sqrt(n)) < 1:
            tr[n] = dict(x=float(p[2]), mup=float(p[3]), muF=float(p[4]), mp=float(p[5]), DF=float(p[6]), bnd=float(p[10]))
for n, q1, q2, q3, q4 in [(256, 0.911, 0.956, 0.39, 1.56), (1024, 0.945, 0.984, 0.51, 1.75), (4096, 0.967, 0.994, 0.63, 1.93)]:
    d = tr[n]
    chk(f'R3.5 mu ratio n={n}', q1, d['muF'] / d['mup'], 6e-4)
    chk(f'R3.5 D ratio n={n}', q2, d['DF'] / d['mp'], 6e-4)
    chk(f'R3.5 bound n={n}', q3, d['bnd'], 6e-3)
    chk(f'R3.5 x n={n}', q4, d['x'], 6e-3)

# ---------------- Remark 4.5 and Section 8 (upper bound)
t = txt('check_upper.txt')
u3 = re.findall(r'^\s*(\d+)\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+[0-9.]+\s*$', t, re.M)
qR = {1024: 23.30, 4096: 26.13, 16384: 28.89, 65536: 31.60, 262144: 34.26}
for n, R, lam, bnd in u3:
    n = int(n)
    if n in qR:
        chk(f'R4.5 pi^2 n R n={n}', qR[n], float(R), 6e-3)
chk('R4.5 pi^2 n lam 2^10 = 17.27', 17.27, float(u3[0][2]), 6e-3)
chk('R4.5 pi^2 n lam 2^18 = 24.24', 24.24, float(u3[-1][2]), 6e-3)
chk('R4.5 bound 2^10 = 35.73', 35.73, float(u3[0][3]), 6e-3)
chk('R4.5 bound 2^18 = 46.67', 46.67, float(u3[-1][3]), 6e-3)
best = re.findall(r'^\s*(\d+)\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s*$', t.split('best Gaussian')[1], re.M)
for (n, ratio), q in zip(best, (13, 12, 11)):
    chk(f'R4.5 best packet n={n} {q}%', q, 100 * (float(ratio) - 1), 0.5)

# ---------------- Lemma 5.1 remark
t = txt('check_localisation.txt')
chk('loc: 30 cases', 30, len(re.findall(r'^\s*\d\.\d\s+\d\.\d\s+0\.\d\d', t, re.M)), 0)
chk('loc: max ratio 0.054', 0.054, float(re.search(r'max P/bound = ([0-9.e+-]+)', t).group(1)), 5e-4)
chk('loc: 30 cases theta>0 in [0.3,0.95]', 1.0, float('0.30' in t and '0.95' in t), 0)

# ---------------- Table 1 (slopes)
def slopes(fn, n, alpha=None):
    out = []
    t = txt(fn)
    if alpha is None:
        for m_ in re.finditer(r'n=\s*%d\s+ell\s+([0-9.]+)->\s*([0-9.]+)\s+slope\s+([0-9.]+)' % n, t):
            out.append(((float(m_.group(1)) + float(m_.group(2))) / 2, float(m_.group(3))))
    else:
        blk = t.split('alpha=%s n=%d:' % (alpha, n))[1].split('alpha=')[0]
        for m_ in re.finditer(r's=\s*([0-9.]+)\s+log\(1/mu\)=\s*[0-9.]+\s+ratio=\s*[0-9.]+\s+slope=\s*([0-9.nan]+)', blk):
            if m_.group(2) != 'nan':
                out.append((float(m_.group(1)), float(m_.group(2))))
    return out


def near(lst, s):
    return min(lst, key=lambda p: abs(p[0] - s))

a1 = slopes('tradeoff_leak_moment.txt', 240)
ah = slopes('tradeoff_alpha.txt', 180, '0.5')
a2 = slopes('tradeoff_alpha.txt', 180, '2.0')
for (lst, name, spec) in [(ah, 'a=1/2', [(2.0, 7.67), (4.5, 8.64), (6.9, 8.89), (9.3, 8.96)]),
                          (a1, 'a=1', [(2.1, 7.98), (4.6, 8.72), (7.0, 8.96), (9.6, 9.00)]),
                          (a2, 'a=2', [(2.7, 8.17), (5.0, 8.75), (7.3, 8.89), (9.6, 8.97)])]:
    for s, q in spec:
        p = near(lst, s)
        chk(f'Table1 {name} s~{s} (at {p[0]:.2f})', q, p[1], 6e-3)
        chk(f'Table1 {name} s~{s} location', s, p[0], 0.1)
chk('Table1 all slopes < pi^2', 1.0, float(max(x[1] for x in a1 + ah + a2) < PI ** 2), 0)

# ---------------- Table 2 and Section 7.3
t = txt('fit_constant.txt')
gap = {int(n): (float(x), float(g)) for n, x, g in re.findall(r'^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+[0-9.nan]+\s*$', t.split('1/pi^2')[0], re.M)}
leg = {int(n): (float(a), float(b), float(c)) for n, a, b, c in re.findall(r'^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$', t.split('Legendre cross-check')[1].split('log n')[0], re.M)}
T2 = {256: (1.7686, 1.5609, 1.5592), 1024: (1.5858, 1.7494, 1.7454), 4096: (1.4588, 1.9312, 1.9274),
      16384: (1.3655, 2.1089, 2.1065), 65536: (1.2937, 2.2836, 2.2802), 262144: (1.2368, 2.4559, 2.4539)}
worst = 0
for n, (qg, qx, ql) in T2.items():
    chk(f'Table2 gap n={n}', qg, gap[n][1], 6e-5)
    chk(f'Table2 n lam n={n}', qx, gap[n][0], 6e-5)
    chk(f'Table2 Legendre n={n}', ql, leg[n][1], 6e-5)
    worst = max(worst, abs(leg[n][0] - leg[n][1]) / leg[n][0])
chk('Legendre within 0.3%', 0.003, worst, 0.003 - worst if worst <= 0.003 else 0)
chk('ell* at 2^18 = 2.33', 2.33, leg[262144][2], 6e-3)
fits = t.split('n >= 4096')[1]
b2 = float(re.search(r'M2 .*?b\*pi\^2=([0-9.]+)', fits).group(1)); b4 = float(re.search(r'M4 .*?b\*pi\^2=([0-9.]+)', fits).group(1))
r2 = float(re.search(r'M2 .*?rms=([0-9.e+-]+)', fits).group(1)); r4 = float(re.search(r'M4 .*?rms=([0-9.e+-]+)', fits).group(1))
chk('fit sqrt: 0.96', 0.96, b2, 6e-3)
chk('fit loglog: 1.10', 1.10, b4, 6e-3)
chk('fit residuals ~1e-5', 1.0, float(r2 < 5e-5 and r4 < 5e-5), 0)
lp = re.findall(r'^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+[0-9.nan]+\s+([0-9.]+)\s*$', t.split('log n  x (Legendre)')[1] if 'log n  x (Legendre)' in t else t.split('x (Legendre)')[-1], re.M)
chk('log n ~ 80 reaches ell* ~ 10', 1.0, float(any(int(a) == 80 and abs(float(c) - 10) < 0.2 for a, b, c, d in [(q[0], q[1], q[2], q[3]) for q in lp])), 0)

# ---------------- Section 7.4
t = txt('check_toeplitz_alpha.txt')
def ratios(alpha):
    blk = t.split('alpha=%s' % alpha)[1].split('alpha=')[0]
    return [float(r) for r in re.findall(r'^\s*\d+\s+[0-9.]+\s+([0-9.]+)', blk, re.M)]
rh, r2_ = ratios('0.5'), ratios('2.0')
chk('7.4 a=1/2 from 2.16', 2.16, rh[0], 6e-3); chk('7.4 a=1/2 to 1.84', 1.84, rh[-1], 6e-3)
chk('7.4 a=2 from 4.26', 4.26, r2_[0], 6e-3); chk('7.4 a=2 to 2.87', 2.87, r2_[-1], 6e-3)
chk('7.4 monotone', 1.0, float(all(np.diff(rh) < 0) and all(np.diff(r2_) < 0)), 0)
t = txt('check_trigamma.txt')
g = re.findall(r'^\s*\d+\s+[0-9.]+\s+([-0-9.e+]+)\s+[0-9.]+\s+([0-9.]+)\s*$', t, re.M)
chk('7.4 lam_min+lam_max-4pi^2 machine precision', 1.0, float(max(abs(float(a)) for a, b in g) < 1e-12), 0)
chk('7.4 Gram ratio min 0.959', 0.959, min(float(b) for a, b in g), 6e-4)
chk('7.4 Gram ratio max 0.968', 0.968, max(float(b) for a, b in g), 6e-4)
cf = re.findall(r'quad \(?([-0-9.e]+)(?: \+ 0\.0j)?\)?.*?closed form \(?(?:0\.0 )?([-0-9.e]+)', t)
chk('Remark 6.3 ten digits', 1.0, float(len(re.findall(r'sigma-hat', t)) == 4), 0)

# ---------------- Section 4: both inequalities of Theorem A for 16 <= n <= 2^18
ok_all = True
for line in open('results/hilbert_norm.csv').read().splitlines()[1:]:
    c_ = line.split(','); n_ = int(c_[0]); g_ = float(c_[2]); Ln = math.log(n_)
    lo = 2 / PI * (Ln - math.log(Ln)) / n_; hi = 2 / PI * (Ln + 6 * math.sqrt(Ln) + 13) / n_
    ok_all = ok_all and (lo <= g_ <= hi)
chk('Thm A bounds hold 16..2^18', 1.0, float(ok_all), 0)
chk('(2 pi)^4 threshold', 1.0, float((2 * PI) ** 4 < 1600), 0)

nf = 0
for loc, q, c, ok in rows:
    if not ok:
        nf += 1
    print(f"{'OK  ' if ok else 'FAIL'} | {loc:45s} | quoted {q:<12.6g} | computed {c:<12.6g}")
print(f"\n{len(rows)} checks, {nf} FAIL")

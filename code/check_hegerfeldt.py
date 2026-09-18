"""Numerical check of the sharp Hegerfeldt theorem (Subsection 7.2, Theorem 7.6 and Proposition 7.7).

(a) inequality E_- >= C0 E exp(-pi r omega_bar) on random smooth COMPLEX data in B_r, built as a
    right-moving positive-frequency wave plus a random perturbation of size 10^(-4)..1
    (for real data E_- = E/2 identically and the check would be empty);
(b) the extremal family of the sharpness proposition, u = q - q(.-h), q = transported Gaussian packet,
    for small lambda0 only: in double precision E_- cannot be resolved below about 1e-18, so the
    large-lambda0 behaviour (ratio log(E/E_-)/(pi r omega_bar) -> 1) is checked in
    check_hegerfeldt_sharpness.py with the leak computed exactly from the Mellin formula.
"""
import numpy as np

r0 = max(2*l/(np.exp(2*np.pi*l)+1) for l in np.linspace(0, 1, 200001))
C0 = 0.5*np.exp(-np.pi*r0)

def energies(u, v, x):
    dx = x[1]-x[0]
    N = len(x)
    k = 2*np.pi*np.fft.fftfreq(N, dx)
    U = np.fft.fft(u)*dx; V = np.fft.fft(v)*dx
    dk = 2*np.pi/(N*dx)
    Em = (np.sum(np.abs(U[k < 0])**2) + np.sum(np.abs(V[k > 0])**2))*dk/(2*np.pi)
    E = (np.sum(np.abs(u)**2)+np.sum(np.abs(v)**2))*dx
    Mp = (np.sum(k[k > 0]*np.abs(U[k > 0])**2) + np.sum(-k[k < 0]*np.abs(V[k < 0])**2))*dk/(2*np.pi)
    return Em, E, Mp

def bump(x, a, b):
    y = np.zeros_like(x); m = (x > a) & (x < b)
    s = (x[m]-a)/(b-a); y[m] = np.exp(-1/(s*(1-s)))
    return y

rng = np.random.default_rng(1)
r = 1.0
x = np.linspace(-4, 4, 2**16, endpoint=False)
worst = np.inf
minEm = np.inf
for trial in range(200):
    kk = rng.uniform(1, 40)
    ph0 = bump(x, -r, r)*np.exp(1j*kk*x)*(1+rng.normal()*np.cos(rng.uniform(0, 20)*x))
    ph0p = np.gradient(ph0, x)
    eps = 10.0**rng.uniform(-4, 0)
    ph1 = -ph0p + eps*bump(x, -0.7, 0.9)*(rng.normal()+1j*rng.normal())*np.exp(1j*rng.uniform(-40, 40)*x)
    u = 0.5*(ph1-ph0p); v = 0.5*(ph1+ph0p)
    Em, E, Mp = energies(u, v, x)
    minEm = min(minEm, Em/E)
    ratio = Em/(C0*E*np.exp(-np.pi*r*Mp/E))
    worst = min(worst, ratio)
print(f"C0={C0:.6f}  (a) 200 random complex data: min E_-/E = {minEm:.2e}, min ratio E_-/bound = {worst:.3f}")

# (b) sharpness family, built in the Mellin variable
for lam in [2, 4, 8]:
    d = lam**-0.25
    s = np.linspace(1e-9, 1-1e-9, 400001)
    rho = np.log(s/(1-s)); a = s*(1-s)
    g = (2*np.pi*d*d)**-0.25*np.exp(-rho**2/(4*d*d))
    phi = g*np.exp(-1j*lam*rho)/np.sqrt(a)
    rp = r*(1-1/lam); L = 2*rp
    tbar = (lam/np.pi)*(1+np.exp(d*d/2)); h = L/(2*tbar)
    eta = (2*r-L-h)/2
    x0 = -r+eta
    X = np.linspace(-3, 3, 2**17, endpoint=False)
    w = np.interp(X, x0+L*s, np.conj(phi).real, left=0, right=0) + 1j*np.interp(X, x0+L*s, np.conj(phi).imag, left=0, right=0)
    w /= np.sqrt(L)
    wsh = np.interp(X-h, X, w.real, left=0, right=0)+1j*np.interp(X-h, X, w.imag, left=0, right=0)
    u = w-wsh
    supp = X[np.abs(u) > 1e-12]
    Em, E, Mp = energies(u, np.zeros_like(u), X)
    Om = Mp/E
    print(f"(b) lam={lam:2d} supp=[{supp.min():+.4f},{supp.max():+.4f}]  E_-/E={Em/E:.2e}  "
          f"log(E/E_-)/(pi r omega_bar)={np.log(E/Em)/(np.pi*r*Om):.4f}  thm holds: {Em >= C0*E*np.exp(-np.pi*r*Om)}")

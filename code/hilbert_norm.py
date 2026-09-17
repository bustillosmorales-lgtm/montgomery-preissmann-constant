"""||H_n|| for the truncated discrete Hilbert matrix (H_n)_{jk} = 1/(j-k), j != k.

T_n(xi) = (1/2) I + (i/(2 pi)) H_n, so  lambda_min(T_n(xi)) = (pi - ||H_n||)/(2 pi).
Conjecture (Montgomery, Preissmann): (pi - ||H_n||) n / log n -> 2/pi = 0.63662.

n <= 4096: dense eigvalsh of the Hermitian matrix iH_n (cross-check of Lanczos).
n  > 4096: Lanczos (scipy eigsh) with an O(n log n) FFT matvec.
Output: results/hilbert_norm.csv
"""
import time
import numpy as np
from scipy.linalg import eigvalsh
from scipy.signal import fftconvolve
from scipy.sparse.linalg import LinearOperator, eigsh

PI = np.pi


def dense_norm(n):
    j = np.arange(n)
    d = j[:, None] - j[None, :]
    with np.errstate(divide="ignore"):
        A = np.where(d == 0, 0.0, 1.0 / np.where(d == 0, 1, d))
    return eigvalsh(1j * A, subset_by_index=[n - 1, n - 1])[0]


def lanczos_norm(n, tol=1e-14):
    m = np.arange(-(n - 1), n)
    with np.errstate(divide="ignore"):
        ker = np.where(m == 0, 0.0, 1.0 / np.where(m == 0, 1, m))
    # (H x)_j = sum_k x_k/(j-k) = (x * ker)[j + n - 1] with ker indexed by j-k
    def mv(x):
        x = np.ravel(x)
        return 1j * fftconvolve(x, ker, mode="full")[n - 1:2 * n - 1]
    op = LinearOperator((n, n), matvec=mv, dtype=complex)
    v0 = np.sin(PI * (np.arange(n) + 0.5) / n) * np.exp(-1j * PI / 2 * np.arange(n))
    val = eigsh(op, k=1, which="LA", tol=tol, v0=v0, ncv=60, maxiter=200000, return_eigenvectors=False)
    return val[0]


if __name__ == "__main__":
    rows = []
    print(f"{'n':>7} {'method':>7} {'pi-||H_n||':>14} {'lam_min*n/log n':>16} {'(pi-||H||)n/log n':>18} {'sec':>7}")
    ns = [2 ** k for k in range(4, 17)]
    for n in ns:
        for method in (["dense", "lanczos"] if 256 <= n <= 4096 else (["dense"] if n < 256 else ["lanczos"])):
            t0 = time.time()
            nrm = dense_norm(n) if method == "dense" else lanczos_norm(n)
            gap = PI - nrm
            lam = gap / (2 * PI)
            rows.append((n, method, gap, lam, lam * n / np.log(n), gap * n / np.log(n)))
            print(f"{n:7d} {method:>7} {gap:14.10e} {lam*n/np.log(n):16.6f} {gap*n/np.log(n):18.6f} {time.time()-t0:7.1f}", flush=True)
    with open("results/hilbert_norm.csv", "w") as fh:
        fh.write("n,method,pi_minus_norm,lambda_min,lambda_min_n_over_log_n,gap_n_over_log_n\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]:.15e},{r[3]:.15e},{r[4]:.10f},{r[5]:.10f}\n")

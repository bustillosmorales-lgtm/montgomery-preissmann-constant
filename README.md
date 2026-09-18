# The constant in the Montgomery–Preissmann conjecture: code and data

Python code and computed results accompanying the article

> F. Bustillos, *The constant in the Montgomery–Preissmann conjecture: a sharp one-sided uncertainty principle and Toeplitz matrices with a zero at a jump*.

Let `H_n` be the `n × n` matrix with entries `1/(j-k)` off the diagonal. The article proves that

```
pi - ||H_n||  ~  (2/pi) log(n)/n ,
```

which settles the conjecture of Montgomery and Preissmann and identifies its constant, with the explicit bounds

```
(2/pi) (log(n) - log(log(n)))/n  <=  pi - ||H_n||  <=  (2/pi) log(n)/n + sqrt(2 log(n))/n + 30/(pi n)
```

for `n >= n_0`. It also proves a one-sided uncertainty principle with sharp exponent `pi^2`, with consequences for Toeplitz matrices whose symbol vanishes on one side of a jump and a sharp quantitative form of the Hegerfeldt theorem for the wave equation in `1+1` dimensions.

## Contents

| File | What it computes |
|---|---|
| `code/check_local_model.py` | The Mellin representation, the Koppelman–Pincus formula and its sign, the leak formula, the key identity and the sharp inequality, on sixteen test functions. |
| `code/hilbert_norm.py` | `pi - ||H_n||` for `n = 2^4 … 2^18`, dense up to `n = 4096` and by Lanczos with an FFT matrix–vector product beyond. |
| `code/tradeoff_leak_moment.py` | The leak–moment curve at 320 bits (python-flint) for `alpha = 1`, and in double precision for several dimensions. |
| `code/tradeoff_alpha.py` | The same curve for `alpha = 1/2` and `alpha = 2`. |
| `code/check_localisation.py` | The localisation lemma on thirty spectrally localised test functions. |
| `code/check_transfer.py` | The transfer lemma on the minimising vectors. |
| `code/check_upper.py` | The Gaussian packets: exact first moment, leak bound, and the resulting upper bound with the parameters of the proof (`delta^4 = 2 pi^2 / log(n)`). |
| `code/check_toeplitz_alpha.py` | `lambda_min(T_n(xi^alpha))` against `(alpha log(n)/(pi^2 n))^alpha`. |
| `code/check_trigamma.py` | The trigamma symbol: Fourier coefficients, the reflection identity and the Gram matrix. |
| `code/fit_constant.py` | Fits of `n lambda_n` and the Legendre cross-check against the leak–moment curve. |
| `code/check_hegerfeldt.py` | The sharp Hegerfeldt inequality on random initial data, and the extremal family for small `lambda0`. |
| `code/check_hegerfeldt_sharpness.py` | The extremal family for large `lambda0`, with the leak computed exactly from the Mellin formula. |
| `code/verify_all.py` | Recomputes every numerical value quoted in the article and reports discrepancies. |
| `results/` | The outputs of the scripts above, as produced on the machine described in the article. |

## Reproducing

```
python -m pip install numpy scipy mpmath python-flint pillow
python code/hilbert_norm.py          # writes results/hilbert_norm.csv
python code/verify_all.py            # 120 checks against results/
```

`verify_all.py` prints one line per quoted value, with the value in the article and the value recomputed from `results/`, and ends with the number of failures. On the stored results it reports `120 checks, 0 FAIL`.

Runtimes on a desktop machine with 22 logical cores: a few seconds for the dense diagonalisations, about three minutes for `n = 2^17` and ten for `n = 2^18` in `hilbert_norm.py`, and about three minutes for each high-precision curve on 20 cores.

## Licence

MIT, see `LICENSE`.

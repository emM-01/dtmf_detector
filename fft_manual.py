"""
fft_manual.py — FFT e IFFT manuales (Cooley-Tukey radix-2 recursivo).

Implementa la Transformada Discreta de Fourier desde cero, sin usar
numpy.fft ni ninguna librería de FFT externa.

Algoritmo: decimación en tiempo (DIT), radix-2 recursivo.
Restricción: la longitud de la señal debe ser potencia de 2.

Funciones públicas
------------------
fft(x)         — Transformada Discreta de Fourier
ifft(X)        — Transformada Inversa
rfft(x)        — Parte positiva del espectro (señal real)
rfftfreq(n, d) — Eje de frecuencias para rfft (equivalente a np.fft.rfftfreq)
"""

import numpy as np


# ── Validación interna ──────────────────────────────────────────────────────
def _es_potencia_de_2(n: int) -> bool:
    """Retorna True si n es potencia de 2 (n > 0)."""
    return n > 0 and (n & (n - 1)) == 0


def _siguiente_potencia_de_2(n: int) -> int:
    """Retorna la menor potencia de 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


# ── FFT principal ───────────────────────────────────────────────────────────
def fft(x: np.ndarray) -> np.ndarray:
    """
    FFT recursiva de Cooley-Tukey (decimación en tiempo, radix-2).

    Descompone la DFT de N puntos en dos DFTs de N/2 puntos:
        X[k]         = X_par[k] + W_k · X_impar[k]
        X[k + N/2]   = X_par[k] - W_k · X_impar[k]

    donde W_k = exp(-2πj·k/N) son los factores de giro (twiddle factors).

    Parámetros
    ----------
    x : array_like — señal de entrada; longitud debe ser potencia de 2.

    Retorna
    -------
    np.ndarray complex128 — espectro completo de longitud N.

    Lanza
    -----
    ValueError si len(x) no es potencia de 2.
    """
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)

    if not _es_potencia_de_2(N):
        raise ValueError(
            f"fft: N debe ser potencia de 2, se recibió N={N}. "
            f"Usa zero_pad(x) para ajustar."
        )

    # Caso base — DFT de 1 punto es la señal misma
    if N == 1:
        return x.copy()

    # ── Paso 1: separar en mitades par e impar (bit-reversal implícito) ──
    X_par   = fft(x[0::2])   # índices pares:   x[0], x[2], x[4], ...
    X_impar = fft(x[1::2])   # índices impares: x[1], x[3], x[5], ...

    # ── Paso 2: factores de giro W_k = exp(-2πj·k/N) ──
    k = np.arange(N // 2)
    W = np.exp(-2j * np.pi * k / N)

    # ── Paso 3: combinar (butterfly) ──
    mariposa = W * X_impar
    return np.concatenate([X_par + mariposa,
                           X_par - mariposa])


# ── IFFT ────────────────────────────────────────────────────────────────────
def ifft(X: np.ndarray) -> np.ndarray:
    """
    IFFT mediante la propiedad de conjugación:

        x[n] = (1/N) · conj( FFT( conj(X) ) )

    Reutiliza fft() sin duplicar lógica.

    Parámetros
    ----------
    X : array_like — espectro de entrada; longitud debe ser potencia de 2.

    Retorna
    -------
    np.ndarray complex128 — señal reconstruida de longitud N.
    """
    X = np.asarray(X, dtype=np.complex128)
    N = len(X)
    return np.conj(fft(np.conj(X))) / N


# ── rfft — solo mitad positiva del espectro (señales reales) ────────────────
def rfft(x: np.ndarray) -> np.ndarray:
    """
    Versión de la FFT para señales reales: retorna solo los N//2 + 1
    coeficientes únicos (frecuencias positivas + DC + Nyquist).

    Para una señal real x[n], el espectro es simétrico:
        X[N-k] = conj(X[k])
    por lo tanto la segunda mitad es redundante.

    Parámetros
    ----------
    x : array_like — señal real; longitud debe ser potencia de 2.

    Retorna
    -------
    np.ndarray complex128 de longitud N//2 + 1.
    """
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)
    X = fft(x)
    return X[:N // 2 + 1]


# ── rfftfreq — eje de frecuencias para rfft ─────────────────────────────────
def rfftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """
    Retorna el eje de frecuencias en Hz para una rfft de n puntos.

    Equivalente a np.fft.rfftfreq(n, d).

    Parámetros
    ----------
    n : número de muestras de la señal original.
    d : espaciado entre muestras en segundos (d = 1/sample_rate).

    Retorna
    -------
    np.ndarray float64 de longitud n//2 + 1.
    """
    return np.arange(n // 2 + 1) / (n * d)


# ── zero_pad — relleno para alcanzar potencia de 2 ──────────────────────────
def zero_pad(x: np.ndarray) -> np.ndarray:
    """
    Rellena x con ceros hasta la siguiente potencia de 2.
    Necesario antes de llamar a fft() cuando len(x) no es potencia de 2.

    Parámetros
    ----------
    x : señal de entrada.

    Retorna
    -------
    np.ndarray con len = siguiente potencia de 2 >= len(x).
    """
    N    = len(x)
    M    = _siguiente_potencia_de_2(N)
    xpad = np.zeros(M, dtype=np.complex128)
    xpad[:N] = x
    return xpad


# ── Verificación rápida (python fft_manual.py) ──────────────────────────────
if __name__ == "__main__":
    TOL = 1e-10
    print("=" * 50)
    print("  Verificación de fft_manual.py")
    print("=" * 50)

    resultados = []

    # U-F1: FFT(delta) = [1, 1, ..., 1]
    delta = np.zeros(8, dtype=np.complex128)
    delta[0] = 1.0
    D = fft(delta)
    err = float(np.max(np.abs(D - 1.0)))
    ok = err < TOL
    resultados.append(ok)
    print(f"\nU-F1  FFT(delta) = vector de unos")
    print(f"      error máx = {err:.2e}  →  {'PASS' if ok else 'FAIL'}")

    # U-F2: FFT(constante) — solo bin DC distinto de cero
    const = np.ones(8, dtype=np.complex128)
    C = fft(const)
    err_dc   = abs(C[0] - 8.0)
    err_rest = float(np.max(np.abs(C[1:])))
    ok = err_dc < TOL and err_rest < TOL
    resultados.append(ok)
    print(f"\nU-F2  FFT(constante): bin0={C[0].real:.1f} (esp. 8), resto≈0")
    print(f"      error DC={err_dc:.2e}, error resto={err_rest:.2e}  →  {'PASS' if ok else 'FAIL'}")

    # U-F3: FFT(senoide 1 Hz, fs=8) — bin 1 es el dominante
    t   = np.arange(8) / 8.0
    s8  = np.sin(2 * np.pi * 1.0 * t).astype(np.complex128)
    S   = fft(s8)
    bin_dom = int(np.argmax(np.abs(S)))
    ok = bin_dom == 1
    resultados.append(ok)
    print(f"\nU-F3  FFT(senoide 1Hz, fs=8): bin dominante={bin_dom} (esp. 1)")
    print(f"      →  {'PASS' if ok else 'FAIL'}")

    # U-F4: ifft(fft(x)) ≈ x  (redondeo)
    rng  = np.random.default_rng(0)
    xn   = (rng.uniform(-1, 1, 64) + 0j).astype(np.complex128)
    err  = float(np.max(np.abs(ifft(fft(xn)) - xn)))
    ok   = err < TOL
    resultados.append(ok)
    print(f"\nU-F4  ifft(fft(ruido_64)) ≈ x — error={err:.2e}")
    print(f"      →  {'PASS' if ok else 'FAIL'}")

    # U-F5: rfft coincide con np.fft.rfft
    xr  = rng.uniform(-1, 1, 64)                     # float64
    err = float(np.max(np.abs(rfft(xr) - np.fft.rfft(xr))))
    ok  = err < TOL
    resultados.append(ok)
    print(f"\nU-F5  rfft(x) == np.fft.rfft(x) — error={err:.2e}")
    print(f"      →  {'PASS' if ok else 'FAIL'}")

    # U-F6: rfftfreq coincide con np.fft.rfftfreq
    f1 = rfftfreq(64, d=1/44100)
    f2 = np.fft.rfftfreq(64, d=1/44100)
    err = float(np.max(np.abs(f1 - f2)))
    ok  = err < TOL
    resultados.append(ok)
    print(f"\nU-F6  rfftfreq(64, 1/44100) — error={err:.2e}")
    print(f"      →  {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 50)
    total = len(resultados)
    pasados = sum(resultados)
    print(f"  Resultado: {pasados}/{total} PASS")
    print("=" * 50)
    exit(0 if pasados == total else 1)

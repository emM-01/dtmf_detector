# Detección de Tonos DTMF mediante Transformada de Fourier

Proyecto para el curso **Matemáticas IV — Transformada de Fourier**  
Departamento Académico de Sistemas Computacionales (DASC), UABCS.

Implementa la detección de dígitos DTMF (Dual-Tone Multi-Frequency)
analizando las frecuencias dominantes de una señal de audio mediante la
**Transformada Discreta de Fourier**, implementada desde cero con el
algoritmo de Cooley-Tukey radix-2.

---

## Estructura

```
dtmf_detector/
├── fft_manual.py       # FFT e IFFT manuales (Cooley-Tukey radix-2)
├── dtmf_detector.py    # Detección DTMF: generación, análisis, agrupación
├── app.py              # Aplicación de escritorio (tkinter + matplotlib)
├── test_unit.py        # Pruebas unitarias (44 casos, 5 secciones)
├── requirements.txt    # Dependencias
└── README.md
```

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Uso

```bash
# 1. Verificar que todo funciona correctamente
python test_unit.py

# 2. Ejecutar la aplicación
python app.py
```

---

## Pruebas unitarias

```
python test_unit.py
```

44 casos en 5 secciones:

| Sección | Qué prueba |
|---|---|
| 1 · fft_manual | FFT(δ), linealidad, desplazamiento, ifft(fft(x))≈x |
| 2 · compute_fft | Eje de frecuencias, ventana Hanning, detección de picos |
| 3 · Generación | Duración, frecuencias de cada dígito, ValueError |
| 4 · detect_digit | Los 16 dígitos DTMF, silencio → None, ruido → None |
| 5 · analyze_audio | Secuencias completas incluyendo "8001234567" |

---

## Algoritmo — FFT de Cooley-Tukey (fft_manual.py)

La FFT se implementa como una recursión radix-2 (decimación en tiempo):

```
fft(x):
    N = len(x)
    si N == 1: retornar x

    X_par   = fft(x[0::2])          # mitad de índices pares
    X_impar = fft(x[1::2])          # mitad de índices impares

    W_k = exp(-2πj·k/N),  k = 0 … N/2−1   # factores de giro

    retornar [ X_par + W_k · X_impar,
               X_par − W_k · X_impar ]
```

Complejidad: **O(N log N)** en lugar de O(N²) de la DFT directa.

La IFFT se obtiene sin duplicar código:

```
ifft(X) = conj( fft( conj(X) ) ) / N
```

---

## Cómo funciona la detección DTMF

Cada dígito DTMF es la suma de **dos senoides puras**:

```
s(t) = sin(2π · f_fila · t) + sin(2π · f_col · t)
```

| | 1209 Hz | 1336 Hz | 1477 Hz | 1633 Hz |
|---|---|---|---|---|
| **697 Hz** | 1 | 2 | 3 | A |
| **770 Hz** | 4 | 5 | 6 | B |
| **852 Hz** | 7 | 8 | 9 | C |
| **941 Hz** | * | 0 | # | D |

El proceso de detección por segmento (50 ms):

1. Filtro Butterworth paso-banda 600–1700 Hz
2. Ventana de Hanning → reduce spectral leakage
3. Zero-padding hasta potencia de 2
4. **FFT manual** (Cooley-Tukey)
5. Búsqueda de pico en cada una de las 8 frecuencias DTMF
6. Filtros anti-ruido: RMS mínimo, SNR, dominancia del pico
7. Consulta en tabla → dígito identificado

---

## Restricciones de implementación

| Elemento | Usado |
|---|---|
| FFT | Implementación propia en `fft_manual.py` |
| `numpy.fft` | ❌ No se usa en detección (solo en pruebas como referencia) |
| Arrays | `numpy` (ndarray, operaciones básicas) |
| Filtros | `scipy.signal` (Butterworth paso-banda) |
| Gráficas | `matplotlib` |
| UI | `tkinter` |

---

## Licencia

MIT — libre para uso académico.

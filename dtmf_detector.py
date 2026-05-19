"""
dtmf_detector.py
----------------
Lógica principal para detección de tonos DTMF usando FFT (Transformada de Fourier).

DTMF (Dual-Tone Multi-Frequency) usa dos frecuencias simultáneas:
    Filas:    697 Hz, 770 Hz, 852 Hz, 941 Hz
    Columnas: 1209 Hz, 1336 Hz, 1477 Hz, 1633 Hz
"""

import numpy as np
from scipy.signal import butter, lfilter
from fft_manual import rfft, rfftfreq, zero_pad

# ── Tabla DTMF ─────────────────────────────────────────────────────────────
ROW_FREQS    = [697, 770, 852, 941]          # frecuencias de fila (Hz)
COL_FREQS    = [1209, 1336, 1477, 1633]      # frecuencias de columna (Hz)

DTMF_TABLE = {
    (697,  1209): '1', (697,  1336): '2', (697,  1477): '3', (697,  1633): 'A',
    (770,  1209): '4', (770,  1336): '5', (770,  1477): '6', (770,  1633): 'B',
    (852,  1209): '7', (852,  1336): '8', (852,  1477): '9', (852,  1633): 'C',
    (941,  1209): '*', (941,  1336): '0', (941,  1477): '#', (941,  1633): 'D',
}

TOLERANCE_HZ  = 30     # margen de tolerancia para reconocer una frecuencia (Hz)
MIN_ENERGY    = 0.01   # energía mínima normalizada (no se usa directamente, ver detect_digit)
MIN_RMS       = 0.015  # RMS mínimo de la señal cruda — descarta silencio y estática baja
SNR_THRESHOLD = 6.0    # el pico DTMF debe ser ≥ N veces el nivel medio del espectro
PEAK_RATIO    = 3.5    # el pico de fila/columna debe ser ≥ N veces el segundo mejor candidato


# ── Filtro paso-banda ───────────────────────────────────────────────────────
def bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float,
                    sample_rate: int, order: int = 4) -> np.ndarray:
    """Aplica un filtro Butterworth paso-banda."""
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)


# ── FFT de un segmento ──────────────────────────────────────────────────────
def compute_fft(segment: np.ndarray, sample_rate: int):
    """
    Calcula la FFT de un segmento de audio usando fft_manual (Cooley-Tukey).

    Pasos:
      1. Aplica ventana de Hanning  → reduce spectral leakage
      2. Zero-padding hasta potencia de 2  → requisito del algoritmo radix-2
      3. rfft manual  → solo frecuencias positivas (señal real)
      4. Normaliza por N  → magnitud en unidades de amplitud

    Returns
    -------
    freqs     : np.ndarray  — eje de frecuencias en Hz
    magnitude : np.ndarray  — magnitud del espectro (normalizada)
    """
    n = len(segment)

    # 1. Ventana de Hanning para reducir spectral leakage
    window   = np.hanning(n)
    windowed = segment * window

    # 2. Zero-padding hasta la siguiente potencia de 2
    padded = zero_pad(windowed)
    m      = len(padded)

    # 3. FFT manual — solo mitad positiva del espectro
    fft_vals  = rfft(padded)
    magnitude = np.abs(fft_vals) / m

    # 4. Eje de frecuencias
    freqs = rfftfreq(m, d=1.0 / sample_rate)

    return freqs, magnitude


# ── Frecuencia dominante cerca de un objetivo ───────────────────────────────
def find_peak_near(freqs: np.ndarray, magnitude: np.ndarray,
                   target_hz: float) -> tuple[float, float]:
    """
    Busca el pico de magnitud en una ventana ±TOLERANCE_HZ alrededor de target_hz.

    Returns
    -------
    (peak_freq, peak_magnitude)
    """
    mask = np.abs(freqs - target_hz) <= TOLERANCE_HZ
    if not mask.any():
        return target_hz, 0.0
    local_mag  = magnitude[mask]
    local_freq = freqs[mask]
    idx = np.argmax(local_mag)
    return float(local_freq[idx]), float(local_mag[idx])


# ── Detectar dígito en un segmento ─────────────────────────────────────────
def detect_digit(segment: np.ndarray, sample_rate: int,
                 min_rms: float = MIN_RMS,
                 snr_threshold: float = SNR_THRESHOLD,
                 peak_ratio: float = PEAK_RATIO) -> dict:
    """
    Analiza un segmento de audio y devuelve el dígito DTMF detectado (si existe).

    Filtros anti-ruido aplicados en orden:
      1. RMS mínimo — descarta silencio y estática baja
      2. SNR        — el pico debe sobresalir N veces sobre el nivel medio del espectro
      3. Peak ratio — el mejor candidato debe ser ≥ N veces el segundo mejor (dominancia)
      4. Tweak      — ambas frecuencias (fila Y columna) deben pasar todos los filtros

    Parameters
    ----------
    min_rms       : nivel RMS mínimo de la señal cruda (0.0–1.0)
    snr_threshold : cuántas veces debe superar el pico al nivel medio del espectro
    peak_ratio    : cuántas veces debe superar el mejor candidato al segundo mejor

    Returns
    -------
    dict con claves:
        digit, row_freq, col_freq, row_mag, col_mag, freqs, magnitude, confidence
    """
    empty = {
        'digit': None, 'row_freq': ROW_FREQS[0], 'col_freq': COL_FREQS[0],
        'row_mag': 0.0, 'col_mag': 0.0, 'confidence': 0.0,
    }

    # ── 1. Filtro RMS: descartar señal demasiado débil (estática / silencio) ──
    rms = float(np.sqrt(np.mean(segment ** 2)))
    if rms < min_rms:
        return {**empty, 'freqs': np.array([]), 'magnitude': np.array([])}

    # ── 2. FFT sobre señal filtrada por banda ──
    filtered = bandpass_filter(segment, 600, 1700, sample_rate)
    freqs, magnitude = compute_fft(filtered, sample_rate)

    # Nivel medio del espectro (ruido de fondo estimado)
    noise_floor = float(magnitude.mean()) + 1e-9

    # ── 3. Buscar picos en las 8 frecuencias DTMF ──
    row_results = [find_peak_near(freqs, magnitude, f) for f in ROW_FREQS]
    col_results = [find_peak_near(freqs, magnitude, f) for f in COL_FREQS]

    row_mags = [r[1] for r in row_results]
    col_mags = [c[1] for c in col_results]

    best_row_idx = int(np.argmax(row_mags))
    best_col_idx = int(np.argmax(col_mags))

    row_freq, row_mag = row_results[best_row_idx]
    col_freq, col_mag = col_results[best_col_idx]

    # ── 4. Filtro SNR: el pico debe sobresalir sobre el ruido de fondo ──
    row_snr = row_mag / noise_floor
    col_snr = col_mag / noise_floor
    if row_snr < snr_threshold or col_snr < snr_threshold:
        return {**empty, 'freqs': freqs, 'magnitude': magnitude}

    # ── 5. Filtro de dominancia: el mejor pico debe superar al segundo candidato ──
    sorted_row = sorted(row_mags, reverse=True)
    sorted_col = sorted(col_mags, reverse=True)
    second_row = sorted_row[1] if len(sorted_row) > 1 else 0.0
    second_col = sorted_col[1] if len(sorted_col) > 1 else 0.0

    if second_row > 0 and (row_mag / (second_row + 1e-9)) < peak_ratio:
        return {**empty, 'freqs': freqs, 'magnitude': magnitude}
    if second_col > 0 and (col_mag / (second_col + 1e-9)) < peak_ratio:
        return {**empty, 'freqs': freqs, 'magnitude': magnitude}

    # ── 6. Confirmar dígito en tabla DTMF ──
    key        = (ROW_FREQS[best_row_idx], COL_FREQS[best_col_idx])
    digit      = DTMF_TABLE.get(key)
    max_mag    = magnitude.max() if magnitude.max() > 0 else 1.0
    row_norm   = row_mag / max_mag
    col_norm   = col_mag / max_mag
    confidence = float(np.sqrt(row_norm * col_norm))

    return {
        'digit'     : digit,
        'row_freq'  : ROW_FREQS[best_row_idx],
        'col_freq'  : COL_FREQS[best_col_idx],
        'row_mag'   : row_mag,
        'col_mag'   : col_mag,
        'freqs'     : freqs,
        'magnitude' : magnitude,
        'confidence': confidence,
    }


# ── Análisis completo de un archivo de audio ────────────────────────────────
def analyze_audio(samples: np.ndarray, sample_rate: int,
                  segment_ms: int = 50) -> list[dict]:
    """
    Divide el audio en segmentos y detecta DTMF en cada uno.

    Parameters
    ----------
    samples     : señal de audio (mono, float32 normalizado -1..1)
    sample_rate : tasa de muestreo (Hz)
    segment_ms  : duración de cada segmento en milisegundos

    Returns
    -------
    Lista de resultados por segmento (cada uno con las claves de detect_digit
    más 'time_start' y 'time_end').
    """
    seg_len = int(sample_rate * segment_ms / 1000)
    results = []

    for i in range(0, len(samples) - seg_len, seg_len):
        segment = samples[i:i + seg_len]
        res = detect_digit(segment, sample_rate)
        res['time_start'] = i / sample_rate
        res['time_end']   = (i + seg_len) / sample_rate
        results.append(res)

    return results


def group_digits(results: list[dict]) -> list[dict]:
    """
    Agrupa segmentos consecutivos con el mismo dígito detectado.

    Returns
    -------
    Lista de dict con: digit, time_start, time_end, confidence_avg
    """
    grouped = []
    current = None

    for r in results:
        d = r['digit']
        if d is None:
            current = None
            continue

        if current and current['digit'] == d:
            current['time_end']       = r['time_end']
            current['confidence_avg'] = (current['confidence_avg'] + r['confidence']) / 2
        else:
            current = {
                'digit'          : d,
                'time_start'     : r['time_start'],
                'time_end'       : r['time_end'],
                'confidence_avg' : r['confidence'],
                'row_freq'       : r['row_freq'],
                'col_freq'       : r['col_freq'],
            }
            grouped.append(current)

    return grouped


# ── Generador de tonos DTMF (para pruebas) ──────────────────────────────────
def generate_dtmf_tone(digit: str, duration: float = 0.3,
                       sample_rate: int = 44100, amplitude: float = 0.5) -> np.ndarray:
    """
    Genera una señal DTMF sintética para un dígito dado.

    Returns
    -------
    np.ndarray de float32 con la señal generada.
    """
    # Buscar las frecuencias correspondientes al dígito
    pair = None
    for (r, c), d in DTMF_TABLE.items():
        if d == digit:
            pair = (r, c)
            break
    if pair is None:
        raise ValueError(f"Dígito '{digit}' no reconocido en la tabla DTMF.")

    row_hz, col_hz = pair
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = amplitude * (np.sin(2 * np.pi * row_hz * t) +
                          np.sin(2 * np.pi * col_hz * t))
    return signal.astype(np.float32)


def generate_dtmf_sequence(digits: str, tone_ms: int = 300,
                           silence_ms: int = 100,
                           sample_rate: int = 44100) -> np.ndarray:
    """
    Genera una secuencia de tonos DTMF con silencios entre ellos.
    """
    parts = []
    silence = np.zeros(int(sample_rate * silence_ms / 1000), dtype=np.float32)

    for ch in digits:
        if ch == ' ':
            parts.append(np.zeros(int(sample_rate * 0.3), dtype=np.float32))
        else:
            tone = generate_dtmf_tone(ch, duration=tone_ms / 1000,
                                      sample_rate=sample_rate)
            parts.append(tone)
            parts.append(silence)

    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)

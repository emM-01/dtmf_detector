"""
test_unit.py — Pruebas unitarias del detector DTMF.

Ejecutar con:
    python test_unit.py

Cada caso imprime PASS o FAIL con detalle del error.
El script retorna código 0 si todos pasan, 1 si alguno falla.

Módulos probados
----------------
fft_manual      — FFT/IFFT manuales (Cooley-Tukey)
dtmf_detector   — Generación de señales, detección de dígitos
"""

import numpy as np
import sys

from fft_manual    import fft, ifft, rfft, rfftfreq, zero_pad
from dtmf_detector import (
    compute_fft, detect_digit, generate_dtmf_tone,
    generate_dtmf_sequence, analyze_audio, group_digits,
    ROW_FREQS, COL_FREQS, DTMF_TABLE,
)

TOL    = 1e-10   # tolerancia para pruebas de FFT
TOL_HZ = 30      # tolerancia en Hz para detección de frecuencias

resultados: list[tuple[str, bool, str]] = []   # (id, ok, detalle)


def caso(id_: str, ok: bool, detalle: str = "") -> None:
    resultados.append((id_, ok, detalle))
    estado = "PASS" if ok else "FAIL"
    print(f"  {estado}  {id_}  {detalle}")


def seccion(titulo: str) -> None:
    print(f"\n{'─'*52}")
    print(f"  {titulo}")
    print(f"{'─'*52}")


# ══════════════════════════════════════════════════════
#  SECCIÓN 1 — fft_manual: corrección matemática
# ══════════════════════════════════════════════════════
seccion("1 · fft_manual — corrección matemática")

# U-F1: FFT(delta) = [1, 1, ..., 1]
delta = np.zeros(8, dtype=np.complex128); delta[0] = 1.0
err = float(np.max(np.abs(fft(delta) - 1.0)))
caso("U-F1", err < TOL, f"FFT(δ)=[1,…,1]  error={err:.2e}")

# U-F2: FFT(constante) → solo bin DC = N
const = np.ones(8, dtype=np.complex128)
C     = fft(const)
e_dc  = abs(C[0] - 8.0)
e_rt  = float(np.max(np.abs(C[1:])))
caso("U-F2", e_dc < TOL and e_rt < TOL,
     f"FFT(cte): DC={C[0].real:.1f}(esp 8) resto_max={e_rt:.2e}")

# U-F3: FFT(senoide 1 Hz, fs=8) → bin dominante = 1
t_  = np.arange(8) / 8.0
s8  = np.sin(2*np.pi*1.0*t_).astype(np.complex128)
bd  = int(np.argmax(np.abs(fft(s8))))
caso("U-F3", bd == 1, f"bin_dominante={bd} (esp 1)")

# U-F4: ifft(fft(x)) ≈ x
rng = np.random.default_rng(42)
xn  = rng.uniform(-1, 1, 64).astype(np.complex128)
err = float(np.max(np.abs(ifft(fft(xn)) - xn)))
caso("U-F4", err < TOL, f"ifft(fft(x))≈x  error={err:.2e}")

# U-F5: rfft coincide con np.fft.rfft (referencia)
xr  = rng.uniform(-1, 1, 64)
err = float(np.max(np.abs(rfft(xr) - np.fft.rfft(xr))))
caso("U-F5", err < TOL, f"rfft vs np.fft.rfft  error={err:.2e}")

# U-F6: rfftfreq coincide con np.fft.rfftfreq
f1  = rfftfreq(128, d=1/44100)
f2  = np.fft.rfftfreq(128, d=1/44100)
err = float(np.max(np.abs(f1 - f2)))
caso("U-F6", err < TOL, f"rfftfreq  error={err:.2e}")

# U-F7: zero_pad produce potencia de 2 y preserva datos originales
xp  = zero_pad(np.ones(100, dtype=np.float64))
n_ok = (len(xp) == 128)
d_ok = float(np.max(np.abs(xp[:100] - 1.0))) < TOL
caso("U-F7", n_ok and d_ok, f"len={len(xp)}(esp 128) datos_ok={d_ok}")

# U-F8: linealidad — FFT(a·x + b·y) = a·FFT(x) + b·FFT(y)
x = rng.uniform(-1, 1, 32).astype(np.complex128)
y = rng.uniform(-1, 1, 32).astype(np.complex128)
a, b = 3.0 + 1j, -2.0 + 0.5j
err  = float(np.max(np.abs(fft(a*x + b*y) - (a*fft(x) + b*fft(y)))))
caso("U-F8", err < TOL, f"linealidad  error={err:.2e}")

# U-F9: desplazamiento — si y[n] = x[n-k] → Y[m] = X[m]·exp(-2πjkm/N)
k_rot = 3
xroll = np.roll(x, k_rot)
N     = len(x)
n_idx = np.arange(N)
Xrot_esperado = fft(x) * np.exp(-2j * np.pi * k_rot * n_idx / N)
err   = float(np.max(np.abs(fft(xroll) - Xrot_esperado)))
caso("U-F9", err < TOL, f"desplazamiento  error={err:.2e}")


# ══════════════════════════════════════════════════════
#  SECCIÓN 2 — compute_fft: ventana y eje de frecuencias
# ══════════════════════════════════════════════════════
seccion("2 · compute_fft — ventana Hanning y eje de frecuencias")

SR = 44100

# U-C1: el eje de frecuencias tiene el tamaño correcto
seg = np.sin(2*np.pi*697*np.linspace(0, 0.05, int(SR*0.05))).astype(np.float32)
freqs, mag = compute_fft(seg, SR)
caso("U-C1", len(freqs) == len(mag) and freqs[0] == 0.0,
     f"len(freqs)={len(freqs)} len(mag)={len(mag)} freqs[0]={freqs[0]:.1f}")

# U-C2: pico detectado en ±30 Hz de la frecuencia de prueba (697 Hz)
pico_hz = float(freqs[np.argmax(mag)])
caso("U-C2", abs(pico_hz - 697) <= TOL_HZ,
     f"pico={pico_hz:.1f} Hz (esp ≈697 Hz, tol={TOL_HZ})")

# U-C3: señal de silencio → magnitud ≈ 0
silencio = np.zeros(2048, dtype=np.float32)
_, mag_sil = compute_fft(silencio, SR)
caso("U-C3", float(mag_sil.max()) < 1e-10,
     f"max_mag_silencio={mag_sil.max():.2e}")

# U-C4: dos tonos simultáneos → dos picos dominantes en las frecuencias correctas
t2  = np.linspace(0, 0.1, int(SR*0.1), endpoint=False)
s2  = (np.sin(2*np.pi*770*t2) + np.sin(2*np.pi*1336*t2)).astype(np.float32)
f2, m2 = compute_fft(s2, SR)
# Encontrar los dos picos más altos
idx_sorted = np.argsort(m2)[::-1]
top2_hz    = sorted([float(f2[idx_sorted[0]]), float(f2[idx_sorted[1]])])
ok_770  = abs(top2_hz[0] - 770)  <= TOL_HZ
ok_1336 = abs(top2_hz[1] - 1336) <= TOL_HZ
caso("U-C4", ok_770 and ok_1336,
     f"picos={top2_hz[0]:.0f}Hz,{top2_hz[1]:.0f}Hz (esp 770,1336)")


# ══════════════════════════════════════════════════════
#  SECCIÓN 3 — Generación de tonos DTMF
# ══════════════════════════════════════════════════════
seccion("3 · Generación de tonos DTMF")

# U-G1: generate_dtmf_tone produce la duración correcta
tone = generate_dtmf_tone('5', duration=0.3, sample_rate=SR)
esp  = int(SR * 0.3)
caso("U-G1", len(tone) == esp, f"len={len(tone)} (esp {esp})")

# U-G2: el tono de cada dígito tiene sus dos frecuencias como picos dominantes
for digit, (f_row, f_col) in [
    ('1', (697, 1209)), ('5', (770, 1336)),
    ('9', (852, 1477)), ('0', (941, 1336)),
]:
    t_sig = generate_dtmf_tone(digit, duration=0.3, sample_rate=SR)
    fd, md = compute_fft(t_sig, SR)
    idx_s  = np.argsort(md)[::-1]
    top2   = sorted([float(fd[idx_s[0]]), float(fd[idx_s[1]])])
    ok_r   = abs(top2[0] - f_row) <= TOL_HZ
    ok_c   = abs(top2[1] - f_col) <= TOL_HZ
    caso(f"U-G2-{digit}", ok_r and ok_c,
         f"'{digit}': picos={top2[0]:.0f},{top2[1]:.0f} Hz (esp {f_row},{f_col})")

# U-G3: generate_dtmf_sequence produce la longitud correcta
seq   = generate_dtmf_sequence("12", tone_ms=300, silence_ms=100, sample_rate=SR)
esp   = 2 * int(SR*0.3) + 2 * int(SR*0.1)
caso("U-G3", len(seq) == esp, f"len={len(seq)} (esp {esp})")

# U-G4: dígito inválido lanza ValueError
try:
    generate_dtmf_tone('Z')
    caso("U-G4", False, "no lanzó ValueError")
except ValueError:
    caso("U-G4", True, "ValueError lanzado correctamente")


# ══════════════════════════════════════════════════════
#  SECCIÓN 4 — detect_digit
# ══════════════════════════════════════════════════════
seccion("4 · detect_digit — identificación de dígitos")

TODOS = list(DTMF_TABLE.values())

# U-D1: cada dígito de la tabla se detecta correctamente
for digit in TODOS:
    tone = generate_dtmf_tone(digit, duration=0.3, sample_rate=SR)
    res  = detect_digit(tone, SR)
    caso(f"U-D1-{digit}", res['digit'] == digit,
         f"detectado='{res['digit']}' (esp '{digit}') conf={res['confidence']:.0%}")

# U-D2: silencio → None
sil = np.zeros(int(SR*0.05), dtype=np.float32)
res = detect_digit(sil, SR)
caso("U-D2", res['digit'] is None, f"silencio→'{res['digit']}' (esp None)")

# U-D3: ruido blanco → None (con umbrales por defecto)
rng2  = np.random.default_rng(99)
ruido = rng2.uniform(-0.005, 0.005, int(SR*0.05)).astype(np.float32)
res   = detect_digit(ruido, SR)
caso("U-D3", res['digit'] is None,
     f"ruido_bajo→'{res['digit']}' (esp None, RMS={float(np.sqrt(np.mean(ruido**2))):.4f})")

# U-D4: confianza entre 0 y 1 para un tono válido
tone5  = generate_dtmf_tone('5', duration=0.3, sample_rate=SR)
res5   = detect_digit(tone5, SR)
caso("U-D4", 0.0 <= res5['confidence'] <= 1.0,
     f"confianza={res5['confidence']:.3f}")


# ══════════════════════════════════════════════════════
#  SECCIÓN 5 — analyze_audio y group_digits
# ══════════════════════════════════════════════════════
seccion("5 · analyze_audio y group_digits — secuencias completas")

# U-A1: secuencia "1234" se detecta completa y en orden
seq_ref = "1234"
audio   = generate_dtmf_sequence(seq_ref, tone_ms=300, silence_ms=150)
results = analyze_audio(audio, SR, segment_ms=50)
grouped = group_digits(results)
detectado = ''.join(g['digit'] for g in grouped)
caso("U-A1", detectado == seq_ref,
     f"detectado='{detectado}' (esp '{seq_ref}')")

# U-A2: secuencia "*#0ABCD" (dígitos extendidos)
seq_ref2 = "*#0ABCD"
audio2   = generate_dtmf_sequence(seq_ref2, tone_ms=300, silence_ms=150)
grouped2 = group_digits(analyze_audio(audio2, SR, segment_ms=50))
detectado2 = ''.join(g['digit'] for g in grouped2)
caso("U-A2", detectado2 == seq_ref2,
     f"detectado='{detectado2}' (esp '{seq_ref2}')")

# U-A3: número telefónico "8001234567"
seq_ref3 = "8001234567"
audio3   = generate_dtmf_sequence(seq_ref3, tone_ms=400, silence_ms=200)
grouped3 = group_digits(analyze_audio(audio3, SR, segment_ms=50))
detectado3 = ''.join(g['digit'] for g in grouped3)
caso("U-A3", detectado3 == seq_ref3,
     f"detectado='{detectado3}' (esp '{seq_ref3}')")

# U-A4: group_digits agrupa correctamente los segmentos consecutivos
# Generamos un solo tono largo y verificamos que agrupa en 1 entrada
audio4  = generate_dtmf_sequence("7", tone_ms=500, silence_ms=0)
grouped4 = group_digits(analyze_audio(audio4, SR, segment_ms=50))
caso("U-A4", len(grouped4) == 1 and grouped4[0]['digit'] == '7',
     f"grupos={len(grouped4)} dígito='{grouped4[0]['digit'] if grouped4 else None}'")

# U-A5: audio de silencio → lista vacía
audio5  = np.zeros(int(SR*1.0), dtype=np.float32)
grouped5 = group_digits(analyze_audio(audio5, SR, segment_ms=50))
caso("U-A5", len(grouped5) == 0, f"grupos={len(grouped5)} (esp 0)")


# ══════════════════════════════════════════════════════
#  RESUMEN FINAL
# ══════════════════════════════════════════════════════
total   = len(resultados)
pasados = sum(ok for _, ok, _ in resultados)
fallidos = [(id_, det) for id_, ok, det in resultados if not ok]

print(f"\n{'═'*52}")
print(f"  Resultado: {pasados}/{total} PASS")
if fallidos:
    print(f"\n  Casos fallidos:")
    for id_, det in fallidos:
        print(f"    ✗  {id_}  {det}")
print(f"{'═'*52}")

sys.exit(0 if pasados == total else 1)

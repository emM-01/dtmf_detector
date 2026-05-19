import time
import numpy as np
import matplotlib.pyplot as plt

def exportar_grafica_dft_fft():
    # 1. Configuración de parámetros del proyecto
    SR = 8000       # Tasa de muestreo (8 kHz)
    N = 512         # Tamaño de ventana
    t = np.arange(N) / SR
    
    # Generar tono DTMF del dígito '5' (770 Hz + 1336 Hz)
    f1, f2 = 770, 1336
    x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    
    # 2. Implementación de la DFT directa O(N^2)
    def dft_directa(signal):
        M = len(signal)
        X_dft = np.zeros(M, dtype=complex)
        for k in range(M):
            suma = 0.0
            for n in range(M):
                suma += signal[n] * np.exp(-2j * np.pi * k * n / M)
            X_dft[k] = suma
        return X_dft

    # 3. Implementación de nuestra FFT manual (Radix-2 Cooley-Tukey)
    def fft_manual(signal):
        signal = np.asarray(signal, dtype=np.complex128)
        M = len(signal)
        if M == 1:
            return signal.copy()
        X_par = fft_manual(signal[0::2])
        X_impar = fft_manual(signal[1::2])
        k = np.arange(M // 2)
        W = np.exp(-2j * np.pi * k / M)
        mariposa = W * X_impar
        return np.concatenate([X_par + mariposa, X_par - mariposa])

    # 4. Medición de tiempos de ejecución
    t0 = time.perf_counter()
    X_dft = dft_directa(x)
    t_dft = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    X_fft = fft_manual(x)
    t_fft = (time.perf_counter() - t1) * 1000

    # Extraer la mitad del espectro para señales reales
    K = N // 2 + 1
    freqs = np.arange(K) * SR / N
    mag_dft = np.abs(X_dft[:K])
    mag_fft = np.abs(X_fft[:K])
    error_max = np.max(np.abs(mag_dft - mag_fft))

    # 5. Renderizado de la gráfica con el estilo del proyecto
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"Validación de Algoritmos: Dígito '5' ({f1} Hz + {f2} Hz) | Error Máx: {error_max:.2e}", 
                 color='#CCCCEE', fontsize=12, fontweight='bold')
    fig.patch.set_facecolor('#16162a')
    
    # Subplot 1: Espectro de Magnitud
    ax1.set_facecolor('#202030')
    ax1.plot(freqs, mag_dft, label='DFT Directa $O(N^2)$', color='#ff5555', linewidth=2.5, alpha=0.7)
    ax1.plot(freqs, mag_fft, label='FFT Radix-2 $O(N\\log N)$', color='#55ff55', linestyle='--', linewidth=1.5)
    ax1.axvline(x=f1, color='#ffaa00', linestyle=':', alpha=0.8, label=f'Fila ({f1} Hz)')
    ax1.axvline(x=f2, color='#00aaff', linestyle=':', alpha=0.8, label=f'Columna ({f2} Hz)')
    
    ax1.set_title("Espectro de Magnitud", color='#CCCCEE', fontsize=10)
    ax1.set_xlabel("Frecuencia (Hz)", color='#8888aa', fontsize=9)
    ax1.set_ylabel("Magnitud", color='#8888aa', fontsize=9)
    ax1.set_xlim(500, 1500)
    ax1.grid(True, color='#2a2a45', linestyle='-')
    ax1.tick_params(colors='#8888aa', labelsize=8)
    ax1.legend(facecolor='#16162a', edgecolor='#383858', labelcolor='#CCCCEE', fontsize=8)

    # Subplot 2: Tiempos de Cómputo
    ax2.set_facecolor('#202030')
    algoritmos = ['DFT Directa', 'FFT Radix-2']
    tiempos = [t_dft, t_fft]
    
    barras = ax2.bar(algoritmos, tiempos, color=['#ff5555', '#55ff55'], width=0.5, edgecolor='#383858')
    ax2.set_title("Tiempo de Cómputo ($N=512$)", color='#CCCCEE', fontsize=10)
    ax2.set_ylabel("Tiempo (ms)", color='#8888aa', fontsize=9)
    ax2.grid(True, color='#2a2a45', axis='y', linestyle='-')
    ax2.tick_params(colors='#8888aa', labelsize=8)
    
    for bar in barras:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (max(tiempos)*0.02), 
                 f"{yval:.2f} ms", ha='center', va='bottom', color='#CCCCEE', fontsize=8, fontweight='bold')

    plt.tight_layout()
    
    # GUARDAR DIRECTAMENTE LA IMAGEN
    nombre_archivo = "dft_vs_fft.png"
    plt.savefig(nombre_archivo, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"¡Gráfica guardada exitosamente como '{nombre_archivo}'!")

# Ejecutar la generación
exportar_grafica_dft_fft()
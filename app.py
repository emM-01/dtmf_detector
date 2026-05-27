"""
app.py  —  Detector de Tonos DTMF  (rediseño profesional)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, collections, time
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import matplotlib.gridspec as mgridspec

from dtmf_detector import (
    analyze_audio, group_digits, compute_fft, detect_digit,
    generate_dtmf_sequence, ROW_FREQS, COL_FREQS, DTMF_TABLE,
    MIN_RMS, SNR_THRESHOLD, PEAK_RATIO,
)

# ╔══════════════════════════════════════════════════════╗
#  PALETA
# ╚══════════════════════════════════════════════════════╝
BG        = "#0a0e17"        # fondo principal
SIDEBAR   = "#0f1520"        # sidebar oscura
CARD      = "#131b2e"        # tarjetas / paneles
CARD2     = "#1a2540"        # tarjetas secundarias
BORDER    = "#1e2d4a"        # bordes sutiles
ACCENT    = "#4d9fff"        # azul principal
ACCENT2   = "#7c5cbf"        # violeta
GREEN     = "#22c55e"        # éxito / activo
YELLOW    = "#eab308"        # advertencia
RED       = "#ef4444"        # error / stop
ORANGE    = "#f97316"        # micrófono
TEAL      = "#06b6d4"        # espectrograma
TEXT      = "#e2e8f0"        # texto principal
TEXT_DIM  = "#64748b"        # texto secundario
TEXT_MID  = "#94a3b8"        # texto medio

DIGIT_COLORS = {
    '0':'#4d9fff','1':'#22c55e','2':'#eab308','3':'#ef4444',
    '4':'#a855f7','5':'#f97316','6':'#06b6d4','7':'#ec4899',
    '8':'#84cc16','9':'#8b5cf6','*':'#fb923c','#':'#34d399',
    'A':'#f472b6','B':'#67e8f9','C':'#fde68a','D':'#c4b5fd',
}

plt.rcParams.update({
    'figure.facecolor': BG,    'axes.facecolor' : CARD,
    'axes.edgecolor'  : BORDER,'axes.labelcolor': TEXT_MID,
    'xtick.color'     : TEXT_DIM,'ytick.color'  : TEXT_DIM,
    'text.color'      : TEXT,  'grid.color'     : BORDER,
    'grid.linestyle'  : '-',   'grid.alpha'     : 0.4,
    'legend.facecolor': CARD,  'legend.edgecolor': BORDER,
    'font.family'     : 'sans-serif',
})

MIC_SR        = 44100
MIC_CHUNK_MS  = 60
MIC_HISTORY_S = 4


# ╔══════════════════════════════════════════════════════╗
#  HELPER — tarjeta métrica (tkinter puro)
# ╚══════════════════════════════════════════════════════╝
def MetricCard(parent, title, init_val, unit="", color=ACCENT, width=130):
    f = tk.Frame(parent, bg=CARD, padx=12, pady=8,
                 highlightbackground=BORDER, highlightthickness=1)
    tk.Label(f, text=title, font=("Segoe UI", 8), bg=CARD, fg=TEXT_DIM).pack(anchor="w")
    row = tk.Frame(f, bg=CARD)
    row.pack(anchor="w")
    val_lbl = tk.Label(row, text=init_val, font=("Segoe UI", 18, "bold"),
                       bg=CARD, fg=color)
    val_lbl.pack(side=tk.LEFT)
    if unit:
        tk.Label(row, text=f" {unit}", font=("Segoe UI", 9), bg=CARD,
                 fg=TEXT_DIM).pack(side=tk.LEFT, anchor="s", pady=(0, 3))
    return f, val_lbl


# ╔══════════════════════════════════════════════════════╗
#  APLICACIÓN PRINCIPAL
# ╚══════════════════════════════════════════════════════╝
class DTMFApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("DTMF Analyzer  •  Detector de Tonos Telefónicos")
        self.configure(bg=BG)
        self.geometry("1380x820")
        self.minsize(1100, 700)

        self.samples     : np.ndarray | None = None
        self.sample_rate : int = MIC_SR
        self.results     : list[dict] = []
        self.grouped     : list[dict] = []
        self._play_thread = None

        # Micrófono
        self._mic_active     = False
        self._mic_stream     = None
        self._mic_buffer     = collections.deque(maxlen=int(MIC_SR * MIC_HISTORY_S))
        self._mic_chunk      = collections.deque(maxlen=int(MIC_SR * MIC_CHUNK_MS / 1000))
        self._mic_digits     = []
        self._last_digit     = None
        self._last_digit_t   = 0.0
        self._digit_cooldown = 0.4
        self._current_view   = "file"   # "file" | "mic"

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════
    #  ESTRUCTURA PRINCIPAL
    # ══════════════════════════════════════════════════════
    def _build_ui(self):
        # ── Root grid: sidebar | contenido
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_content_area()
        self._switch_view("file")   # ahora frame_mic ya existe

    # ── SIDEBAR ──────────────────────────────────────────
    def _build_sidebar(self):
        sb = tk.Frame(self, bg=SIDEBAR, width=200)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.pack_propagate(False)

        # Logo
        logo = tk.Frame(sb, bg=SIDEBAR, pady=20)
        logo.pack(fill=tk.X)
        tk.Label(logo, text="DTMF", font=("Segoe UI", 22, "bold"),
                 bg=SIDEBAR, fg=ACCENT).pack()
        tk.Label(logo, text="Detector de Tonos", font=("Segoe UI", 10),
                 bg=SIDEBAR, fg=TEXT_DIM).pack()

        # Separador
        tk.Frame(sb, bg=BORDER, height=1).pack(fill=tk.X, padx=16)

        # ── Botones de navegación ──
        nav = tk.Frame(sb, bg=SIDEBAR, pady=12)
        nav.pack(fill=tk.X)

        self._nav_btns = {}
        nav_items = [
            ("file",   "📄", "Análisis de Archivo"),
            ("mic",    "🎙", "Micrófono en Vivo"),
            ("teoria", "📐", "Teoría de Fourier"),
        ]
        for key, icon, label in nav_items:
            btn_f = tk.Frame(nav, bg=SIDEBAR, cursor="hand2")
            btn_f.pack(fill=tk.X, padx=8, pady=2)
            tk.Label(btn_f, text=f"  {icon}  {label}",
                     font=("Segoe UI", 10), bg=SIDEBAR, fg=TEXT_MID,
                     anchor="w", padx=8, pady=10).pack(fill=tk.X)
            btn_f.bind("<Button-1>", lambda e, k=key: self._switch_view(k))
            for w in btn_f.winfo_children():
                w.bind("<Button-1>", lambda e, k=key: self._switch_view(k))
            self._nav_btns[key] = btn_f

        # ── Sección de acciones ──
        tk.Frame(sb, bg=BORDER, height=1).pack(fill=tk.X, padx=16, pady=8)
        tk.Label(sb, text="ACCIONES", font=("Segoe UI", 7, "bold"),
                 bg=SIDEBAR, fg=TEXT_DIM).pack(anchor="w", padx=16, pady=(0,6))

        def _sbtn(text, cmd, color=TEXT):
            f = tk.Frame(sb, bg=SIDEBAR, cursor="hand2")
            f.pack(fill=tk.X, padx=8, pady=2)
            lbl = tk.Label(f, text=text, font=("Segoe UI", 9),
                           bg=SIDEBAR, fg=color, anchor="w",
                           padx=12, pady=7)
            lbl.pack(fill=tk.X)
            for w in (f, lbl):
                w.bind("<Button-1>", lambda e, c=cmd: c())
                w.bind("<Enter>",    lambda e, w=f: w.config(bg=CARD2))
                w.bind("<Leave>",    lambda e, w=f: w.config(bg=SIDEBAR))

        _sbtn("📂  Cargar WAV",      self._load_wav)
        _sbtn("🎹  Generar tonos",   self._open_generate_dialog)
        _sbtn("▶  Reproducir",       self._play_audio, GREEN)
        _sbtn("⏹  Detener audio",    self._stop_audio, RED)

        tk.Frame(sb, bg=BORDER, height=1).pack(fill=tk.X, padx=16, pady=8)

        # Botón de micrófono grande
        self.mic_btn_frame = tk.Frame(sb, bg="#1a1f2e", padx=12, pady=10,
                                      cursor="hand2")
        self.mic_btn_frame.pack(fill=tk.X, padx=12, pady=4)
        self.mic_btn_lbl = tk.Label(self.mic_btn_frame,
                                    text="🎙  Activar Micrófono",
                                    font=("Segoe UI", 10, "bold"),
                                    bg="#1a1f2e", fg=ORANGE)
        self.mic_btn_lbl.pack()
        for w in (self.mic_btn_frame, self.mic_btn_lbl):
            w.bind("<Button-1>", lambda e: self._toggle_mic())

        # Status al fondo
        sb.pack_propagate(False)
        status_frame = tk.Frame(sb, bg=SIDEBAR)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=12)
        tk.Frame(status_frame, bg=BORDER, height=1).pack(fill=tk.X, pady=(0,8))
        self.status_var = tk.StringVar(value="Listo")
        tk.Label(status_frame, textvariable=self.status_var,
                 font=("Segoe UI", 8), bg=SIDEBAR, fg=TEXT_DIM,
                 wraplength=170, justify="left").pack(anchor="w")

        # _switch_view se llama después de _build_content_area

    def _switch_view(self, key):
        self._current_view = key
        for k, f in self._nav_btns.items():
            is_sel = (k == key)
            f.config(bg=CARD2 if is_sel else SIDEBAR)
            for w in f.winfo_children():
                w.config(bg=CARD2 if is_sel else SIDEBAR,
                         fg=ACCENT if is_sel else TEXT_MID)
        frames = {"file": self.frame_file, "mic": self.frame_mic,
                  "teoria": self.frame_teoria}
        for k, fr in frames.items():
            if k == key:
                fr.grid()
            else:
                fr.grid_remove()

    # ── ÁREA DE CONTENIDO ────────────────────────────────
    def _build_content_area(self):
        content = tk.Frame(self, bg=BG)
        content.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        # ── Vista Archivo ──
        self.frame_file = tk.Frame(content, bg=BG)
        self.frame_file.grid(row=0, column=0, sticky="nsew")
        self.frame_file.columnconfigure(0, weight=1)
        self.frame_file.rowconfigure(1, weight=1)
        self._build_file_view(self.frame_file)

        # ── Vista Micrófono ──
        self.frame_mic = tk.Frame(content, bg=BG)
        self.frame_mic.grid(row=0, column=0, sticky="nsew")
        self.frame_mic.columnconfigure(0, weight=1)
        self.frame_mic.rowconfigure(1, weight=1)
        self._build_mic_view(self.frame_mic)

        # ── Vista Teoría ──
        self.frame_teoria = tk.Frame(content, bg=BG)
        self.frame_teoria.grid(row=0, column=0, sticky="nsew")
        self.frame_teoria.columnconfigure(0, weight=1)
        self.frame_teoria.rowconfigure(1, weight=1)
        self._build_teoria_view(self.frame_teoria)

    # ══════════════════════════════════════════════════════
    #  VISTA ARCHIVO
    # ══════════════════════════════════════════════════════
    def _build_file_view(self, parent):
        # ── Header con métricas ──
        hdr = tk.Frame(parent, bg=BG, pady=12, padx=16)
        hdr.grid(row=0, column=0, sticky="ew")

        tk.Label(hdr, text="Análisis de Archivo",
                 font=("Segoe UI", 14, "bold"), bg=BG, fg=TEXT).pack(side=tk.LEFT)

        cards_f = tk.Frame(hdr, bg=BG)
        cards_f.pack(side=tk.RIGHT)

        c1, self.mc_digits   = MetricCard(cards_f, "Dígitos",    "—",   color=GREEN)
        c2, self.mc_dur      = MetricCard(cards_f, "Duración",   "—",   "s", ACCENT)
        c3, self.mc_sr       = MetricCard(cards_f, "Muestreo",   "—",   "Hz", TEAL)
        c4, self.mc_conf     = MetricCard(cards_f, "Confianza",  "—",   "%", YELLOW)
        for c in (c1, c2, c3, c4):
            c.pack(side=tk.LEFT, padx=4)

        # ── Cuerpo principal ──
        body = tk.Frame(parent, bg=BG)
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Panel izquierdo — gráficas
        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self._build_file_plots(left)

        # Panel derecho — resultados
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        self._build_results_panel(right)

    def _build_file_plots(self, parent):
        self.fig = plt.Figure(tight_layout=dict(pad=1.2))
        gs = mgridspec.GridSpec(3, 2, figure=self.fig,
                                hspace=0.55, wspace=0.30,
                                height_ratios=[1.1, 1, 1.3])

        self.ax_wave    = self.fig.add_subplot(gs[0, :])
        self.ax_fft     = self.fig.add_subplot(gs[1, :])
        self.ax_spec    = self.fig.add_subplot(gs[2, :])

        self._style_ax(self.ax_wave,    "Forma de Onda",                    "Tiempo (s)",      "Amplitud")
        self._style_ax(self.ax_fft,     "Espectro FFT",                     "Frecuencia (Hz)", "Magnitud")
        self._style_ax(self.ax_spec,    "Espectrograma — Tiempo × Frecuencia × Intensidad",
                                                                            "Tiempo (s)",      "Frecuencia (Hz)")

        for ax in (self.ax_wave, self.ax_fft, self.ax_spec):
            ax.text(0.5, 0.5, "Sin señal cargada",
                    transform=ax.transAxes, ha='center', va='center',
                    color=TEXT_DIM, fontsize=10)

        canvas = FigureCanvasTkAgg(self.fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas

    def _style_ax(self, ax, title, xlabel, ylabel):
        ax.set_title(title, color=TEXT_MID, fontsize=8.5,
                     fontweight='bold', pad=6, loc='left')
        ax.set_xlabel(xlabel, fontsize=7.5, color=TEXT_DIM)
        ax.set_ylabel(ylabel, fontsize=7.5, color=TEXT_DIM)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    def _build_results_panel(self, parent):
        # ── Secuencia grande ──
        seq_card = tk.Frame(parent, bg=CARD, padx=14, pady=14,
                            highlightbackground=BORDER, highlightthickness=1)
        seq_card.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        tk.Label(seq_card, text="SECUENCIA DETECTADA",
                 font=("Segoe UI", 7, "bold"), bg=CARD, fg=TEXT_DIM).pack(anchor="w")
        self.lbl_digits = tk.Label(seq_card, text="—",
                                   font=("Consolas", 26, "bold"),
                                   bg=CARD, fg=GREEN,
                                   wraplength=260, justify="left")
        self.lbl_digits.pack(anchor="w", pady=(4, 0))

        # ── Teclado DTMF visual ──
        kb_card = tk.Frame(parent, bg=CARD, padx=14, pady=10,
                           highlightbackground=BORDER, highlightthickness=1)
        kb_card.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        parent.rowconfigure(1, weight=1)

        tk.Label(kb_card, text="TECLADO DTMF",
                 font=("Segoe UI", 7, "bold"), bg=CARD, fg=TEXT_DIM).pack(anchor="w", pady=(0,6))

        self._dtmf_key_labels = {}
        kb_rows = [['1','2','3','A'], ['4','5','6','B'],
                   ['7','8','9','C'], ['*','0','#','D']]
        col_headers = ['1209 Hz','1336 Hz','1477 Hz','1633 Hz']
        row_headers = ['697 Hz','770 Hz','852 Hz','941 Hz']

        grid_f = tk.Frame(kb_card, bg=CARD)
        grid_f.pack()

        # Cabeceras de columna (frecuencias)
        for j, h in enumerate(col_headers):
            tk.Label(grid_f, text=h, font=("Segoe UI", 7), bg=CARD,
                     fg=YELLOW).grid(row=0, column=j+1, padx=4, pady=(0,2))

        for i, row_keys in enumerate(kb_rows):
            tk.Label(grid_f, text=row_headers[i], font=("Segoe UI", 7),
                     bg=CARD, fg=GREEN).grid(row=i+1, column=0, padx=(0,6), sticky="e")
            for j, key in enumerate(row_keys):
                color = DIGIT_COLORS.get(key, ACCENT)
                cell = tk.Frame(grid_f, bg=CARD2, width=46, height=38,
                                highlightbackground=BORDER, highlightthickness=1)
                cell.grid(row=i+1, column=j+1, padx=3, pady=3)
                cell.pack_propagate(False)
                lbl = tk.Label(cell, text=key,
                               font=("Segoe UI", 13, "bold"),
                               bg=CARD2, fg=TEXT_DIM)
                lbl.pack(expand=True)
                self._dtmf_key_labels[key] = (cell, lbl, color)

        # ── Tabla de detalle ──
        tbl_card = tk.Frame(parent, bg=CARD, padx=12, pady=10,
                            highlightbackground=BORDER, highlightthickness=1)
        tbl_card.grid(row=2, column=0, sticky="ew")

        tk.Label(tbl_card, text="DETALLE",
                 font=("Segoe UI", 7, "bold"), bg=CARD, fg=TEXT_DIM).pack(anchor="w", pady=(0,4))

        s = ttk.Style()
        s.theme_use("default")
        s.configure("P.Treeview", background=CARD, foreground=TEXT_MID,
                    fieldbackground=CARD, rowheight=20, font=("Segoe UI", 8),
                    borderwidth=0)
        s.configure("P.Treeview.Heading", background=CARD2, foreground=ACCENT,
                    font=("Segoe UI", 8, "bold"), relief="flat")
        s.map("P.Treeview", background=[("selected", CARD2)],
              foreground=[("selected", ACCENT)])

        cols = ("digit","row","col","conf","dur")
        self.tree = ttk.Treeview(tbl_card, columns=cols, show="headings",
                                 height=6, style="P.Treeview")
        for col, hdr, w in [("digit","#",35),("row","Fila",75),
                             ("col","Columna",80),("conf","Conf.",55),("dur","ms",50)]:
            self.tree.heading(col, text=hdr)
            self.tree.column(col, width=w, anchor="center", stretch=False)
        self.tree.pack(fill=tk.X)

        self.lbl_info = tk.Label(tbl_card, text="",
                                 font=("Segoe UI", 7), bg=CARD,
                                 fg=TEXT_DIM, justify="left")
        self.lbl_info.pack(anchor="w", pady=(6,0))

    # ══════════════════════════════════════════════════════
    #  VISTA MICRÓFONO
    # ══════════════════════════════════════════════════════
    def _build_mic_view(self, parent):
        # ── Header ──
        hdr = tk.Frame(parent, bg=BG, pady=12, padx=16)
        hdr.grid(row=0, column=0, sticky="ew")
        tk.Label(hdr, text="Micrófono en Vivo",
                 font=("Segoe UI", 14, "bold"), bg=BG, fg=TEXT).pack(side=tk.LEFT)
        self.lbl_mic_status = tk.Label(hdr, text="⚪  Inactivo",
                                       font=("Segoe UI", 9), bg=BG, fg=TEXT_DIM)
        self.lbl_mic_status.pack(side=tk.LEFT, padx=16)

        # ── Cuerpo ──
        body = tk.Frame(parent, bg=BG)
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0,12))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # ── Panel izquierdo: indicadores ──
        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,6))
        left.columnconfigure(0, weight=1)
        self._build_mic_indicators(left)

        # ── Panel derecho: gráficas ──
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self._build_mic_plots(right)

    def _build_mic_indicators(self, parent):
        # Dígito actual — grande
        dig_card = tk.Frame(parent, bg=CARD, padx=20, pady=20,
                            highlightbackground=BORDER, highlightthickness=1)
        dig_card.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        tk.Label(dig_card, text="DÍGITO DETECTADO",
                 font=("Segoe UI", 7, "bold"), bg=CARD, fg=TEXT_DIM).pack()
        self.lbl_live_digit = tk.Label(dig_card, text="—",
                                       font=("Segoe UI", 80, "bold"),
                                       bg=CARD, fg=GREEN, width=3)
        self.lbl_live_digit.pack()
        self.lbl_live_freq = tk.Label(dig_card, text="",
                                      font=("Segoe UI", 8),
                                      bg=CARD, fg=TEXT_DIM)
        self.lbl_live_freq.pack(pady=(4, 0))

        # Nivel de señal con barra horizontal
        lvl_card = tk.Frame(parent, bg=CARD, padx=16, pady=12,
                            highlightbackground=BORDER, highlightthickness=1)
        lvl_card.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        hrow = tk.Frame(lvl_card, bg=CARD)
        hrow.pack(fill=tk.X)
        tk.Label(hrow, text="NIVEL DE SEÑAL",
                 font=("Segoe UI", 7, "bold"), bg=CARD, fg=TEXT_DIM).pack(side=tk.LEFT)
        self.lbl_level_db = tk.Label(hrow, text="— dB",
                                     font=("Segoe UI", 8, "bold"), bg=CARD, fg=ACCENT)
        self.lbl_level_db.pack(side=tk.RIGHT)

        self.level_canvas = tk.Canvas(lvl_card, height=16, bg=CARD2,
                                      highlightthickness=0)
        self.level_canvas.pack(fill=tk.X, pady=(6, 0))

        # Secuencia capturada
        seq_card = tk.Frame(parent, bg=CARD, padx=16, pady=12,
                            highlightbackground=BORDER, highlightthickness=1)
        seq_card.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        hrow2 = tk.Frame(seq_card, bg=CARD)
        hrow2.pack(fill=tk.X)
        tk.Label(hrow2, text="SECUENCIA CAPTURADA",
                 font=("Segoe UI", 7, "bold"), bg=CARD, fg=TEXT_DIM).pack(side=tk.LEFT)
        tk.Button(hrow2, text="✕ Limpiar", font=("Segoe UI", 7),
                  bg=CARD, fg=RED, relief="flat", cursor="hand2",
                  command=self._clear_mic_sequence).pack(side=tk.RIGHT)

        self.lbl_mic_sequence = tk.Label(seq_card, text="",
                                         font=("Consolas", 22, "bold"),
                                         bg=CARD, fg=ACCENT,
                                         wraplength=220, justify="left")
        self.lbl_mic_sequence.pack(anchor="w", pady=(8, 0))

        # Botón — mandar al análisis
        self.btn_analizar_mic = tk.Button(
            seq_card,
            text="📊  Analizar en gráficas",
            font=("Segoe UI", 9, "bold"),
            bg=ACCENT, fg=BG,
            relief="flat", cursor="hand2",
            padx=10, pady=6,
            state=tk.DISABLED,
            command=self._analizar_secuencia_mic,
        )
        self.btn_analizar_mic.pack(anchor="w", pady=(10, 0))

        # Controles de sensibilidad
        sens_card = tk.Frame(parent, bg=CARD, padx=16, pady=12,
                             highlightbackground=BORDER, highlightthickness=1)
        sens_card.grid(row=3, column=0, sticky="ew")

        tk.Label(sens_card, text="SENSIBILIDAD DEL MICRÓFONO",
                 font=("Segoe UI", 7, "bold"), bg=CARD, fg=TEXT_DIM).pack(anchor="w", pady=(0,8))

        def _slider(lbl_text, from_, to, default, var_name, fmt):
            row = tk.Frame(sens_card, bg=CARD)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=lbl_text, font=("Segoe UI", 8),
                     bg=CARD, fg=TEXT_MID, width=14, anchor="w").pack(side=tk.LEFT)
            var = tk.DoubleVar(value=default)
            setattr(self, var_name, var)
            val_lbl = tk.Label(row, text=fmt.format(default),
                               font=("Segoe UI", 8, "bold"),
                               bg=CARD, fg=ORANGE, width=5)
            val_lbl.pack(side=tk.RIGHT)
            sl = ttk.Scale(row, from_=from_, to=to, orient=tk.HORIZONTAL,
                           variable=var, length=120,
                           command=lambda v, l=val_lbl, f=fmt: l.config(text=f.format(float(v))))
            sl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        _slider("RMS mínimo",  0.002, 0.10, MIN_RMS,       "_slider_rms",  "{:.3f}")
        _slider("SNR mínimo",  2.0,   20.0, SNR_THRESHOLD, "_slider_snr",  "{:.1f}")
        _slider("Dominancia",  1.5,   10.0, PEAK_RATIO,    "_slider_peak", "{:.1f}")

        tk.Label(sens_card, text="← más sensible          menos sensible →",
                 font=("Segoe UI", 7), bg=CARD, fg=TEXT_DIM).pack(pady=(6,0))

    def _build_mic_plots(self, parent):
        self.fig_mic = plt.Figure(tight_layout=dict(pad=1.2))
        gs = mgridspec.GridSpec(2, 1, figure=self.fig_mic, hspace=0.5)

        self.ax_mic_wave = self.fig_mic.add_subplot(gs[0])
        self.ax_mic_fft  = self.fig_mic.add_subplot(gs[1])

        self._style_ax(self.ax_mic_wave, "Señal de Audio en Vivo  (últimos 4 s)",
                       "Tiempo (s)", "Amplitud")
        self._style_ax(self.ax_mic_fft,  "Espectro de Frecuencias en Vivo",
                       "Frecuencia (Hz)", "Magnitud")
        self.ax_mic_fft.set_xlim(600, 1800)

        canvas_mic = FigureCanvasTkAgg(self.fig_mic, master=parent)
        canvas_mic.draw()
        canvas_mic.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_mic = canvas_mic

    # ══════════════════════════════════════════════════════
    #  VISTA TEORÍA DE FOURIER
    # ══════════════════════════════════════════════════════
    def _build_teoria_view(self, parent):
        # ── Header ──
        hdr = tk.Frame(parent, bg=BG, pady=12, padx=16)
        hdr.grid(row=0, column=0, sticky="ew")
        tk.Label(hdr, text="Teoría de Fourier aplicada a DTMF",
                 font=("Segoe UI", 14, "bold"), bg=BG, fg=TEXT).pack(side=tk.LEFT)
        tk.Label(hdr, text="Matemáticas IV  •  Transformada de Fourier",
                 font=("Segoe UI", 9), bg=BG, fg=TEXT_DIM).pack(side=tk.RIGHT)

        # ── Selector de demostración ──
        ctrl = tk.Frame(parent, bg=BG, padx=16)
        ctrl.grid(row=1, column=0, sticky="ew")

        tk.Label(ctrl, text="Selecciona la demostración:",
                 font=("Segoe UI", 9), bg=BG, fg=TEXT_MID).pack(side=tk.LEFT)

        self._teoria_var = tk.StringVar(value="construccion")
        demos = [
            ("construccion", "1 · Construcción de la señal DTMF"),
            ("dft",          "2 · La DFT paso a paso"),
            ("leakage",      "3 · Ventana de Hanning vs Sin ventana"),
            ("tabla",        "4 · Tabla de frecuencias DTMF"),
        ]
        for val, txt in demos:
            tk.Radiobutton(ctrl, text=txt, variable=self._teoria_var, value=val,
                           font=("Segoe UI", 9), bg=BG, fg=TEXT_MID,
                           selectcolor=CARD2, activebackground=BG,
                           activeforeground=ACCENT,
                           command=self._render_teoria).pack(side=tk.LEFT, padx=10)

        # ── Área de gráfica ──
        plot_frame = tk.Frame(parent, bg=BG)
        plot_frame.grid(row=2, column=0, sticky="nsew", padx=12, pady=(6,12))
        parent.rowconfigure(2, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.fig_teoria = plt.Figure(tight_layout=dict(pad=1.6))
        self.canvas_teoria = FigureCanvasTkAgg(self.fig_teoria, master=plot_frame)
        self.canvas_teoria.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # ── Panel explicativo lateral ──
        self._teoria_text_var = tk.StringVar()
        txt_frame = tk.Frame(parent, bg=CARD, padx=16, pady=14,
                             highlightbackground=BORDER, highlightthickness=1)
        txt_frame.grid(row=3, column=0, sticky="ew", padx=12, pady=(0,12))
        tk.Label(txt_frame, text="¿QUÉ ESTÁS VIENDO?",
                 font=("Segoe UI", 7, "bold"), bg=CARD, fg=TEXT_DIM).pack(anchor="w")
        self.lbl_teoria_txt = tk.Label(txt_frame, textvariable=self._teoria_text_var,
                                       font=("Segoe UI", 9), bg=CARD, fg=TEXT_MID,
                                       justify="left", wraplength=1100)
        self.lbl_teoria_txt.pack(anchor="w", pady=(4,0))

        self._render_teoria()

    def _render_teoria(self):
        demo = self._teoria_var.get()
        self.fig_teoria.clf()

        if   demo == "construccion": self._demo_construccion()
        elif demo == "dft":          self._demo_dft()
        elif demo == "leakage":      self._demo_leakage()
        elif demo == "tabla":        self._demo_tabla()

        self.canvas_teoria.draw()

    # ── Demo 1: Construcción de la señal DTMF ─────────────
    def _demo_construccion(self):
        SR   = 8000
        dur  = 0.3
        t    = np.linspace(0, dur, int(SR * dur), endpoint=False)

        # Dígito 5 → 770 Hz + 1336 Hz
        digit  = "5"
        f_row, f_col = 770, 1336
        s_row  = np.sin(2 * np.pi * f_row  * t)
        s_col  = np.sin(2 * np.pi * f_col  * t)
        s_dtmf = 0.5 * (s_row + s_col)

        freqs, M_row  = compute_fft(s_row,  SR)
        _,     M_col  = compute_fft(s_col,  SR)
        _,     M_dtmf = compute_fft(s_dtmf, SR)
        mask = freqs <= 2000

        gs = mgridspec.GridSpec(2, 3, figure=self.fig_teoria,
                                hspace=0.55, wspace=0.38)
        axes = [self.fig_teoria.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

        step = max(1, len(t)//1200)

        # Fila 1: señales en tiempo
        for ax, sig, color, label in [
            (axes[0], s_row,  GREEN,  f"Componente de Fila\n{f_row} Hz"),
            (axes[1], s_col,  YELLOW, f"Componente de Columna\n{f_col} Hz"),
            (axes[2], s_dtmf, ACCENT, f"Señal DTMF '{digit}'\n(suma de ambas × 0.5)"),
        ]:
            ax.plot(t[::step]*1000, sig[::step], color=color, lw=1.2)
            ax.fill_between(t[::step]*1000, sig[::step], alpha=0.15, color=color)
            self._style_ax(ax, label, "Tiempo (ms)", "Amplitud")
            ax.set_xlim(0, dur*1000)
            ax.set_ylim(-1.15, 1.15)
            # Fórmula
            freq_val = f_row if color == GREEN else (f_col if color == YELLOW else None)
            if freq_val:
                ax.text(0.97, 0.95, f"sin(2π·{freq_val}·t)",
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=7.5, color=color, style='italic',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=CARD, alpha=0.8))

        # Fila 2: espectros FFT
        for ax, M, color, label in [
            (axes[3], M_row,  GREEN,  f"FFT — {f_row} Hz"),
            (axes[4], M_col,  YELLOW, f"FFT — {f_col} Hz"),
            (axes[5], M_dtmf, ACCENT, f"FFT — Señal DTMF '{digit}'"),
        ]:
            ax.fill_between(freqs[mask], M[mask], alpha=0.25, color=color)
            ax.plot(freqs[mask], M[mask], color=color, lw=1.2)
            self._style_ax(ax, label, "Frecuencia (Hz)", "|X(f)|")
            ax.set_xlim(400, 1800)
            # Marcar el pico
            peak_idx = np.argmax(M[mask])
            peak_f   = freqs[mask][peak_idx]
            ax.axvline(x=peak_f, color=color, lw=1.5, ls='--', alpha=0.8)
            ax.text(peak_f, M[mask].max()*0.85, f"{peak_f:.0f} Hz",
                    ha='center', fontsize=7, color=color, fontweight='bold')

        self._teoria_text_var.set(
            "La señal DTMF se construye como la SUMA de dos senoides puras: una de baja frecuencia (fila) y una de alta frecuencia (columna).\n"
            f"Para el dígito '{digit}': s(t) = sin(2π·{f_row}·t)  +  sin(2π·{f_col}·t)\n"
            "La FFT (Transformada Discreta de Fourier) descompone esta señal y revela exactamente esas dos frecuencias como picos en el espectro — eso es lo que permite identificar el dígito."
        )

    # ── Demo 2: DFT paso a paso ───────────────────────────
    def _demo_dft(self):
        N     = 64
        SR    = 8000
        t     = np.arange(N) / SR
        f1, f2 = 770, 1336
        x     = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)

        # ── DFT calculada manualmente con la fórmula explícita ──
        # X[k] = Σ x[n] · e^(-j2πkn/N)  →  |X[k]| / N
        K         = N // 2
        freqs_dft = np.arange(K) * SR / N
        X_mag     = np.zeros(K)
        for k in range(K):
            n_arr      = np.arange(N)
            real_part  = np.sum(x * np.cos(2*np.pi*k*n_arr/N))
            imag_part  = np.sum(x * np.sin(2*np.pi*k*n_arr/N))
            X_mag[k]   = np.sqrt(real_part**2 + imag_part**2) / N

        # ── FFT manual (Cooley-Tukey) para comparar ──
        from fft_manual import rfft, rfftfreq, zero_pad
        xpad  = zero_pad(x)
        X_fft = np.abs(rfft(xpad)) / len(xpad)
        f_fft = rfftfreq(len(xpad), 1/SR)

        gs = mgridspec.GridSpec(2, 2, figure=self.fig_teoria,
                                hspace=0.58, wspace=0.38)

        # ── Gráfica 1: señal discreta x[n] (usando vlines, sin stem) ──
        ax1 = self.fig_teoria.add_subplot(gs[0, :])
        ax1.vlines(np.arange(N), 0, x, color=ACCENT, lw=0.9, alpha=0.8)
        ax1.plot(np.arange(N), x, 'o', color=ACCENT,
                 markersize=2.5, markeredgewidth=0)
        ax1.axhline(0, color=BORDER, lw=0.8)
        self._style_ax(ax1,
            f"Señal discreta  x[n]  —  N={N} muestras  "
            f"(f₁={f1} Hz  +  f₂={f2} Hz,  fs={SR} Hz)",
            "Muestra  n", "x[n]")
        # Anotar la fórmula dentro del gráfico
        ax1.text(0.99, 0.95,
                 f"x[n] = sin(2π·{f1}·n/fs) + sin(2π·{f2}·n/fs)",
                 transform=ax1.transAxes, ha='right', va='top',
                 fontsize=8, color=ACCENT, style='italic',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor=CARD,
                           edgecolor=BORDER, alpha=0.9))

        # ── Gráfica 2: DFT manual ──
        ax2 = self.fig_teoria.add_subplot(gs[1, 0])
        ax2.bar(freqs_dft, X_mag, width=SR/N*0.65,
                color=ACCENT2, alpha=0.85, zorder=3)
        # Etiquetar los dos picos dominantes
        for k in np.where(X_mag > X_mag.max()*0.25)[0]:
            ax2.text(freqs_dft[k], X_mag[k] + X_mag.max()*0.04,
                     f"{freqs_dft[k]:.0f} Hz",
                     ha='center', fontsize=7, color=ACCENT2, fontweight='bold')
        self._style_ax(ax2,
            "DFT calculada manualmente\n"
            r"$X[k]=\sum_{n=0}^{N-1}x[n]\,e^{-j2\pi kn/N}$"
            f"   (O(N²), N={N})",
            "Frecuencia (Hz)", "|X[k]| / N")
        ax2.set_xlim(0, 2200)
        ax2.set_ylim(0, X_mag.max() * 1.3)
        ax2.grid(True, alpha=0.3)

        # ── Gráfica 3: FFT manual Cooley-Tukey ──
        ax3 = self.fig_teoria.add_subplot(gs[1, 1])
        mask_fft = f_fft <= 2200
        ax3.bar(f_fft[mask_fft], X_fft[mask_fft],
                width=SR/len(xpad)*0.65,
                color=TEAL, alpha=0.85, zorder=3)
        for i in np.where(X_fft[mask_fft] > X_fft[mask_fft].max()*0.25)[0]:
            ax3.text(f_fft[mask_fft][i],
                     X_fft[mask_fft][i] + X_fft[mask_fft].max()*0.04,
                     f"{f_fft[mask_fft][i]:.0f} Hz",
                     ha='center', fontsize=7, color=TEAL, fontweight='bold')
        self._style_ax(ax3,
            "FFT manual (Cooley-Tukey radix-2)\n"
            f"Resultado idéntico   (O(N log N), N={len(xpad)} con zero-pad)",
            "Frecuencia (Hz)", "|X[k]| / N")
        ax3.set_xlim(0, 2200)
        ax3.set_ylim(0, X_fft[mask_fft].max() * 1.3)
        ax3.grid(True, alpha=0.3)

        # Verificar que son iguales numéricamente
        err = float(np.max(np.abs(
            X_mag - X_fft[np.round(f_fft[:K]).astype(int) < SR//2 + 1][:K]
        ))) if len(X_fft) >= K else float('nan')

        self._teoria_text_var.set(
            f"La DFT se define como:  X[k] = Σ x[n]·exp(−j2πkn/N)  para k = 0, 1, …, N−1\n"
            f"Izquierda: calculada con dos bucles for explícitos — complejidad O(N²) = O({N}²) = {N**2:,} operaciones.\n"
            f"Derecha: misma señal con FFT manual Cooley-Tukey radix-2 — complejidad O(N·log₂N) = O({len(xpad)}·{int(np.log2(len(xpad)))}) = {len(xpad)*int(np.log2(len(xpad))):,} operaciones. "
            f"Los picos aparecen en {f1} Hz y {f2} Hz — exactamente las frecuencias del dígito DTMF '5'."
        )

    # ── Demo 3: Leakage — con y sin ventana Hanning ───────
    def _demo_leakage(self):
        SR   = 8000
        N    = 256
        t    = np.arange(N) / SR
        # Frecuencia NO alineada con la grilla DFT (peor caso de leakage)
        f_signal = 823.0
        x = np.sin(2 * np.pi * f_signal * t)

        win_rect  = np.ones(N)
        win_hann  = np.hanning(N)

        def fft_mag(sig, win):
            s = sig * win
            X = np.abs(np.fft.rfft(s, n=N)) / (N * np.mean(win) + 1e-9)
            return np.fft.rfftfreq(N, 1/SR), X

        f_rect, M_rect = fft_mag(x, win_rect)
        f_hann, M_hann = fft_mag(x, win_hann)

        gs = mgridspec.GridSpec(2, 3, figure=self.fig_teoria,
                                hspace=0.55, wspace=0.38)

        # Ventanas en tiempo
        ax1 = self.fig_teoria.add_subplot(gs[0, 0])
        ax1.plot(np.arange(N), win_rect, color=RED,   lw=1.4, label="Rectangular")
        ax1.plot(np.arange(N), win_hann, color=GREEN, lw=1.4, label="Hanning")
        ax1.fill_between(np.arange(N), win_hann, alpha=0.2, color=GREEN)
        self._style_ax(ax1, "Funciones de ventana", "Muestra n", "w[n]")
        ax1.legend(fontsize=7)
        ax1.set_ylim(-0.05, 1.15)

        # Señal enventanada
        ax2 = self.fig_teoria.add_subplot(gs[0, 1])
        step = max(1, N//300)
        ax2.plot(np.arange(N)[::step], (x*win_rect)[::step],
                 color=RED,   lw=1.0, alpha=0.8, label="x·w_rect")
        ax2.plot(np.arange(N)[::step], (x*win_hann)[::step],
                 color=GREEN, lw=1.0, alpha=0.8, label="x·w_hann")
        self._style_ax(ax2, "Señal × Ventana", "Muestra n", "Amplitud")
        ax2.legend(fontsize=7)

        # Fórmula ventana Hanning
        ax_f = self.fig_teoria.add_subplot(gs[0, 2])
        ax_f.axis('off')
        formula_txt = (
            "Ventana de Hanning:\n\n"
            "w[n] = 0.5 · (1 − cos(2πn / (N−1)))\n\n"
            "Reduce el spectral leakage\n"
            "suavizando los bordes de la\n"
            "señal antes de calcular la DFT.\n\n"
            "Costo: ligera pérdida de\n"
            "resolución en frecuencia."
        )
        ax_f.text(0.08, 0.92, formula_txt, transform=ax_f.transAxes,
                  va='top', fontsize=9, color=TEXT_MID, linespacing=1.7,
                  bbox=dict(boxstyle='round,pad=0.8', facecolor=CARD2,
                            edgecolor=BORDER, alpha=0.95))

        # Espectros comparados
        mask = f_rect <= 2000
        ax3 = self.fig_teoria.add_subplot(gs[1, 0])
        ax3.fill_between(f_rect[mask], M_rect[mask], alpha=0.3, color=RED)
        ax3.plot(f_rect[mask], M_rect[mask], color=RED, lw=1.1)
        ax3.axvline(x=f_signal, color=TEXT_DIM, lw=1, ls='--')
        ax3.text(f_signal+20, M_rect[mask].max()*0.9, f"{f_signal} Hz",
                 fontsize=7, color=TEXT_DIM)
        self._style_ax(ax3, "FFT SIN ventana (rectangular)\n→ Spectral Leakage visible",
                       "Frecuencia (Hz)", "|X(f)|")

        ax4 = self.fig_teoria.add_subplot(gs[1, 1])
        ax4.fill_between(f_hann[mask], M_hann[mask], alpha=0.3, color=GREEN)
        ax4.plot(f_hann[mask], M_hann[mask], color=GREEN, lw=1.1)
        ax4.axvline(x=f_signal, color=TEXT_DIM, lw=1, ls='--')
        ax4.text(f_signal+20, M_hann[mask].max()*0.9, f"{f_signal} Hz",
                 fontsize=7, color=TEXT_DIM)
        self._style_ax(ax4, "FFT CON ventana Hanning\n→ Pico limpio, sin derrame",
                       "Frecuencia (Hz)", "|X(f)|")

        # Diferencia
        ax5 = self.fig_teoria.add_subplot(gs[1, 2])
        diff = np.abs(M_rect[mask] - M_hann[mask])
        ax5.fill_between(f_rect[mask], diff, alpha=0.4, color=YELLOW)
        ax5.plot(f_rect[mask], diff, color=YELLOW, lw=1.1)
        self._style_ax(ax5, "Diferencia  |FFT_rect − FFT_hann|\n→ Energía 'derramada' eliminada",
                       "Frecuencia (Hz)", "Diferencia")

        self._teoria_text_var.set(
            "Spectral Leakage (derrame espectral): ocurre cuando la frecuencia de la señal NO cae exactamente en un bin de la DFT. "
            "La ventana rectangular trata los bordes de la señal como discontinuidades, generando 'lóbulos laterales' falsos en el espectro.\n"
            f"Señal de prueba: {f_signal} Hz — frecuencia NO alineada con la grilla DFT (N={N}, SR={SR} Hz → resolución = {SR/N:.1f} Hz/bin).\n"
            "La ventana de Hanning: w[n] = 0.5·(1 − cos(2πn/(N−1))) atenúa suavemente los extremos → los lóbulos laterales desaparecen → detección más precisa de los tonos DTMF."
        )

    # ── Demo 4: Tabla de frecuencias DTMF ─────────────────
    def _demo_tabla(self):
        SR  = 8000
        dur = 0.25
        t   = np.linspace(0, dur, int(SR*dur), endpoint=False)

        gs = mgridspec.GridSpec(4, 4, figure=self.fig_teoria,
                                hspace=0.15, wspace=0.15)

        labels = [['1','2','3','A'], ['4','5','6','B'],
                  ['7','8','9','C'], ['*','0','#','D']]
        col_headers = [1209, 1336, 1477, 1633]
        row_headers = [697,  770,  852,  941]

        for ri, f_row in enumerate(row_headers):
            for ci, f_col in enumerate(col_headers):
                ax = self.fig_teoria.add_subplot(gs[ri, ci])
                digit = labels[ri][ci]
                color = DIGIT_COLORS.get(digit, ACCENT)

                s = np.sin(2*np.pi*f_row*t) + np.sin(2*np.pi*f_col*t)
                freqs, M = compute_fft(s, SR)
                mask = (freqs >= 500) & (freqs <= 1800)

                ax.fill_between(freqs[mask], M[mask], alpha=0.3, color=color)
                ax.plot(freqs[mask], M[mask], color=color, lw=1.0)
                ax.axvline(x=f_row, color=GREEN,  lw=1.2, ls='--', alpha=0.7)
                ax.axvline(x=f_col, color=YELLOW, lw=1.2, ls='--', alpha=0.7)

                ax.set_facecolor(CARD)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(1.5)

                # Título dentro de la celda
                ax.text(0.5, 0.85, digit, transform=ax.transAxes,
                        ha='center', fontsize=13, fontweight='bold',
                        color=color)
                ax.text(0.5, 0.08,
                        f"{f_row}+{f_col}",
                        transform=ax.transAxes,
                        ha='center', fontsize=5.5, color=TEXT_DIM)

        # Etiquetas de filas y columnas
        for ci, fc in enumerate(col_headers):
            self.fig_teoria.text(
                0.13 + ci * 0.215, 0.97,
                f"↔ {fc} Hz", ha='center', fontsize=8,
                color=YELLOW, fontweight='bold')
        for ri, fr in enumerate(row_headers):
            self.fig_teoria.text(
                0.01, 0.84 - ri * 0.215,
                f"↕\n{fr}", ha='center', fontsize=7,
                color=GREEN, fontweight='bold')

        self._teoria_text_var.set(
            "Los 16 dígitos DTMF se forman combinando 4 frecuencias de fila (bajas: 697, 770, 852, 941 Hz) con 4 frecuencias de columna (altas: 1209, 1336, 1477, 1633 Hz).\n"
            "Cada celda muestra el espectro FFT del tono correspondiente — siempre aparecen exactamente DOS picos: uno verde (fila) y uno amarillo (columna).\n"
            "Esto se diseñó así en 1963 (Bell Labs) para que la FFT pueda identificar cualquier dígito de forma unívoca sin ambigüedad, incluso con ruido de fondo."
        )

    # ══════════════════════════════════════════════════════
    #  MICRÓFONO — lógica
    # ══════════════════════════════════════════════════════
    def _toggle_mic(self):
        if self._mic_active: self._stop_mic()
        else:                self._start_mic()

    def _start_mic(self):
        self._switch_view("mic")
        try:
            self._mic_buffer.clear()
            self._mic_chunk.clear()
            self._mic_active = True
            chunk_size = int(MIC_SR * MIC_CHUNK_MS / 1000)

            def _cb(indata, frames, ti, status):
                mono = indata[:, 0].astype(np.float32)
                self._mic_buffer.extend(mono)
                self._mic_chunk.extend(mono)

            self._mic_stream = sd.InputStream(samplerate=MIC_SR, channels=1,
                                               dtype='float32', blocksize=chunk_size,
                                               callback=_cb)
            self._mic_stream.start()
            self.mic_btn_lbl.config(text="🔴  Detener Micrófono", fg=RED)
            self.mic_btn_frame.config(bg="#2a1a1a")
            self.mic_btn_lbl.config(bg="#2a1a1a")
            self.lbl_mic_status.config(text="🔴  Escuchando…  acerca tu celular", fg=GREEN)
            self.status_var.set("Micrófono activo")
            self._mic_update_loop()
        except Exception as e:
            self._mic_active = False
            messagebox.showerror("Error", f"No se pudo abrir el micrófono:\n{e}")

    def _stop_mic(self):
        self._mic_active = False
        if self._mic_stream:
            self._mic_stream.stop(); self._mic_stream.close()
            self._mic_stream = None
        self.mic_btn_lbl.config(text="🎙  Activar Micrófono", fg=ORANGE,
                                 bg="#1a1f2e")
        self.mic_btn_frame.config(bg="#1a1f2e")
        self.lbl_mic_status.config(text="⚪  Inactivo", fg=TEXT_DIM)
        self.status_var.set("Micrófono detenido")
        self.lbl_live_digit.config(text="—", fg=GREEN)
        self.lbl_live_freq.config(text="")

    def _mic_update_loop(self):
        if not self._mic_active:
            return
        chunk_size = int(MIC_SR * MIC_CHUNK_MS / 1000)
        if len(self._mic_chunk) >= chunk_size:
            segment = np.array(list(self._mic_chunk), dtype=np.float32)
            result  = detect_digit(segment, MIC_SR,
                                   min_rms=self._slider_rms.get(),
                                   snr_threshold=self._slider_snr.get(),
                                   peak_ratio=self._slider_peak.get())
            self._process_live_digit(result)
            rms = float(np.sqrt(np.mean(segment ** 2)))
            db  = 20 * np.log10(rms + 1e-9)
            self._update_level_meter(db)
            self._update_mic_plots(segment, result)
        self.after(60, self._mic_update_loop)

    def _process_live_digit(self, result):
        digit = result['digit']
        now   = time.time()
        if digit:
            color = DIGIT_COLORS.get(digit, GREEN)
            self.lbl_live_digit.config(text=digit, fg=color)
            self.lbl_live_freq.config(
                text=f"↕ Fila: {result['row_freq']} Hz   ↔ Col: {result['col_freq']} Hz",
                fg=color)
            if digit != self._last_digit or (now - self._last_digit_t) > self._digit_cooldown:
                self._mic_digits.append(digit)
                self._last_digit, self._last_digit_t = digit, now
                seq = ''.join(self._mic_digits)
                self.lbl_mic_sequence.config(text=seq)
                # Habilitar botón de análisis
                self.btn_analizar_mic.config(state=tk.NORMAL)
            # iluminar tecla en teclado DTMF del panel archivo
            self._highlight_dtmf_key(digit)
        else:
            self.lbl_live_digit.config(text="—", fg=TEXT_DIM)
            self.lbl_live_freq.config(text="")
            if (now - self._last_digit_t) > self._digit_cooldown:
                self._last_digit = None

    def _highlight_dtmf_key(self, active_digit):
        """Ilumina brevemente la tecla detectada en la vista de archivo."""
        for k, (cell, lbl, color) in self._dtmf_key_labels.items():
            if k == active_digit:
                cell.config(bg=color, highlightbackground=color)
                lbl.config(bg=color, fg=BG)
            else:
                cell.config(bg=CARD2, highlightbackground=BORDER)
                lbl.config(bg=CARD2, fg=TEXT_DIM)

    def _update_level_meter(self, db: float):
        c = self.level_canvas
        c.update_idletasks()
        w = c.winfo_width() or 200
        h = 16
        c.delete("all")
        fill  = max(0.0, min(1.0, (db + 60) / 60))
        bar_w = int(fill * w)
        # Fondo segmentado
        segs = 30
        seg_w = w / segs
        for i in range(segs):
            pct = i / segs
            col = GREEN if pct < 0.6 else (YELLOW if pct < 0.85 else RED)
            x0 = int(i * seg_w) + 1
            x1 = int((i + 1) * seg_w) - 1
            alpha_col = col if (x0 < bar_w) else CARD2
            c.create_rectangle(x0, 2, x1, h-2, fill=alpha_col, outline="")
        self.lbl_level_db.config(text=f"{db:.0f} dB")

    def _update_mic_plots(self, segment, result):
        # Onda
        ax = self.ax_mic_wave
        ax.cla()
        self._style_ax(ax, "Señal de Audio en Vivo  (últimos 4 s)", "Tiempo (s)", "Amplitud")
        buf  = np.array(list(self._mic_buffer), dtype=np.float32)
        if len(buf) > 0:
            t    = np.linspace(-len(buf) / MIC_SR, 0, len(buf))
            step = max(1, len(buf) // 3000)
            ax.plot(t[::step], buf[::step], color=ORANGE, linewidth=0.7, alpha=0.9)
            ax.fill_between(t[::step], buf[::step], alpha=0.15, color=ORANGE)
        ax.set_xlim(-MIC_HISTORY_S, 0)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(0, color=BORDER, linewidth=0.8, alpha=0.5)

        # FFT
        ax2 = self.ax_mic_fft
        ax2.cla()
        self._style_ax(ax2, "Espectro de Frecuencias en Vivo", "Frecuencia (Hz)", "Magnitud")
        freqs, mag = compute_fft(segment, MIC_SR)
        mask = freqs <= 2000
        f, m = freqs[mask], mag[mask]
        ax2.fill_between(f, m, alpha=0.25, color=TEAL)
        ax2.plot(f, m, color=TEAL, linewidth=1.2)

        for freq in ROW_FREQS:
            ax2.axvline(x=freq, color=GREEN,  lw=1.0, ls='--', alpha=0.6)
        for freq in COL_FREQS:
            ax2.axvline(x=freq, color=YELLOW, lw=1.0, ls='--', alpha=0.6)

        if result['digit'] and len(m) > 0:
            peak = m.max()
            for freq, col in [(result['row_freq'], GREEN), (result['col_freq'], YELLOW)]:
                ax2.axvline(x=freq, color=col, lw=2.5, alpha=1.0)
                ax2.annotate(f"{freq} Hz",
                             xy=(freq, peak * 0.9), xytext=(freq + 40, peak * 0.9),
                             fontsize=7, color=col, fontweight='bold',
                             arrowprops=dict(arrowstyle='->', color=col, lw=1.2))
        ax2.set_xlim(600, 1800)
        self.canvas_mic.draw_idle()

    def _analizar_secuencia_mic(self):
        """Genera audio DTMF de la secuencia capturada y lo manda al análisis."""
        seq = ''.join(self._mic_digits)
        if not seq:
            return

        # Generar la señal DTMF sintética de la secuencia capturada
        samples = generate_dtmf_sequence(
            seq, tone_ms=300, silence_ms=100, sample_rate=MIC_SR)

        self.samples     = samples
        self.sample_rate = MIC_SR
        self.status_var.set(f"Desde micrófono: '{seq}'  •  {len(samples)/MIC_SR:.1f} s")

        # Ir a la vista de análisis y procesar
        self._switch_view("file")
        self._analyze_and_plot()

    def _clear_mic_sequence(self):
        self._mic_digits.clear()
        self._last_digit = None
        self.lbl_mic_sequence.config(text="")
        self.lbl_live_digit.config(text="—", fg=GREEN)
        self.lbl_live_freq.config(text="")
        self.btn_analizar_mic.config(state=tk.DISABLED)
        for k, (cell, lbl, color) in self._dtmf_key_labels.items():
            cell.config(bg=CARD2, highlightbackground=BORDER)
            lbl.config(bg=CARD2, fg=TEXT_DIM)

    # ══════════════════════════════════════════════════════
    #  ANÁLISIS Y GRÁFICAS (Archivo)
    # ══════════════════════════════════════════════════════
    def _analyze_and_plot(self):
        if self.samples is None:
            return
        s, sr = self.samples, self.sample_rate
        self.results = analyze_audio(s, sr, segment_ms=50)
        self.grouped = group_digits(self.results)
        freqs_full, mag_full = compute_fft(s, sr)

        self._plot_waveform(s, sr)
        self._plot_fft(freqs_full, mag_full)
        self._plot_spectrogram(s, sr)
        self.canvas.draw()
        self._update_results()

    def _plot_waveform(self, samples, sr):
        ax = self.ax_wave
        ax.cla()
        self._style_ax(ax, "Forma de Onda", "Tiempo (s)", "Amplitud")
        t    = np.linspace(0, len(samples) / sr, len(samples))
        step = max(1, len(samples) // 8000)
        ax.plot(t[::step], samples[::step], color=ACCENT, lw=0.7, alpha=0.9)
        ax.fill_between(t[::step], samples[::step], alpha=0.1, color=ACCENT)
        ax.axhline(0, color=BORDER, lw=0.8)

        for g in self.grouped:
            color = DIGIT_COLORS.get(g['digit'], GREEN)
            ax.axvspan(g['time_start'], g['time_end'],
                       alpha=0.2, color=color, linewidth=0)
            mid = (g['time_start'] + g['time_end']) / 2
            ax.text(mid, 0.88, g['digit'],
                    transform=ax.get_xaxis_transform(),
                    ha='center', fontsize=9, fontweight='bold', color=color,
                    path_effects=[pe.withStroke(linewidth=2, foreground=CARD)])

        ax.set_xlim(0, len(samples) / sr)
        ax.set_ylim(-1.15, 1.15)

    def _plot_fft(self, freqs, magnitude):
        ax = self.ax_fft
        ax.cla()
        self._style_ax(ax, "Espectro FFT — Frecuencias Dominantes",
                       "Frecuencia (Hz)", "Magnitud")
        mask = freqs <= 2000
        f, m = freqs[mask], magnitude[mask]
        ax.fill_between(f, m, alpha=0.2, color=ACCENT2)
        ax.plot(f, m, color=ACCENT2, lw=1.2)

        for freq in ROW_FREQS:
            ax.axvline(x=freq, color=GREEN,  lw=1.0, ls='--', alpha=0.7)
            ax.text(freq, m.max() * 1.02 if m.max() > 0 else 0.01,
                    f"{freq}", fontsize=6, color=GREEN, ha='center', va='bottom')
        for freq in COL_FREQS:
            ax.axvline(x=freq, color=YELLOW, lw=1.0, ls='--', alpha=0.7)
            ax.text(freq, m.max() * 1.02 if m.max() > 0 else 0.01,
                    f"{freq}", fontsize=6, color=YELLOW, ha='center', va='bottom')
        ax.set_xlim(600, 1800)

        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0],[0], color=GREEN,  ls='--', lw=1.2, label='Frec. de Fila'),
            Line2D([0],[0], color=YELLOW, ls='--', lw=1.2, label='Frec. de Columna'),
        ], fontsize=7, loc='upper right', framealpha=0.8)

    def _plot_spectrogram(self, samples, sr):
        ax = self.ax_spec
        ax.cla()
        self._style_ax(ax, "Espectrograma — cada mancha brillante es un tono DTMF",
                       "Tiempo (s)", "Frecuencia (Hz)")

        nfft     = 1024
        noverlap = nfft * 3 // 4
        Pxx, freqs_sg, t_sg, im = ax.specgram(
            samples, NFFT=nfft, Fs=sr, noverlap=noverlap,
            cmap='magma', scale='dB', vmin=-100, vmax=-20)
        ax.set_ylim(500, 1900)

        for freq in ROW_FREQS:
            ax.axhline(y=freq, color=GREEN,  lw=0.8, ls='--', alpha=0.6)
            ax.text(0.005, freq + 20, f"{freq}", transform=ax.get_yaxis_transform(),
                    fontsize=6, color=GREEN, va='bottom')
        for freq in COL_FREQS:
            ax.axhline(y=freq, color=YELLOW, lw=0.8, ls='--', alpha=0.6)
            ax.text(0.005, freq + 20, f"{freq}", transform=ax.get_yaxis_transform(),
                    fontsize=6, color=YELLOW, va='bottom')

        for g in self.grouped:
            color = DIGIT_COLORS.get(g['digit'], ACCENT2)
            mid_t = (g['time_start'] + g['time_end']) / 2
            ax.axvspan(g['time_start'], g['time_end'],
                       alpha=0.15, color=color, linewidth=0)
            for freq in (g['row_freq'], g['col_freq']):
                ax.plot(mid_t, freq, 'o', color=color, markersize=5,
                        markeredgecolor=BG, markeredgewidth=0.8, alpha=0.9)
            ax.text(mid_t, 1870, g['digit'], ha='center', va='top',
                    fontsize=8, fontweight='bold', color=color,
                    path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

        try:
            cbar = self.fig.colorbar(im, ax=ax, pad=0.005, fraction=0.012)
            cbar.set_label("dB", color=TEXT_DIM, fontsize=7)
            cbar.ax.yaxis.set_tick_params(labelsize=6, color=TEXT_DIM)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_DIM)
        except Exception:
            pass
        ax.grid(False)

    def _update_results(self):
        seq = ''.join(g['digit'] for g in self.grouped)
        self.lbl_digits.config(text=seq if seq else "—")

        # Métricas
        dur  = len(self.samples) / self.sample_rate
        conf = np.mean([g['confidence_avg'] for g in self.grouped]) if self.grouped else 0
        self.mc_digits.config(text=str(len(self.grouped)))
        self.mc_dur.config(text=f"{dur:.1f}")
        self.mc_sr.config(text=f"{self.sample_rate//1000}k")
        self.mc_conf.config(text=f"{conf:.0%}".replace('%',''))

        # Iluminar teclado DTMF
        detected = {g['digit'] for g in self.grouped}
        for k, (cell, lbl, color) in self._dtmf_key_labels.items():
            if k in detected:
                cell.config(bg=color, highlightbackground=color)
                lbl.config(bg=color, fg=BG)
            else:
                cell.config(bg=CARD2, highlightbackground=BORDER)
                lbl.config(bg=CARD2, fg=TEXT_DIM)

        # Tabla
        for item in self.tree.get_children():
            self.tree.delete(item)
        for g in self.grouped:
            dur_ms = int((g['time_end'] - g['time_start']) * 1000)
            self.tree.insert("", tk.END, values=(
                g['digit'], f"{g['row_freq']}Hz", f"{g['col_freq']}Hz",
                f"{g['confidence_avg']:.0%}", f"{dur_ms}"))

        self.lbl_info.config(
            text=f"{self.sample_rate} Hz  •  {len(self.samples):,} muestras  •  {dur:.2f} s")

    # ══════════════════════════════════════════════════════
    #  CARGAR / GENERAR / REPRODUCIR
    # ══════════════════════════════════════════════════════
    def _load_wav(self):
        path = filedialog.askopenfilename(
            title="Selecciona un archivo WAV",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not path:
            return
        try:
            sr, data = wav.read(path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            s = data.astype(np.float32)
            s /= np.abs(s).max() + 1e-9
            self.samples, self.sample_rate = s, sr
            name = path.split("/")[-1].split("\\")[-1]
            self.status_var.set(f"{name}  •  {sr} Hz  •  {len(s)/sr:.1f}s")
            self._switch_view("file")
            self._analyze_and_plot()
        except Exception as e:
            messagebox.showerror("Error al cargar", str(e))

    def _open_generate_dialog(self):
        dlg = tk.Toplevel(self)
        dlg.title("Generar secuencia DTMF")
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text="Generar secuencia DTMF", font=("Segoe UI", 12, "bold"),
                 bg=BG, fg=TEXT).pack(pady=(16, 4))
        tk.Label(dlg, text="Escribe o usa el teclado de abajo",
                 font=("Segoe UI", 8), bg=BG, fg=TEXT_DIM).pack()

        entry = tk.Entry(dlg, font=("Consolas", 18, "bold"), width=18,
                         bg=CARD, fg=GREEN, insertbackground=GREEN,
                         relief="flat", bd=8, justify="center")
        entry.insert(0, "1234567890")
        entry.pack(padx=20, pady=12)

        # Teclado visual en el diálogo
        kb = tk.Frame(dlg, bg=BG)
        kb.pack(pady=4)
        for i, row_keys in enumerate([['1','2','3','A'], ['4','5','6','B'],
                                       ['7','8','9','C'], ['*','0','#','D']]):
            for j, key in enumerate(row_keys):
                color = DIGIT_COLORS.get(key, ACCENT)
                btn = tk.Frame(kb, bg=CARD2, width=54, height=44, cursor="hand2",
                               highlightbackground=color, highlightthickness=1)
                btn.grid(row=i, column=j, padx=3, pady=3)
                btn.pack_propagate(False)
                lbl = tk.Label(btn, text=key, font=("Segoe UI", 14, "bold"),
                               bg=CARD2, fg=color)
                lbl.pack(expand=True)
                for w in (btn, lbl):
                    w.bind("<Button-1>", lambda e, k=key: entry.insert(tk.END, k))
                    w.bind("<Enter>", lambda e, b=btn, c=color: b.config(bg=c))
                    w.bind("<Leave>", lambda e, b=btn: b.config(bg=CARD2))

        # Parámetros
        pf = tk.Frame(dlg, bg=BG, pady=8)
        pf.pack()
        for col, label, default in [(0,"Duración tono (ms)","300"),
                                    (1,"Silencio entre tonos (ms)","100")]:
            tk.Label(pf, text=label, font=("Segoe UI", 8), bg=BG,
                     fg=TEXT_DIM).grid(row=0, column=col, padx=12)
            sp = tk.Spinbox(pf, from_=50, to=1000, increment=50, width=6,
                            bg=CARD, fg=TEXT, font=("Segoe UI", 10),
                            relief="flat", buttonbackground=CARD2)
            sp.delete(0, tk.END); sp.insert(0, default)
            sp.grid(row=1, column=col, padx=12, pady=4)
            if col == 0: spin_tone = sp
            else:        spin_sil  = sp

        def do_gen():
            seq = entry.get().strip().upper()
            if not seq:
                messagebox.showwarning("Vacío", "Ingresa al menos un dígito.", parent=dlg)
                return
            valid = set(DTMF_TABLE.values()) | {' '}
            bad = [c for c in seq if c not in valid]
            if bad:
                messagebox.showerror("Error", f"Caracteres no válidos: {bad}", parent=dlg)
                return
            s = generate_dtmf_sequence(seq, tone_ms=int(spin_tone.get()),
                                       silence_ms=int(spin_sil.get()),
                                       sample_rate=MIC_SR)
            self.samples, self.sample_rate = s, MIC_SR
            self.status_var.set(f"Generado: '{seq}'  •  {len(s)/MIC_SR:.1f} s")
            dlg.destroy()
            self._switch_view("file")
            self._analyze_and_plot()

        tk.Button(dlg, text="✔  Generar y analizar",
                  font=("Segoe UI", 10, "bold"), bg=ACCENT, fg=BG,
                  relief="flat", padx=20, pady=10, cursor="hand2",
                  command=do_gen).pack(pady=16)

    def _play_audio(self):
        if self.samples is None:
            messagebox.showinfo("Sin audio", "Primero carga o genera audio.")
            return
        sd.stop()
        def _p():
            sd.play(self.samples, self.sample_rate); sd.wait()
        threading.Thread(target=_p, daemon=True).start()

    def _stop_audio(self):
        sd.stop()

    def _on_close(self):
        self._stop_mic()
        self.destroy()


if __name__ == "__main__":
    app = DTMFApp()
    app.mainloop()

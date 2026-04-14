# information_theory/gui_experiment.py

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

# Localiza o diretório principal do projeto (uma pasta acima deste arquivo)
ROOT_DIR = Path(__file__).resolve().parents[1]  # Assumindo estrutura: projeto_completo/information_theory/gui_experiment.py
sys.path.insert(0, str(ROOT_DIR))

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os # Para criar diretórios

import numpy as np
import pandas as pd # Para exportar dados em CSV de forma mais robusta
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # Para a elipse

# NOVO: Importar pywt para transformada wavelet
import pywt

from config import DATABASE, EMBEDDING_DIM, RANDOM_STATE
from data.dataset_it import load_dataset_it
from signal_processing.text_signal import text_to_signal
from information_theory.experiment_cache import save_experiment, load_experiment
from information_theory.fisher_shannon_experiment import compute_hs_f
from information_theory.bandt_pompe_complexity import bandt_pompe_complexity

from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# Garante que o diretório de resultados exista
RESULTS_DIR = Path("results") / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Tooltip Class ---
class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 27
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def create_tooltip(widget, text):
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.showtip(text)
    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

# --- Main GUI Class ---
class ExperimentGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Experimento de Complexidade-Entropia / Fisher-Shannon")

        np.random.seed(RANDOM_STATE)

        # --------------------------------------------------
        # Carrega os códigos de idioma diretamente de load_dataset_it
        # --------------------------------------------------
        try:
            # load_dataset_it retorna: texts, labels, lang_codes, raw_labels, lang_names
            # Pegamos apenas os lang_codes para os checkboxes
            _, _, self.all_lang_codes, _, _ = load_dataset_it()
            self.all_lang_codes.sort() # Ordena os códigos para exibição

        except Exception as e: # Captura qualquer erro ao carregar o dataset
            messagebox.showerror("Erro ao carregar dados", f"Não foi possível carregar os idiomas do dataset: {e}")
            self.all_lang_codes = []
        # --------------------------------------------------

        # Variáveis para os checkboxes de idioma
        self.lang_vars = {} # Armazena {código_do_idioma: tk.BooleanVar}
        for lang_code in self.all_lang_codes:
            self.lang_vars[lang_code] = tk.BooleanVar(value=False)

        # Variáveis para dimensão e atraso
        self.dim_var = tk.IntVar(value=EMBEDDING_DIM) # Valor padrão do config
        self.tau_var = tk.IntVar(value=1) # Atraso padrão 1
        self.normalize_var = tk.BooleanVar(value=True) # Padrão: normalizado

        # Armazena os valores originais de dim e tau para restaurar após a comparação
        self._original_dim = self.dim_var.get()
        self._original_tau = self.tau_var.get()

        # Radio button: 'bp', 'fs' (Histograma removido)
        self.plot_type = tk.StringVar(value="bp") 

        # NOVO: Variável para o tipo de sinal (Original ou Wavelet)
        self.signal_type_var = tk.StringVar(value="original") # Padrão: original
        self.wavelet_level_var = tk.IntVar(value=3) # Nível de detalhe padrão para wavelet (D3)

        self.fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self.ax.set_aspect('equal', adjustable='box')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self._build_layout()

        # Dados do experimento carregado
        self.current_space = None   # 'bp', 'fs'
        self.current_stats_multi = {} # Dict para armazenar stats de múltiplos idiomas (usando CÓDIGO do idioma como chave)
        self.current_data_points_multi = {} # Dict para armazenar dados de múltiplos idiomas (usando CÓDIGO do idioma como chave)

        # Ciclo de cores para os gráficos
        self.colors = plt.cm.get_cmap('tab10', max(1, len(self.all_lang_codes))) # Garante pelo menos 1 cor
        self.color_map = {code: self.colors(i) for i, code in enumerate(self.all_lang_codes)}

    # ------------------------------------------------------------------
    # Parâmetros adaptativos para textos curtos
    # ------------------------------------------------------------------
    def _get_adaptive_params(self, text_length, ref_dim, ref_tau):
        """
        Define m (dim) e τ (tau) adaptativos conforme o tamanho do texto.
        - Para textos muito curtos, reduzimos m para garantir amostragem suficiente.
        - ref_dim/ref_tau são os usados no experimento do idioma (para textos longos).
        """
        # Faixas de exemplo – você pode ajustar depois com base nos seus dados
        if text_length < 200:
            m = 3
            tau = 1
        elif text_length < 500:
            m = 4
            tau = 1
        else:
            # Para textos maiores, usamos os mesmos parâmetros do experimento
            m = ref_dim
            tau = ref_tau

        return m, tau

    def _get_adaptive_threshold(self, text_length):
        """
        Limiar de pertencimento adaptativo.
        Para textos curtos, toleramos uma distância um pouco maior ao centroide.
        """
        base = 2.0  # valor atual
        if text_length < 200:
            return base + 0.7
        elif text_length < 500:
            return base + 0.4
        else:
            return base

    # ------------------------------------------------------------------
    # NOVO: Função para gerar sinal wavelet
    # ------------------------------------------------------------------
    def _get_wavelet_signal(self, original_signal, wavelet_name="db4", level=5, detail_level_to_use=3):
        """
        Aplica a transformada wavelet discreta e retorna os coeficientes de detalhe de um nível específico.
        wavelet_name: Família da wavelet (ex: "db4")
        level: Número total de níveis de decomposição.
        detail_level_to_use: O nível de detalhe (D1 a D5) cujos coeficientes serão usados como sinal.
        """
        try:
            wavelet = pywt.Wavelet(wavelet_name)
        except ValueError:
            raise ValueError(f"Wavelet '{wavelet_name}' inválida.")

        # Garante que o sinal seja numpy array e float para pywt
        signal_float = np.array(original_signal, dtype=float)

        # Verifica se o sinal é longo o suficiente para a decomposição
        # pywt.dwt_max_level retorna o nível máximo possível para um dado comprimento de sinal e wavelet
        max_possible_level = pywt.dwt_max_level(len(signal_float), wavelet)
        if level > max_possible_level:
            # Reduz o nível de decomposição se o sinal for muito curto
            level = max_possible_level
            if level == 0: # Se nem nível 1 for possível
                messagebox.showwarning("Aviso Wavelet", "Sinal muito curto para qualquer decomposição wavelet. Usando sinal original.")
                return original_signal # Retorna o sinal original como fallback

        # Realiza a decomposição wavelet
        coeffs = pywt.wavedec(signal_float, wavelet, level=level)

        # coeffs[0] é o coeficiente de aproximação (cA_level)
        # coeffs[1] é o coeficiente de detalhe D_level
        # ...
        # coeffs[level] é o coeficiente de detalhe D1

        # Queremos o coeficiente de detalhe do nível especificado (D_detail_level_to_use)
        # A ordem em coeffs é [cA_N, cD_N, cD_N-1, ..., cD_1]
        # Então, cD_detail_level_to_use está em coeffs[level - detail_level_to_use + 1]

        # Ex: se level=5 e detail_level_to_use=3 (D3)
        # coeffs = [cA5, cD5, cD4, cD3, cD2, cD1]
        # cD3 está em coeffs[5 - 3 + 1] = coeffs[3]

        # Verifica se o nível de detalhe solicitado é válido
        if not (1 <= detail_level_to_use <= level):
            messagebox.showwarning("Aviso Wavelet", f"Nível de detalhe {detail_level_to_use} inválido para {level} níveis. Usando D1.")
            detail_level_to_use = 1 # Fallback para D1

        # Pega os coeficientes de detalhe do nível desejado
        wavelet_signal = coeffs[level - detail_level_to_use + 1]

        # Opcional: Normalizar os coeficientes para evitar valores muito grandes/pequenos
        # Isso pode ajudar na estabilidade dos cálculos de Hs/C/F
        if len(wavelet_signal) > 0:
            wavelet_signal = (wavelet_signal - np.mean(wavelet_signal)) / (np.std(wavelet_signal) + 1e-10)

        return wavelet_signal

        # ------------------------------------------------------------------
    # Avaliação de separabilidade (sem ML treinado)
    # ------------------------------------------------------------------
    def evaluate_current_space(self):
        """
        Avalia quão bem os idiomas se separam no espaço atual (bp ou fs),
        usando apenas as métricas de Teoria da Informação.

        Métricas:
          - Índice de Silhueta (quanto maior, melhor, típico 0.0–1.0)
          - Razão intra/inter-distância (quanto menor, melhor)
          - Acurácia por centróide (classificação por idioma mais próximo)
        """
        if self.current_space not in ("bp", "fs"):
            messagebox.showwarning(
                "Aviso",
                "Gere primeiro um gráfico Bandt-Pompe ou Fisher-Shannon para avaliar a separabilidade."
            )
            return

        if not self.current_data_points_multi:
            messagebox.showwarning(
                "Aviso",
                "Nenhum dado disponível para avaliação. Gere um gráfico primeiro."
            )
            return

        # Monta matriz de pontos X e rótulos numéricos y
        all_points = []
        all_labels = []
        lang_list = []

        for idx, (lang, data) in enumerate(self.current_data_points_multi.items()):
            if data.get("type") != self.current_space:
                continue
            hs = data["hs"]
            yvals = data["y"]
            if len(hs) == 0:
                continue
            pts = np.column_stack((hs, yvals))  # (n_textos, 2)
            all_points.append(pts)
            all_labels.append(np.full(len(pts), idx))
            lang_list.append(lang)

        if not all_points:
            messagebox.showwarning(
                "Aviso",
                "Nenhum dado válido no espaço atual para avaliação."
            )
            return

        X = np.vstack(all_points)          # todos os pontos (N_total, 2)
        y = np.concatenate(all_labels)     # rótulos numéricos (N_total,)

        if len(np.unique(y)) < 2:
            messagebox.showwarning(
                "Aviso",
                "É necessário pelo menos 2 idiomas para calcular a separabilidade."
            )
            return

        # 1. Índice de Silhueta
        try:
            sil_score = silhouette_score(X, y, metric="euclidean")
        except Exception as e:
            sil_score = np.nan
            print(f"Aviso: não foi possível calcular Silhueta: {e}")

        # 2. Razão intra/inter-distância
        intra_dists = []
        inter_dists = []

        # Calcula centróides por idioma no plano atual
        centroids = {}
        for idx, lang in enumerate(lang_list):
            mask = (y == idx)
            pts_lang = X[mask]
            if len(pts_lang) == 0:
                continue
            centroids[lang] = pts_lang.mean(axis=0)

        # Se não tiver pelo menos 2 centróides válidos, não faz sentido continuar
        if len(centroids) < 2:
            messagebox.showwarning(
                "Aviso",
                "Não há idiomas suficientes com dados válidos para calcular distâncias intra/inter."
            )
            return

        for idx, lang in enumerate(lang_list):
            mask = (y == idx)
            pts_lang = X[mask]
            if len(pts_lang) < 2:
                # não dá pra calcular distância intra com 1 ponto só
                continue

            # Distância média intra-idioma
            dist_matrix = cdist(pts_lang, pts_lang, metric="euclidean")
            # pega apenas parte superior da matriz, sem diagonal
            triu_indices = np.triu_indices(len(pts_lang), k=1)
            intra = dist_matrix[triu_indices].mean()
            intra_dists.append(intra)

            # Distância média do centróide desse idioma para os centróides dos outros
            this_centroid = centroids[lang]
            other_centroids = [
                c for l, c in centroids.items() if l != lang
            ]
            dists_to_others = [np.linalg.norm(this_centroid - oc) for oc in other_centroids]
            inter_dists.append(np.mean(dists_to_others))

        if intra_dists and inter_dists:
            R = float(np.mean(intra_dists) / (np.mean(inter_dists) + 1e-10))
        else:
            R = np.nan

        # 3. Acurácia por centróide mais próximo
        correct = 0
        total = 0
        for point, label_idx in zip(X, y):
            true_lang = lang_list[label_idx]
            # distâncias para todos os centróides calculados a partir dos dados
            dists = {lang: np.linalg.norm(point - c) for lang, c in centroids.items()}
            predicted_lang = min(dists, key=dists.get)
            if predicted_lang == true_lang:
                correct += 1
            total += 1

        centroid_accuracy = correct / total if total > 0 else np.nan

        # Mostra os resultados em uma janela
        space_name = "Bandt-Pompe (CH)" if self.current_space == "bp" else "Fisher-Shannon (FS)"
        msg = (
            f"Avaliação de separabilidade no espaço {space_name}:\n\n"
            f"- Índice de Silhueta médio: {sil_score:.3f}\n"
            f"- Razão intra/inter-distância (R): {R:.3f}\n"
            f"  (quanto menor R, melhor separação entre idiomas)\n"
            f"- Acurácia por centróide mais próximo: {centroid_accuracy*100:.2f}%\n\n"
            f"Interpretação rápida:\n"
            f"• Silhueta alta (≈0.5 ou mais) indica clusters bem separados.\n"
            f"• R baixo (< 0.7) indica que textos de um mesmo idioma estão, em média,\n"
            f"  mais próximos entre si do que de outros idiomas.\n"
            f"• Acurácia por centróide alta sugere que um classificador simples baseado\n"
            f"  apenas em distância no plano de Teoria da Informação já seria eficaz."
        )
        messagebox.showinfo("Avaliação de separabilidade", msg)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        # Frame superior para controles de seleção e geração
        top_frame = ttk.Frame(self.master)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Frame para os checkboxes de idioma
        lang_selection_frame = ttk.LabelFrame(top_frame, text="Idiomas")
        lang_selection_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)

        # Adiciona um scrollbar se houver muitos idiomas
        canvas_lang = tk.Canvas(lang_selection_frame, height=100, width=100) # Largura ajustada para códigos
        scrollbar_lang = ttk.Scrollbar(lang_selection_frame, orient="vertical", command=canvas_lang.yview)
        scrollable_frame_lang = ttk.Frame(canvas_lang)

        scrollable_frame_lang.bind(
            "<Configure>",
            lambda e: canvas_lang.configure(
                scrollregion=canvas_lang.bbox("all")
            )
        )

        canvas_lang.create_window((0, 0), window=scrollable_frame_lang, anchor="nw")
        canvas_lang.configure(yscrollcommand=scrollbar_lang.set)

        for lang_code in self.all_lang_codes: # Itera sobre os códigos
            ttk.Checkbutton(scrollable_frame_lang, text=lang_code, variable=self.lang_vars[lang_code]).pack(anchor="w")

        canvas_lang.pack(side="left", fill="both", expand=True)
        scrollbar_lang.pack(side="right", fill="y")


        # Frame para parâmetros (Dimensão e Atraso)
        params_frame = ttk.LabelFrame(top_frame, text="Parâmetros")
        params_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.Y)

        ttk.Label(params_frame, text="Dimensão (m):").pack(anchor="w", padx=2, pady=2)
        ttk.Entry(params_frame, textvariable=self.dim_var, width=5).pack(anchor="w", padx=2, pady=2)

        ttk.Label(params_frame, text="Atraso (τ):").pack(anchor="w", padx=2, pady=2)
        ttk.Entry(params_frame, textvariable=self.tau_var, width=5).pack(anchor="w", padx=2, pady=2)

        # NOVO: Frame para opções de normalização
        normalization_frame = ttk.LabelFrame(top_frame, text="Normalização")
        normalization_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.Y)
        ttk.Checkbutton(normalization_frame, text="Normalizar Hs e C/F", variable=self.normalize_var).pack(anchor="w", padx=2)

        # NOVO: Frame para seleção do tipo de sinal
        signal_type_frame = ttk.LabelFrame(top_frame, text="Tipo de Sinal")
        signal_type_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.Y)

        ttk.Radiobutton(signal_type_frame, text="Original (Codepoints)", variable=self.signal_type_var, value="original", command=self._toggle_wavelet_options).pack(anchor="w", padx=2)
        ttk.Radiobutton(signal_type_frame, text="Wavelet (db4, 5 níveis)", variable=self.signal_type_var, value="wavelet", command=self._toggle_wavelet_options).pack(anchor="w", padx=2)

        # NOVO: Opções de nível wavelet (inicialmente desabilitadas)
        self.wavelet_level_label = ttk.Label(signal_type_frame, text="Nível Detalhe (D1-D5):")
        self.wavelet_level_label.pack(anchor="w", padx=2, pady=2)
        self.wavelet_level_entry = ttk.Entry(signal_type_frame, textvariable=self.wavelet_level_var, width=5)
        self.wavelet_level_entry.pack(anchor="w", padx=2, pady=2)

        # Desabilita as opções de wavelet por padrão
        self.wavelet_level_label.config(state=tk.DISABLED)
        self.wavelet_level_entry.config(state=tk.DISABLED)


        # Radio buttons para seleção do tipo de gráfico (Histograma removido)
        radio_frame = ttk.LabelFrame(top_frame, text="Tipo de Gráfico")
        radio_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.Y)

        ttk.Radiobutton(radio_frame, text="Bandt-Pompe", variable=self.plot_type, value="bp").pack(anchor="w", padx=2)
        ttk.Radiobutton(radio_frame, text="Fisher-Shannon", variable=self.plot_type, value="fs").pack(anchor="w", padx=2)

        # Frame para o botão Gerar e Loading
        generate_control_frame = ttk.Frame(top_frame)
        generate_control_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.Y)

        self.generate_button = ttk.Button(generate_control_frame, text="Gerar gráfico", command=self.on_generate)
        self.generate_button.pack(pady=5)

        self.list_outliers_button = ttk.Button(generate_control_frame, text="Listar Outliers", command=self.list_outliers)
        self.list_outliers_button.pack(fill=tk.X, pady=2)
        create_tooltip(self.list_outliers_button, "Identifica e lista textos que são outliers no gráfico atual.")

        self.loading_label = ttk.Label(generate_control_frame, text="", foreground="blue")
        self.loading_label.pack(pady=5)

        # Botão "Ver cálculos"
        self.view_calculations_button = ttk.Button(generate_control_frame, text="Ver cálculos", command=self.show_calculations)
        self.view_calculations_button.pack(pady=5)


        # Canvas matplotlib
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Frame para botões de exportação
        export_frame = ttk.Frame(self.master)
        export_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(export_frame, text="Exportar:").pack(side=tk.LEFT)
        self.export_graph_button = ttk.Button(export_frame, text="Gráfico (PNG)", command=self.export_graph)
        self.export_graph_button.pack(side=tk.LEFT, padx=5)
        self.export_data_button = ttk.Button(export_frame, text="Dados (CSV)", command=self.export_data)
        self.export_data_button.pack(side=tk.LEFT, padx=5)

        # Botão para avaliar separabilidade no plano atual (bp ou fs)
        self.evaluate_button = ttk.Button(
            export_frame,
            text="Avaliar separabilidade (TI)",
            command=self.evaluate_current_space
        )
        self.evaluate_button.pack(side=tk.LEFT, padx=5)


        # Área de comparação (texto novo) — só para BP e FS
        bottom_frame = ttk.LabelFrame(self.master, text="Comparar texto com idioma de referência")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.text_input = tk.Text(bottom_frame, height=4)
        self.text_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        control_frame = ttk.LabelFrame(bottom_frame, text="Controles") # Mudei para LabelFrame para organizar
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        self.compare_bp_button = ttk.Button(control_frame, text="Comparar (BP)", command=self.on_compare_bp)
        self.compare_bp_button.pack(fill=tk.X, pady=2)
        create_tooltip(self.compare_bp_button, "Compara o texto no Plano de Complexidade-Entropia (Bandt-Pompe) com o idioma de referência.")

        self.compare_fs_button = ttk.Button(control_frame, text="Comparar (FS)", command=self.on_compare_fs)
        self.compare_fs_button.pack(fill=tk.X, pady=2)
        create_tooltip(self.compare_fs_button, "Compara o texto no Plano de Fisher-Shannon com o idioma de referência.")

        self.result_label = ttk.Label(control_frame, text="", foreground="blue")
        self.result_label.pack(fill=tk.X, pady=5)

        # NOVO: Desabilita os botões de comparação por padrão
        self.compare_bp_button.config(state=tk.DISABLED)
        self.compare_fs_button.config(state=tk.DISABLED)

        # Botão para limpar o texto
        self.clear_text_button = ttk.Button(control_frame, text="Limpar Texto", command=self.clear_text_input)
        self.clear_text_button.pack(fill=tk.X, pady=5)
        create_tooltip(self.clear_text_button, "Limpa o campo de texto para nova entrada.")

    # ------------------------------------------------------------------
    # NOVO: Habilitar/Desabilitar opções de wavelet
    # ------------------------------------------------------------------
    def _toggle_wavelet_options(self):
        if self.signal_type_var.get() == "wavelet":
            self.wavelet_level_label.config(state=tk.NORMAL)
            self.wavelet_level_entry.config(state=tk.NORMAL)
        else:
            self.wavelet_level_label.config(state=tk.DISABLED)
            self.wavelet_level_entry.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Geração dos gráficos
    # ------------------------------------------------------------------
    def on_generate(self):
        selected_lang_codes = [code for code, var in self.lang_vars.items() if var.get()]
        if not selected_lang_codes:
            messagebox.showerror("Erro", "Nenhum idioma selecionado.")
            return

        # Pega os parâmetros de normalização, dimensão e atraso
        try:
            normalize_data = self.normalize_var.get()
            dim = self.dim_var.get()
            tau = self.tau_var.get()
            if dim <= 1 or tau <= 0:
                messagebox.showerror("Erro de Parâmetros", "Dimensão (m) deve ser > 1 e Atraso (τ) deve ser > 0.")
                return
        except tk.TclError:
            messagebox.showerror("Erro de Parâmetros", "Dimensão (m) e Atraso (τ) devem ser números inteiros.")
            return

        # Pega o tipo de sinal e nível wavelet
        signal_type = self.signal_type_var.get()
        wavelet_level = self.wavelet_level_var.get() if signal_type == "wavelet" else None
        if signal_type == "wavelet" and not (1 <= wavelet_level <= 5):
            messagebox.showerror("Erro de Parâmetros", "Nível de detalhe wavelet deve ser entre 1 e 5.")
            return

        # Salva os valores atuais de dim e tau antes de qualquer alteração
        self._original_dim = dim
        self._original_tau = tau

        self.generate_button.config(state=tk.DISABLED)
        self.loading_label.config(text="Calculando...", foreground="blue")
        self.master.update_idletasks() # Força a atualização da GUI

        try:
            self.fig.clf() # Limpa a figura atual
            self.ax = self.fig.add_subplot(111) # Adiciona um novo subplot
            self.ax.set_aspect('equal', adjustable='box')

            plot_type = self.plot_type.get()
            self.current_space = plot_type
            self.current_stats_multi = {}
            self.current_data_points_multi = {}

            all_texts_data = {} # Para carregar textos uma vez
            # Carrega os textos usando os códigos de idioma
            texts_all_db, labels_all_db, lang_codes_all_db, _, _ = load_dataset_it()

            for lang_code in selected_lang_codes:
                if lang_code not in lang_codes_all_db:
                    messagebox.showwarning("Aviso", f"Idioma '{lang_code}' não encontrado nos dados carregados.")
                    continue
                idx_lang = lang_codes_all_db.index(lang_code)
                texts_lang = [texts_all_db[i] for i in range(len(texts_all_db)) if labels_all_db[i] == idx_lang]
                all_texts_data[lang_code] = texts_lang

            if plot_type == "bp":
                self._plot_bp_space(all_texts_data, selected_lang_codes, dim, tau, signal_type, wavelet_level, normalize_data)
            elif plot_type == "fs":
                self._plot_fs_space(all_texts_data, selected_lang_codes, dim, tau, signal_type, wavelet_level, normalize_data)

            self.canvas.draw()
            self.result_label.config(text="")

            # NOVO: Habilita/Desabilita botões de comparação com base no gráfico gerado
            plot_type = self.plot_type.get() # Pega o tipo de gráfico que acabou de ser gerado
            if plot_type == "bp":
                self.compare_bp_button.config(state=tk.NORMAL)
                self.compare_fs_button.config(state=tk.DISABLED)
            elif plot_type == "fs":
                self.compare_bp_button.config(state=tk.DISABLED)
                self.compare_fs_button.config(state=tk.NORMAL)
            else: # Para "hist" ou qualquer outro caso, desabilita ambos
                self.compare_bp_button.config(state=tk.DISABLED)
                self.compare_fs_button.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Erro de processamento", f"Ocorreu um erro: {e}")
            print(f"Erro detalhado: {e}") # Para debug
        finally:
            self.generate_button.config(state=tk.NORMAL)
            self.loading_label.config(text="")

    def _plot_bp_parabolas(self):
        """
        Plota as curvas teóricas (parábolas) do plano Bandt-Pompe.
        - Curva inferior (Complexidade Mínima)
        - Curva superior (Complexidade Máxima)
        """
        # Curva Superior (Complexidade Máxima)
        hs_max_curve = np.linspace(0.001, 0.999, 100) # Evita log(0)
        c_max_curve = -2 * (hs_max_curve * np.log2(hs_max_curve) + (1 - hs_max_curve) * np.log2(1 - hs_max_curve))
        max_val_c_max_curve = np.max(c_max_curve)
        c_max_curve_normalized = c_max_curve / max_val_c_max_curve * 0.5 # Ajusta o pico para 0.5

        self.ax.plot(hs_max_curve, c_max_curve_normalized, color='black', linestyle='-', linewidth=1, zorder=0)

        # NOVO: Curva Inferior (Complexidade Mínima)
        # Esta curva representa a complexidade mínima para um dado Hs,
        # geralmente associada a sistemas com alguma estrutura, mas não totalmente aleatórios.
        # Uma aproximação comum é uma parábola invertida ou uma função que começa e termina em 0,
        # com um pico em Hs=0.5, mas em um valor de C muito menor que a curva superior.
        # Vamos usar uma parábola simples que se abre para baixo, com pico em Hs=0.5 e C=0.05 (exemplo).
        hs_min_curve = np.linspace(0, 1, 100)
        # A forma (x - 0.5)^2 gera uma parábola com vértice em x=0.5.
        # Multiplicamos por um fator negativo para que ela se abra para baixo.
        # Adicionamos um offset para que o pico esteja em um C > 0.
        # Ajuste os coeficientes (0.2 e 0.05) conforme a representação desejada.
        c_min_curve = -0.2 * (hs_min_curve - 0.5)**2 + 0.05
        # Garante que C não seja negativo
        c_min_curve[c_min_curve < 0] = 0

        self.ax.plot(hs_min_curve, c_min_curve, color='black', linestyle='-', linewidth=1, zorder=0)

        # Adiciona os rótulos "z=2" e "z=5/2" se desejar, como na imagem.
        # Estes são rótulos para pontos específicos na curva, não para a curva inteira.
        # Para replicar exatamente a imagem, precisaríamos saber a função exata que gera esses rótulos.
        # Por enquanto, vamos apenas plotar as curvas.

    def _plot_bp_space(self, all_texts_data, selected_lang_codes, dim, tau, signal_type, wavelet_level, normalize_data: bool):

        suffix_norm = ''
        if normalize_data:
            self._plot_bp_parabolas()
            suffix_norm = 'normalizada'

        all_hs_values = []
        all_y_values = []
        all_handles = []
        all_labels = []

        for lang_code in selected_lang_codes:
            # NOVO: Adiciona signal_type e wavelet_level ao cache_key
            cache_key_suffix = f"_{signal_type}"
            if signal_type == "wavelet":
                cache_key_suffix += f"_w{wavelet_level}"
            cache_key_suffix += f"_{suffix_norm}"

            cached = load_experiment(lang_code, "bp", dim, tau, cache_key_suffix=cache_key_suffix)
            if cached is not None and cached["hs"] is not None:
                hs = cached["hs"]
                C  = cached["y"]
                centroid = np.array([cached["centroid_hs"], cached["centroid_y"]])
                std_hs   = cached["std_hs"]
                std_C    = cached["std_y"]
                print(f"{len(hs)} pontos plotados para {lang_code}")
            else:
                hs_list, C_list = [], []
                text_idx_list = []
                for t in all_texts_data[lang_code]:
                    original_sig = text_to_signal(t)

                    # NOVO: Escolhe o sinal a ser usado
                    if signal_type == "original":
                        sig = original_sig
                    else: # wavelet
                        sig = self._get_wavelet_signal(original_sig, level=5, detail_level_to_use=wavelet_level)
                        if sig is None or len(sig) == 0: # Fallback se wavelet falhar ou retornar vazio
                            continue # Pula este texto se o sinal wavelet for inválido

                    if len(sig) >= dim * tau:
                        Hs, C = bandt_pompe_complexity(sig, dim, tau, normalize_data)
                        hs_list.append(Hs)
                        C_list.append(C)

                if not hs_list:
                    messagebox.showwarning("Aviso", f"Nenhum texto longo o suficiente para Bandt-Pompe para o idioma '{lang_code}' com dim={dim}, tau={tau} e sinal '{signal_type}'.")
                    continue

                hs = np.array(hs_list)
                C  = np.array(C_list)

                centroid = np.array([hs.mean(), C.mean()])
                std_hs   = hs.std()
                std_C    = C.std()
                # NOVO: Salva no cache com sufixo
                save_experiment(lang_code, "bp", dim, tau, hs, C, cache_key_suffix=cache_key_suffix)

            self.current_stats_multi[lang_code] = { # Chave é o código
                "lang": lang_code, "space": "bp", "dim": dim, "tau": tau,
                "hs": hs, "y": C, "centroid": centroid, "std_hs": std_hs, "std_y": std_C,
                "signal_type": signal_type, "wavelet_level": wavelet_level
            }
            self.current_data_points_multi[lang_code] = {"type": "bp", "hs": hs, "y": C, "centroid": centroid, "std_hs": std_hs, "std_y": std_C, "texts": all_texts_data[lang_code]}

            # Plotar os pontos e o centroide para cada idioma
            color = self.color_map[lang_code] # Usa o código para pegar a cor
            scatter_points = self.ax.scatter(hs, C, c=color, s=15, alpha=0.6, label=f"{lang_code} (n={len(hs)})", zorder=3) # Zorder para pontos
            all_handles.append(scatter_points)
            all_labels.append(f"{lang_code} (n={len(hs)})")
            scatter_centroid = self.ax.scatter(centroid[0], centroid[1], c=color, s=100, marker="X", edgecolors='black', linewidths=0.8, label=f"Centroide {lang_code}", zorder=4) # Zorder para centroide
            all_handles.append(scatter_centroid)
            all_labels.append(f"Centroide {lang_code}")

            # Elipse de pertencimento
            threshold = 1.0
            ellipse_label = f"Região ±{threshold:.1f}σ {lang_code}" # Define o label da elipse
            ellipse = mpatches.Ellipse(
                xy=(centroid[0], centroid[1]),
                width=2 * threshold * std_hs,
                height=2 * threshold * std_C, # Ou std_F para FS
                edgecolor=color,
                facecolor=color,
                alpha=0.08,
                linestyle="--",
                linewidth=1.2,
                zorder=2,
            )
            self.ax.add_patch(ellipse)

            all_handles.append(ellipse)
            all_labels.append(ellipse_label)

            all_hs_values.extend(hs)
            all_y_values.extend(C)

        if not all_hs_values:
            messagebox.showwarning("Aviso", "Nenhum dado válido para plotar o Plano Bandt-Pompe.")
            return

        # Converte as listas para arrays numpy para facilitar o cálculo de min/max
        all_hs_values_np = np.array(all_hs_values)
        all_y_values_np = np.array(all_y_values)

        min_hs = all_hs_values_np.min()
        max_hs = all_hs_values_np.max()
        min_y = all_y_values_np.min()
        max_y = all_y_values_np.max()

        # Adiciona linhas verticais para min/max Hs
        self.ax.axvline(min_hs, color='gray', linestyle=':', linewidth=1, zorder=1)
        self.ax.axvline(max_hs, color='gray', linestyle=':', linewidth=1, zorder=1)

        # Adiciona linhas horizontais para min/max C
        self.ax.axhline(min_y, color='gray', linestyle=':', linewidth=1, zorder=1)
        self.ax.axhline(max_y, color='gray', linestyle=':', linewidth=1, zorder=1)

        title_suffix = f"Sinal: {signal_type.capitalize()}"
        if signal_type == "wavelet":
            title_suffix += f" (db4, Nível D{wavelet_level})"

        if normalize_data:
            self.ax.set_xlim(-0.02, 1.02)
            self.ax.set_ylim(-0.02, 1.02)
            self.ax.set_xlabel("Entropia de permutação normalizada $H_s$")
            self.ax.set_ylabel("Complexidade estatística $C$")
            title_suffix += " (Normalizado)"
        else:
            # Para dados não normalizados, os limites serão dinâmicos
            min_hs_plot = all_hs_values_np.min() * 0.9
            max_hs_plot = all_hs_values_np.max() * 1.1
            min_y_plot = all_y_values_np.min() * 0.9
            max_y_plot = all_y_values_np.max() * 1.1

            self.ax.set_xlim(min_hs_plot, max_hs_plot)
            self.ax.set_ylim(min_y_plot, max_y_plot)
            self.ax.set_xlabel("Entropia de permutação $H_s$")
            self.ax.set_ylabel("Complexidade estatística $C$")

        self.ax.set_title(f"Plano Complexidade–Entropia (Bandt-Pompe) — {', '.join(selected_lang_codes)}\nDimensão (m)={dim}, Atraso (τ)={tau}. {title_suffix}")
        self.ax.legend(handles=all_handles, labels=all_labels, fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        self.ax.grid(True, linestyle="--", alpha=0.4, zorder=1) # Zorder para grid

    def _plot_fs_space(self, all_texts_data, selected_lang_codes, dim, tau, signal_type, wavelet_level, normalize_data: bool):

        all_hs_values = []
        all_y_values = []
        all_handles = []
        all_labels = []

        for lang_code in selected_lang_codes:
            # NOVO: Adiciona signal_type e wavelet_level ao cache_key
            cache_key_suffix = f"_{signal_type}"
            if signal_type == "wavelet":
                cache_key_suffix += f"_w{wavelet_level}"

            cached = load_experiment(lang_code, "fs", dim, tau, cache_key_suffix=cache_key_suffix)
            if cached is not None and cached["hs"] is not None:
                hs = cached["hs"]
                F  = cached["y"]
                centroid = np.array([cached["centroid_hs"], cached["centroid_y"]])
                std_hs   = cached["std_hs"]
                std_F    = cached["std_y"]
            else:
                hs_list, F_list = [], []
                for t in all_texts_data[lang_code]:
                    original_sig = text_to_signal(t)

                    # NOVO: Escolhe o sinal a ser usado
                    if signal_type == "original":
                        sig = original_sig
                    else: # wavelet
                        sig = self._get_wavelet_signal(original_sig, level=5, detail_level_to_use=wavelet_level)
                        if sig is None or len(sig) == 0: # Fallback se wavelet falhar ou retornar vazio
                            continue # Pula este texto se o sinal wavelet for invál

                    if len(sig) >= dim * tau:
                        Hs, F = compute_hs_f(sig, dim, tau, normalize_data)
                        hs_list.append(Hs)
                        F_list.append(F)

                if not hs_list:
                    messagebox.showwarning("Aviso", f"Nenhum texto longo o suficiente para Fisher-Shannon para o idioma '{lang_code}' com dim={dim}, tau={tau} e sinal '{signal_type}'.")
                    continue

                hs = np.array(hs_list)
                F  = np.array(F_list)

                centroid = np.array([hs.mean(), F.mean()])
                std_hs   = hs.std()
                std_F    = F.std()
                # NOVO: Salva no cache com sufixo
                save_experiment(lang_code, "fs", dim, tau, hs, F, cache_key_suffix=cache_key_suffix)

            self.current_stats_multi[lang_code] = { # Chave é o código
                "lang": lang_code, "space": "fs", "dim": dim, "tau": tau,
                "hs": hs, "y": F, "centroid": centroid, "std_hs": std_hs, "std_y": std_F,
                "signal_type": signal_type, "wavelet_level": wavelet_level # NOVO: Salva tipo de sinal
            }
            self.current_data_points_multi[lang_code] = {"type": "fs", "hs": hs, "y": F, "centroid": centroid, "std_hs": std_hs, "std_y": std_F}

            # Plotar os pontos e o centroide para cada idioma
            color = self.color_map[lang_code] # Usa o código para pegar a cor
            scatter_points = self.ax.scatter(hs, F, c=color, s=15, alpha=0.6, label=f"{lang_code} (n={len(hs)})", zorder=3) # Zorder para pontos
            all_handles.append(scatter_points)
            all_labels.append(f"{lang_code} (n={len(hs)})")
            scatter_centroid = self.ax.scatter(centroid[0], centroid[1], c=color, s=100, marker="X", edgecolors='black', linewidths=0.8, label=f"Centroide {lang_code}", zorder=4) # Zorder para centroide
            all_handles.append(scatter_centroid)
            all_labels.append(f"Centroide {lang_code}")

            # Elipse de pertencimento
            threshold = 2.0
            ellipse_label = f"Região ±{threshold:.1f}σ {lang_code}" # Define o label da elipse
            ellipse = mpatches.Ellipse(
                xy=(centroid[0], centroid[1]),
                width=2 * threshold * std_hs,
                height=2 * threshold * std_F, # Ou std_F para FS
                edgecolor=color,
                facecolor=color,
                alpha=0.08,
                linestyle="--",
                linewidth=1.2,
                zorder=2, # Zorder para elipse
            )
            self.ax.add_patch(ellipse)

            all_handles.append(ellipse)
            all_labels.append(ellipse_label)

            all_hs_values.extend(hs)
            all_y_values.extend(F)

        if not all_hs_values:
            messagebox.showwarning("Aviso", "Nenhum dado válido para plotar o Plano Fisher-Shannon.")
            return

        # Converte as listas para arrays numpy para facilitar o cálculo de min/max
        all_hs_values_np = np.array(all_hs_values)
        all_y_values_np = np.array(all_y_values)

        min_hs = all_hs_values_np.min()
        max_hs = all_hs_values_np.max()
        min_y = all_y_values_np.min()
        max_y = all_y_values_np.max()

        # Adiciona linhas verticais para min/max Hs
        self.ax.axvline(min_hs, color='gray', linestyle=':', linewidth=1, zorder=1)
        self.ax.axvline(max_hs, color='gray', linestyle=':', linewidth=1, zorder=1)

        # Adiciona linhas horizontais para min/max F
        self.ax.axhline(min_y, color='gray', linestyle=':', linewidth=1, zorder=1)
        self.ax.axhline(max_y, color='gray', linestyle=':', linewidth=1, zorder=1)

        title_suffix = f"Sinal: {signal_type.capitalize()}"
        if signal_type == "wavelet":
            title_suffix += f" (db4, Nível D{wavelet_level})"

        if normalize_data:
            self.ax.set_xlim(-0.02, 1.02)
            self.ax.set_ylim(-0.02, 1.02)
            self.ax.set_xlabel("Entropia de permutação normalizada $H_s$")
            self.ax.set_ylabel("Informação de Fisher normalizada $F$")
            title_suffix += " (Normalizado)"
        else:
            # Para dados não normalizados, os limites serão dinâmicos
            min_hs_plot = all_hs_values_np.min() * 0.9
            max_hs_plot = all_hs_values_np.max() * 1.1
            min_y_plot = all_y_values_np.min() * 0.9
            max_y_plot = all_y_values_np.max() * 1.1

            self.ax.set_xlim(min_hs_plot, max_hs_plot)
            self.ax.set_ylim(min_y_plot, max_y_plot)
            self.ax.set_xlabel("Entropia de permutação $H_s$")
            self.ax.set_ylabel("Informação de Fisher $F$")

        self.ax.set_title(f"Plano Fisher–Shannon — {', '.join(selected_lang_codes)}\nDimensão (m)={dim}, Atraso (τ)={tau}. {title_suffix}")
        self.ax.legend(handles=all_handles, labels=all_labels, fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        self.ax.grid(True, linestyle="--", alpha=0.4, zorder=1) # Zorder para grid


    # ------------------------------------------------------------------
    # Comparação de texto novo
    # ------------------------------------------------------------------
    def _compare_text_in_space(self, space: str):
        # Salva os valores atuais de dim e tau da GUI antes de qualquer alteração
        # para que possam ser restaurados ao final da função.
        current_gui_dim = self.dim_var.get()
        current_gui_tau = self.tau_var.get()

        selected_lang_codes = [code for code, var in self.lang_vars.items() if var.get()]
        if not selected_lang_codes:
            messagebox.showerror("Erro", "Nenhum idioma selecionado para comparação.")
            return

        # O idioma de referência para comparação será o primeiro idioma selecionado
        ref_lang_code = selected_lang_codes[0]

        if ref_lang_code not in self.current_stats_multi or self.current_stats_multi[ref_lang_code]["space"] != space:
            messagebox.showerror("Erro", f"Gere primeiro o gráfico {space.upper()} para o idioma de referência '{ref_lang_code}'.")
            return

        texto = self.text_input.get("1.0", tk.END).strip()
        if not texto:
            messagebox.showwarning("Aviso", "Digite um texto para comparar.")
            # Restaura os valores originais na GUI
            self.dim_var.set(current_gui_dim)
            self.tau_var.set(current_gui_tau)
            return

        # Pega as estatísticas do idioma de referência
        ref_stats = self.current_stats_multi[ref_lang_code]
        ref_dim = ref_stats["dim"] # Dimensão usada para gerar o cluster do idioma
        ref_tau = ref_stats["tau"] # Atraso usado para gerar o cluster do idioma
        ref_signal_type = ref_stats["signal_type"] # NOVO: Tipo de sinal do cluster
        ref_wavelet_level = ref_stats["wavelet_level"] # NOVO: Nível wavelet do cluster
        centroid = ref_stats["centroid"]
        std_hs = ref_stats["std_hs"]
        std_y  = ref_stats["std_y"]

        original_sig = text_to_signal(texto)
        L_original = len(original_sig)
        if L_original == 0:
            messagebox.showwarning("Aviso", "Texto vazio após conversão em sinal.")
            # Restaura os valores originais na GUI
            self.dim_var.set(current_gui_dim)
            self.tau_var.set(current_gui_tau)
            return

        # NOVO: Gera o sinal para o texto novo usando o MESMO TIPO de sinal do cluster de referência
        if ref_signal_type == "original":
            sig = original_sig
            L = L_original
        else: # wavelet
            sig = self._get_wavelet_signal(original_sig, level=5, detail_level_to_use=ref_wavelet_level)
            if sig is None or len(sig) == 0:
                messagebox.showwarning("Aviso", "Sinal wavelet do texto novo é inválido ou muito curto.")
                self.dim_var.set(current_gui_dim)
                self.tau_var.set(current_gui_tau)
                return
            L = len(sig) # O comprimento do sinal wavelet pode ser diferente

        # NOVO: parâmetros adaptativos para o texto novo (baseado no comprimento do SINAL FINAL)
        dim, tau = self._get_adaptive_params(L, ref_dim, ref_tau)

        # ATUALIZA A GUI COM OS PARÂMETROS ADAPTATIVOS
        self.dim_var.set(dim)
        self.tau_var.set(tau)
        self.master.update_idletasks() # Força a atualização visual

        if L < dim * tau: # Verifica se o sinal é longo o suficiente para Bandt-Pompe com os parâmetros adaptados
            messagebox.showwarning(
                "Aviso",
                f"Sinal (tipo '{ref_signal_type}') muito curto para análise com m={dim}, τ={tau} "
                f"(mín. {dim * tau} pontos)."
            )
            # Restaura os valores originais na GUI
            self.dim_var.set(current_gui_dim)
            self.tau_var.set(current_gui_tau)
            return

        if space == "bp":
            Hs_new, Y_new = bandt_pompe_complexity(sig, dim, tau, normalize=ref_normalize_data)
            y_label = "C"
        else:  # 'fs'
            Hs_new, Y_new = compute_hs_f(sig, dim, tau, normalize=ref_normalize_data)
            y_label = "F"

        # Distância normalizada tipo Mahalanobis diagonal
        d_Hs = (Hs_new - centroid[0]) / (std_hs + 1e-10)
        d_Y  = (Y_new - centroid[1]) / (std_y  + 1e-10)
        dist = float(np.sqrt(d_Hs**2 + d_Y**2))

        # NOVO: limiar adaptativo (baseado no comprimento do SINAL FINAL)
        threshold = self._get_adaptive_threshold(L)
        belongs = dist <= threshold

        color = "green" if belongs else "red"
        label = "Texto novo"

        # Remove o ponto anterior se existir
        for artist in self.ax.collections:
            if artist.get_label() == label:
                artist.remove()
        for artist in self.ax.lines:
            if artist.get_label() == label:
                artist.remove()

        self.ax.scatter(
            Hs_new, Y_new,
            c=color,
            s=120,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            label=label,
            zorder=10 # Garante que o novo ponto esteja visível
        )
        # Atualiza a legenda para incluir o novo ponto
        self.ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        self.canvas.draw()

        msg = (
            f"Texto novo (m={dim}, τ={tau}, sinal='{ref_signal_type}'"
            f"{f', D{ref_wavelet_level}' if ref_signal_type == 'wavelet' else ''}): "
            f"Hs={Hs_new:.4f}, {y_label}={Y_new:.4f}, "
            f"dist={dist:.2f} (vs {ref_lang_code}, limiar={threshold:.2f}). "
            f"{'Pertence ao idioma.' if belongs else 'Não pertence ao idioma.'}"
        )
        self.result_label.config(text=msg, foreground=("green" if belongs else "red"))

        # RESTAURA OS VALORES ORIGINAIS DE DIM E TAU NA GUI APÓS A COMPARAÇÃO
        self.dim_var.set(current_gui_dim)
        self.tau_var.set(current_gui_tau)
        self.master.update_idletasks() # Força a atualização visual

    def list_outliers(self):
        self.list_outliers_button.config(state=tk.DISABLED)
        self.loading_label.config(text="Calculando...", foreground="blue")
        self.master.update_idletasks() # Força a atualização da GUI

        selected_lang_codes = [code for code, var in self.lang_vars.items() if var.get()]
        if not selected_lang_codes:
            messagebox.showwarning("Aviso", "Selecione pelo menos um idioma para listar os outliers.")
            return

        if not self.current_data_points_multi:
            messagebox.showwarning("Aviso", "Nenhum dado de experimento disponível. Gere um gráfico primeiro.")
            return

        # Pega o limiar do slider ou um valor padrão
        # Você pode adicionar um slider para o limiar de outlier ou usar um valor fixo
        # Por enquanto, vamos usar um valor fixo ou o threshold do último experimento
        # Se você tiver um self.threshold_var, use-o aqui.
        # Caso contrário, um valor padrão como 2.0 ou 2.5 é comum.
        outlier_threshold = 5 # Exemplo: 2.5 desvios-padrão

        outlier_results = {} # Armazenará {lang_code: [(dist, text), ...]}

        for lang_code in selected_lang_codes:
            if lang_code not in self.current_data_points_multi:
                continue

            data_for_lang = self.current_data_points_multi[lang_code]
            if not data_for_lang["hs"].size: # Verifica se há dados
                continue

            hs_values = data_for_lang["hs"]
            y_values = data_for_lang["y"]
            centroid = data_for_lang["centroid"]
            std_hs = data_for_lang["std_hs"]
            std_y = data_for_lang["std_y"]

            # Carrega os textos originais para este idioma
            # Você precisará ter acesso aos textos originais.
            # Uma forma é carregar o dataset novamente ou armazenar os textos
            # em self.current_data_points_multi junto com hs e y.
            # Por simplicidade, vamos carregar os textos aqui, mas idealmente
            # eles estariam já carregados e indexados.
            texts_all_db, labels_all_db, lang_codes_all_db, _, _ = load_dataset_it()
            idx_lang = lang_codes_all_db.index(lang_code)
            original_texts_for_lang = [texts_all_db[i] for i in range(len(texts_all_db)) if labels_all_db[i] == idx_lang]

            # Garante que o número de textos corresponda ao número de pontos
            if len(original_texts_for_lang) != len(hs_values):
                messagebox.showwarning("Aviso", f"Número de textos e pontos não coincide para {lang_code}. Pulando detecção de outliers.")
                continue

            distances = []
            for i in range(len(hs_values)):
                d_Hs = (hs_values[i] - centroid[0]) / (std_hs + 1e-10)
                d_Y = (y_values[i] - centroid[1]) / (std_y + 1e-10)
                dist = np.sqrt(d_Hs**2 + d_Y**2)
                distances.append(dist)

            outliers = []
            for i, dist in enumerate(distances):
                if dist > outlier_threshold:
                    outliers.append((dist, original_texts_for_lang[i]))

            if outliers:
                # Ordena os outliers pela distância (maior distância primeiro)
                outliers.sort(key=lambda x: x[0], reverse=True)
                outlier_results[lang_code] = outliers

        self.list_outliers_button.config(state=tk.NORMAL)
        self.loading_label.config(text="")

        if not outlier_results:
            messagebox.showinfo("Outliers", f"Nenhum outlier encontrado para os idiomas selecionados com limiar > {outlier_threshold:.1f}σ.")
            return

        self._show_outlier_popup(outlier_results, outlier_threshold)

    def _show_outlier_popup(self, outlier_results, threshold):
        """Exibe os resultados dos outliers em uma nova janela pop-up."""
        popup = tk.Toplevel(self.master)
        popup.title(f"Textos Outliers (Limiar > {threshold:.1f}σ)")
        popup.geometry("800x600")

        main_frame = ttk.Frame(popup)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Adiciona um Text widget com scrollbar
        text_area = tk.Text(main_frame, wrap="word", font=("TkDefaultFont", 10))
        text_area.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(main_frame, command=text_area.yview)
        scrollbar.pack(side="right", fill="y")
        text_area.config(yscrollcommand=scrollbar.set)

        content = []
        for lang_code, outliers in outlier_results.items():
            content.append(f"--- Idioma: {lang_code} ({len(outliers)} outliers) ---\n")
            for dist, text in outliers:
                content.append(f"  Distância: {dist:.2f}σ\n")
                content.append(f"  Texto: {text[:500]}...\n") # Limita o texto para não sobrecarregar
                content.append("-" * 70 + "\n")
            content.append("\n======================================\n\n")

        text_area.insert(tk.END, "".join(content))
        text_area.config(state="disabled") # Torna o texto somente leitura

        close_button = ttk.Button(popup, text="Fechar", command=popup.destroy)
        close_button.pack(pady=5)

    def on_compare_bp(self):
        self._compare_text_in_space("bp")

    def on_compare_fs(self):
        self._compare_text_in_space("fs")

    # ------------------------------------------------------------------
    # Limpar campo de texto
    # ------------------------------------------------------------------
    def clear_text_input(self):
        """Limpa o conteúdo do widget de entrada de texto e o resultado da comparação."""
        self.text_input.delete("1.0", tk.END)
        self.result_label.config(text="", foreground="blue")
        # Opcional: Limpar o ponto do texto novo do gráfico, se houver
        label_to_remove = "Texto novo"
        for artist in self.ax.collections:
            if artist.get_label() == label_to_remove:
                artist.remove()
        for artist in self.ax.lines:
            if artist.get_label() == label_to_remove:
                artist.remove()
        self.ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.) # Atualiza a legenda
        self.canvas.draw() # Redesenha o canvas para remover o ponto

    # ------------------------------------------------------------------
    # Funções de Exportação
    # ------------------------------------------------------------------
    def export_graph(self):
        selected_lang_codes = [code for code, var in self.lang_vars.items() if var.get()]
        if not selected_lang_codes:
            messagebox.showwarning("Aviso", "Nenhum idioma selecionado para exportar o gráfico.")
            return
        if not self.current_space:
            messagebox.showwarning("Aviso", "Nenhum gráfico gerado para exportar.")
            return

        # Pega os parâmetros de dimensão e atraso atuais da GUI para o nome do arquivo
        dim = self.dim_var.get()
        tau = self.tau_var.get()
        signal_type = self.signal_type_var.get()
        wavelet_level = self.wavelet_level_var.get() if signal_type == "wavelet" else None
        normalize_data = self.normalize_var.get()

        lang_str = "_".join(selected_lang_codes) # Usa códigos para o nome do arquivo

        # NOVO: Adiciona tipo de sinal e nível wavelet ao nome do arquivo
        file_name_parts = [self.current_space, lang_str, f"m{dim}", f"t{tau}", "norm" if normalize_data else ""]
        if signal_type == "wavelet":
            file_name_parts.append(f"w{wavelet_level}")
        file_name = "_".join(file_name_parts) + ".png"

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=file_name,
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            self.fig.savefig(file_path, dpi=300)
            messagebox.showinfo("Exportar Gráfico", f"Gráfico salvo em:\n{file_path}")

    def export_data(self):
        selected_lang_codes = [code for code, var in self.lang_vars.items() if var.get()]
        if not selected_lang_codes:
            messagebox.showwarning("Aviso", "Nenhum idioma selecionado para exportar os dados.")
            return
        if not self.current_data_points_multi:
            messagebox.showwarning("Aviso", "Nenhum dado disponível para exportar.")
            return

        # Pega os parâmetros de dimensão e atraso atuais da GUI para o nome do arquivo
        dim = self.dim_var.get()
        tau = self.tau_var.get()
        signal_type = self.signal_type_var.get()
        wavelet_level = self.wavelet_level_var.get() if signal_type == "wavelet" else None
        normalize_data = self.normalize_var.get()

        lang_str = "_".join(selected_lang_codes) # Usa códigos para o nome do arquivo

        # NOVO: Adiciona tipo de sinal e nível wavelet ao nome do arquivo
        file_name_parts = ["data", self.current_space, lang_str, f"m{dim}", f"t{tau}", "norm" if normalize_data else ""]
        if signal_type == "wavelet":
            file_name_parts.append(f"w{wavelet_level}")
        file_name = "_".join(file_name_parts) + ".csv"

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=file_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            try:
                data_type = self.current_space
                all_dfs = []
                all_stats_comments = []
                
                for lang_code in selected_lang_codes:
                    if lang_code not in self.current_data_points_multi:
                        continue # Pula idiomas que não geraram dados válidos

                    data_for_lang = self.current_data_points_multi[lang_code]

                    if data_type in ["bp", "fs"]:
                        hs_values = data_for_lang["hs"]
                        y_values = data_for_lang["y"]
                        header_y = "C" if data_type == "bp" else "F"
                        df = pd.DataFrame({f"Hs_{lang_code}": hs_values, f"{header_y}_{lang_code}": y_values})
                        all_dfs.append(df)

                        # Coleta estatísticas para comentários
                        all_stats_comments.append(f"# Stats for {lang_code}:")
                        all_stats_comments.append(f"#   Centroid Hs: {data_for_lang['centroid'][0]:.6f}")
                        all_stats_comments.append(f"#   Centroid {header_y}: {data_for_lang['centroid'][1]:.6f}")
                        all_stats_comments.append(f"#   Std Hs: {data_for_lang['std_hs']:.6f}")
                        all_stats_comments.append(f"#   Std {header_y}: {data_for_lang['std_y']:.6f}")
                        if normalize_data:
                            all_stats_comments.append("#   Dados normalizados")

                if not all_dfs:
                    messagebox.showwarning("Aviso", "Nenhum dado válido para exportar para os idiomas selecionados.")
                    return

                # Concatena todos os DataFrames (pode resultar em NaNs se os tamanhos forem diferentes)
                final_df = pd.concat(all_dfs, axis=1)

                with open(file_path, 'w') as f:
                    # Escreve os comentários das estatísticas primeiro
                    for comment in all_stats_comments:
                        f.write(comment + '\n')
                    # Escreve o DataFrame
                    final_df.to_csv(f, index=False, header=True)

                messagebox.showinfo("Exportar Dados", f"Dados salvos em:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Erro de Exportação", f"Não foi possível exportar os dados: {e}")
                print(f"Erro detalhado na exportação: {e}")

    # ------------------------------------------------------------------
    # Exibir Metodologia e Cálculos
    # ------------------------------------------------------------------
    def show_calculations(self):
        calc_window = tk.Toplevel(self.master)
        calc_window.title("Metodologia e Cálculos")
        calc_window.geometry("800x600")

        text_area = tk.Text(calc_window, wrap="word", font=("TkDefaultFont", 10))
        text_area.pack(expand=True, fill="both", padx=10, pady=10)

        # Conteúdo detalhado da metodologia e cálculos
        content = """
        Metodologia e Cálculos para Planos de Complexidade-Entropia e Fisher-Shannon

        Ambos os planos são ferramentas da Teoria da Informação para caracterizar a complexidade e a aleatoriedade de séries temporais. No nosso caso, a série temporal é composta pelos codepoints Unicode dos caracteres de um texto.

        1.  Série Temporal de Codepoints:
            - Para cada texto, é gerada uma série numérica onde cada elemento x[n] é o valor Unicode (ord(caractere)) do n-ésimo caractere.

        2.  Transformada Wavelet Discreta (DWT) - Daubechies db4, 5 níveis:
            - Quando o modo "Wavelet" é selecionado, o sinal original de codepoints passa por uma DWT usando a família Daubechies de ordem 4 (db4) com 5 níveis de decomposição.
            - A decomposição gera coeficientes de aproximação (A5) e coeficientes de detalhe (D5, D4, D3, D2, D1).
            - Para a análise, é utilizado um nível de detalhe específico (por exemplo, D3), cujos coeficientes formam a nova série temporal para os cálculos de complexidade.
            - Os coeficientes são normalizados (média zero, desvio padrão um) para estabilidade.

        3.  Plano de Complexidade-Entropia (Bandt-Pompe):
            - Este plano utiliza a Entropia de Permutação (Hs) e a Complexidade Estatística (C) para caracterizar a série.
            - Parâmetros:
                - Dimensão de Imersão (m): Número de pontos na janela de observação (ex: 3, 4, 5).
                - Atraso (τ): Intervalo entre os pontos na janela (geralmente 1).

            a.  Padrões Ordinais:
                - Para cada janela de 'm' pontos [x[n], x[n+τ], ..., x[n+(m-1)τ]], a ordem relativa dos valores é determinada. Por exemplo, se m=3 e a janela é [10, 20, 15], a ordem é (0, 2, 1) porque 10 é o menor (índice 0), 15 é o do meio (índice 2) e 20 é o maior (índice 1).
                - Existem m! (m fatorial) permutações possíveis para uma dimensão 'm'.

            b.  Distribuição de Probabilidade (P):
                - A frequência de cada um dos m! padrões ordinais é contada ao longo de toda a série temporal.
                - P = {p_π}, onde p_π é a probabilidade de ocorrência da permutação π.

            c.  Entropia de Permutação Normalizada (Hs):
                - Mede a desordem ou aleatoriedade dos padrões ordinais.
                - Fórmula: Hs = - (1 / log(m!)) * Σ (p_π * log(p_π))
                - Varia de 0 (totalmente ordenada/determinística) a 1 (totalmente aleatória/uniforme).

            d.  Complexidade Estatística (C):
                - Mede a estrutura ou organização da série, distinguindo entre ruído e processos com correlação.
                - Uma forma comum (LMC - López-Ruiz-Mancini-Calbet) é: C = Q_J * Hs
                - Onde Q_J é uma divergência de Jensen normalizada entre a distribuição P observada e uma distribuição uniforme (P_eq).
                - Q_J = J(P, P_eq) / J_max, onde J_max é a divergência máxima possível.
                - J(P, P_eq) = H(0.5*P + 0.5*P_eq) - 0.5*H(P) - 0.5*H(P_eq) (H é a entropia de Shannon).
                - C varia de 0 (ruído ou ordem perfeita) a um valor máximo para sistemas com estrutura complexa.

        4.  Plano de Fisher-Shannon:
            - Este plano utiliza a Entropia de Shannon Normalizada (Hs) e a Informação de Fisher Normalizada (F).
            - Parâmetros:
                - Dimensão de Imersão (m): Número de pontos na janela de observação (ex: 3, 4, 5).
                - Atraso (τ): Intervalo entre os pontos na janela (geralmente 1).

            a.  Entropia de Shannon Normalizada (Hs):
                - É a mesma Entropia de Permutação normalizada do plano Bandt-Pompe, calculada a partir da distribuição de probabilidade P dos padrões ordinais.
                - Hs = - (1 / log(m!)) * Σ (p_π * log(p_π))

            b.  Informação de Fisher Normalizada (F):
                - Mede a capacidade de detectar mudanças locais na distribuição de probabilidade. É sensível à ordem dos elementos.
                - Fórmula (discreta): F = (1 / F_max) * Σ ( (sqrt(p_π+1) - sqrt(p_π))^2 )
                - Onde F_max é o valor máximo possível para a Informação de Fisher para uma dada dimensão.
                - F varia de 0 (distribuição suave, difícil de prever) a 1 (distribuição concentrada, fácil de prever).

        Interpretação dos Planos:
        - Ambos os planos fornecem uma "impressão digital" do comportamento da série temporal.
        - Pontos em diferentes regiões do plano indicam diferentes tipos de processos (ex: ruído, periodicidade, caos, processos com memória).
        - A elipse de pertencimento no gráfico representa a região esperada para o idioma de referência, com base na média e desvio padrão dos pontos de Hs e C/F. Um texto novo que cai dentro dessa elipse é considerado "pertencente" ao idioma.
        - Para textos curtos, os parâmetros (m, τ) e o limiar de pertencimento são ajustados automaticamente para maior robustez.
        """
        text_area.insert(tk.END, content)
        text_area.config(state="disabled") # Torna o texto somente leitura

        # Adiciona um scrollbar ao Text widget
        scrollbar_text = ttk.Scrollbar(calc_window, command=text_area.yview)
        scrollbar_text.pack(side="right", fill="y")
        text_area.config(yscrollcommand=scrollbar_text.set)


def main():
    root = tk.Tk()
    app = ExperimentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
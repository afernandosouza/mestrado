# information_theory/gui_experiment.py

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
# import sqlite3 # Não é mais necessário importar sqlite3 diretamente aqui

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

from config import DATABASE, EMBEDDING_DIM, RANDOM_STATE
from information_theory.dataset_it import load_dataset_it # Usaremos esta para obter os códigos
from signal_processing.text_signal import text_to_signal
from information_theory.experiment_cache import save_experiment, load_experiment
from information_theory.fisher_shannon_experiment import compute_hs_f
from information_theory.bandt_pompe_complexity import bandt_pompe_complexity

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
        # CORREÇÃO AQUI: Carrega os códigos de idioma diretamente de load_dataset_it
        # Não tentamos mais acessar a tabela 'idiomas' diretamente.
        # --------------------------------------------------
        try:
            # load_dataset_it retorna: texts, labels, lang_codes, raw_labels, lang_names
            # Pegamos apenas os lang_codes para os checkboxes
            _, _, self.all_lang_codes, _, _ = load_dataset_it(Path(DATABASE))
            self.all_lang_codes.sort() # Ordena os códigos para exibição

            # Como só usamos códigos, não precisamos de mapeamentos nome<->código
            # self.lang_code_to_name = {code: code for code in self.all_lang_codes}
            # self.lang_name_to_code = {code: code for code in self.all_lang_codes}
            # self.display_lang_names = self.all_lang_codes[:] # Usamos os códigos diretamente como nomes de exibição

        except Exception as e: # Captura qualquer erro ao carregar o dataset
            messagebox.showerror("Erro ao carregar dados", f"Não foi possível carregar os idiomas do dataset: {e}")
            self.all_lang_codes = []
        # --------------------------------------------------
        # FIM DA CORREÇÃO
        # --------------------------------------------------

        # Variáveis para os checkboxes de idioma
        self.lang_vars = {} # Armazena {código_do_idioma: tk.BooleanVar}
        for lang_code in self.all_lang_codes:
            self.lang_vars[lang_code] = tk.BooleanVar(value=False)

        # Variáveis para dimensão e atraso
        self.dim_var = tk.IntVar(value=EMBEDDING_DIM) # Valor padrão do config
        self.tau_var = tk.IntVar(value=1) # Atraso padrão 1

        # Radio button: 'bp', 'fs' (Histograma removido)
        self.plot_type = tk.StringVar(value="bp") 

        self.fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax  = self.fig.add_subplot(111)

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


        # Área de comparação (texto novo) — só para BP e FS
        bottom_frame = ttk.LabelFrame(self.master, text="Comparar texto com idioma de referência")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.text_input = tk.Text(bottom_frame, height=4)
        self.text_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        control_frame = ttk.Frame(bottom_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        self.compare_bp_button = ttk.Button(control_frame, text="Comparar (BP)", command=self.on_compare_bp)
        self.compare_bp_button.pack(fill=tk.X, pady=2)
        create_tooltip(self.compare_bp_button, "Compara o texto no Plano de Complexidade-Entropia (Bandt-Pompe) com o idioma de referência.")

        self.compare_fs_button = ttk.Button(control_frame, text="Comparar (FS)", command=self.on_compare_fs)
        self.compare_fs_button.pack(fill=tk.X, pady=2)
        create_tooltip(self.compare_fs_button, "Compara o texto no Plano de Fisher-Shannon com o idioma de referência.")

        self.result_label = ttk.Label(control_frame, text="", foreground="blue")
        self.result_label.pack(fill=tk.X, pady=5)

    # ------------------------------------------------------------------
    # Geração dos gráficos
    # ------------------------------------------------------------------
    def on_generate(self):
        selected_lang_codes = [code for code, var in self.lang_vars.items() if var.get()]
        if not selected_lang_codes:
            messagebox.showerror("Erro", "Nenhum idioma selecionado.")
            return

        # Pega os parâmetros de dimensão e atraso
        try:
            dim = self.dim_var.get()
            tau = self.tau_var.get()
            if dim <= 1 or tau <= 0:
                messagebox.showerror("Erro de Parâmetros", "Dimensão (m) deve ser > 1 e Atraso (τ) deve ser > 0.")
                return
        except tk.TclError:
            messagebox.showerror("Erro de Parâmetros", "Dimensão (m) e Atraso (τ) devem ser números inteiros.")
            return

        self.generate_button.config(state=tk.DISABLED)
        self.loading_label.config(text="Calculando...", foreground="blue")
        self.master.update_idletasks() # Força a atualização da GUI

        try:
            self.fig.clf() # Limpa a figura atual
            self.ax = self.fig.add_subplot(111) # Adiciona um novo subplot

            plot_type = self.plot_type.get()
            self.current_space = plot_type
            self.current_stats_multi = {}
            self.current_data_points_multi = {}

            all_texts_data = {} # Para carregar textos uma vez
            # Carrega os textos usando os códigos de idioma
            texts_all_db, labels_all_db, lang_codes_all_db, _, _ = load_dataset_it(Path(DATABASE))

            for lang_code in selected_lang_codes:
                if lang_code not in lang_codes_all_db:
                    messagebox.showwarning("Aviso", f"Idioma '{lang_code}' não encontrado nos dados carregados.")
                    continue
                idx_lang = lang_codes_all_db.index(lang_code)
                texts_lang = [texts_all_db[i] for i in range(len(texts_all_db)) if labels_all_db[i] == idx_lang]
                all_texts_data[lang_code] = texts_lang[:1000] # Limita a 1000 textos

            # Histograma foi removido, então não há mais essa opção
            if plot_type == "bp":
                self._plot_bp_space(all_texts_data, selected_lang_codes, dim, tau)
            elif plot_type == "fs":
                self._plot_fs_space(all_texts_data, selected_lang_codes, dim, tau)

            self.canvas.draw()
            self.result_label.config(text="")

        except Exception as e:
            messagebox.showerror("Erro de processamento", f"Ocorreu um erro: {e}")
            print(f"Erro detalhado: {e}") # Para debug
        finally:
            self.generate_button.config(state=tk.NORMAL)
            self.loading_label.config(text="")

    # Removido _plot_histogram

    def _plot_bp_space(self, all_texts_data, selected_lang_codes, dim, tau):

        all_hs_values = []
        all_y_values = []

        for lang_code in selected_lang_codes:
            texts_lang = all_texts_data[lang_code]
            cached = load_experiment(lang_code, "bp", dim, tau) # Usa o código para o cache
            if cached is not None and cached["hs"] is not None:
                hs = cached["hs"]
                C  = cached["y"]
                centroid = np.array([cached["centroid_hs"], cached["centroid_y"]])
                std_hs   = cached["std_hs"]
                std_C    = cached["std_y"]
            else:
                hs_list, C_list = [], []
                for t in texts_lang:
                    sig = text_to_signal(t)
                    if len(sig) >= dim * tau:
                        Hs, C = bandt_pompe_complexity(sig, dim, tau)
                        hs_list.append(Hs)
                        C_list.append(C)

                if not hs_list:
                    messagebox.showwarning("Aviso", f"Nenhum texto longo o suficiente para Bandt-Pompe para o idioma '{lang_code}' com dim={dim}, tau={tau}.")
                    continue

                hs = np.array(hs_list)
                C  = np.array(C_list)

                centroid = np.array([hs.mean(), C.mean()])
                std_hs   = hs.std()
                std_C    = C.std()
                save_experiment(lang_code, "bp", dim, tau, hs, C) # Usa o código para o cache

            self.current_stats_multi[lang_code] = { # Chave é o código
                "lang": lang_code, "space": "bp", "dim": dim, "tau": tau,
                "hs": hs, "y": C, "centroid": centroid, "std_hs": std_hs, "std_y": std_C,
            }
            self.current_data_points_multi[lang_code] = {"type": "bp", "hs": hs, "y": C, "centroid": centroid, "std_hs": std_hs, "std_y": std_C}

            # Plotar os pontos e o centroide para cada idioma
            color = self.color_map[lang_code] # Usa o código para pegar a cor
            self.ax.scatter(hs, C, c=color, s=15, alpha=0.6, label=f"{lang_code} (n={len(hs)})")
            self.ax.scatter(centroid[0], centroid[1], c=color, s=80, marker="X", label=f"Centroide {lang_code}")

            # Elipse de pertencimento
            threshold = 2.0
            ellipse = mpatches.Ellipse(
                xy=(centroid[0], centroid[1]),
                width=2 * threshold * std_hs,
                height=2 * threshold * std_C,
                edgecolor=color,
                facecolor=color,
                alpha=0.08,
                linestyle="--",
                linewidth=1.2,
                zorder=2,
                label=f"Região ±{threshold:.1f}σ {lang_code}",
            )
            self.ax.add_patch(ellipse)

            all_hs_values.extend(hs)
            all_y_values.extend(C)

        if not all_hs_values:
            messagebox.showwarning("Aviso", "Nenhum dado válido para plotar o Plano Bandt-Pompe.")
            return

        self.ax.set_xlim(-0.02, 1.02)
        self.ax.set_ylim(-0.02, 1.02)
        self.ax.set_xlabel("Entropia de permutação normalizada $H_s$")
        self.ax.set_ylabel("Complexidade estatística $C$")
        self.ax.set_title(f"Plano Complexidade–Entropia (Bandt-Pompe) — {', '.join(selected_lang_codes)}\nDimensão (m)={dim}, Atraso (τ)={tau}")
        self.ax.legend(fontsize=8)
        self.ax.grid(True, linestyle="--", alpha=0.4)


    def _plot_fs_space(self, all_texts_data, selected_lang_codes, dim, tau):

        all_hs_values = []
        all_y_values = []

        for lang_code in selected_lang_codes:
            texts_lang = all_texts_data[lang_code]
            cached = load_experiment(lang_code, "fs", dim, tau) # Usa o código para o cache
            if cached is not None and cached["hs"] is not None:
                hs = cached["hs"]
                F  = cached["y"]
                centroid = np.array([cached["centroid_hs"], cached["centroid_y"]])
                std_hs   = cached["std_hs"]
                std_F    = cached["std_y"]
            else:
                hs_list, F_list = [], []
                for t in texts_lang:
                    sig = text_to_signal(t)
                    if len(sig) >= dim * tau:
                        Hs, F = compute_hs_f(sig, dim, tau)
                        hs_list.append(Hs)
                        F_list.append(F)

                if not hs_list:
                    messagebox.showwarning("Aviso", f"Nenhum texto longo o suficiente para Fisher-Shannon para o idioma '{lang_code}' com dim={dim}, tau={tau}.")
                    continue

                hs = np.array(hs_list)
                F  = np.array(F_list)

                centroid = np.array([hs.mean(), F.mean()])
                std_hs   = hs.std()
                std_F    = F.std()
                save_experiment(lang_code, "fs", dim, tau, hs, F) # Usa o código para o cache

            self.current_stats_multi[lang_code] = { # Chave é o código
                "lang": lang_code, "space": "fs", "dim": dim, "tau": tau,
                "hs": hs, "y": F, "centroid": centroid, "std_hs": std_hs, "std_y": std_F,
            }
            self.current_data_points_multi[lang_code] = {"type": "fs", "hs": hs, "y": F, "centroid": centroid, "std_hs": std_hs, "std_y": std_F}

            # Plotar os pontos e o centroide para cada idioma
            color = self.color_map[lang_code] # Usa o código para pegar a cor
            self.ax.scatter(hs, F, c=color, s=15, alpha=0.6, label=f"{lang_code} (n={len(hs)})")
            self.ax.scatter(centroid[0], centroid[1], c=color, s=80, marker="X", label=f"Centroide {lang_code}")

            # Elipse de pertencimento
            threshold = 2.0
            ellipse = mpatches.Ellipse(
                xy=(centroid[0], centroid[1]),
                width=2 * threshold * std_hs,
                height=2 * threshold * std_F,
                edgecolor=color,
                facecolor=color,
                alpha=0.08,
                linestyle="--",
                linewidth=1.2,
                zorder=2,
                label=f"Região ±{threshold:.1f}σ {lang_code}",
            )
            self.ax.add_patch(ellipse)

            all_hs_values.extend(hs)
            all_y_values.extend(F)

        if not all_hs_values:
            messagebox.showwarning("Aviso", "Nenhum dado válido para plotar o Plano Fisher-Shannon.")
            return

        self.ax.set_xlim(-0.02, 1.02)
        self.ax.set_ylim(-0.02, 1.02)
        self.ax.set_xlabel("Entropia de permutação normalizada $H_s$")
        self.ax.set_ylabel("Informação de Fisher normalizada $F$")
        self.ax.set_title(f"Plano Fisher–Shannon — {', '.join(selected_lang_codes)}\nDimensão (m)={dim}, Atraso (τ)={tau}")
        self.ax.legend(fontsize=8)
        self.ax.grid(True, linestyle="--", alpha=0.4)


    # ------------------------------------------------------------------
    # Comparação de texto novo
    # ------------------------------------------------------------------
    def _compare_text_in_space(self, space: str):
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
            return

        # Pega os parâmetros de dimensão e atraso do gráfico atual
        try:
            dim = self.dim_var.get()
            tau = self.tau_var.get()
        except tk.TclError:
            messagebox.showerror("Erro de Parâmetros", "Dimensão (m) e Atraso (τ) devem ser números inteiros.")
            return

        # Pega as estatísticas do idioma de referência (usando o CÓDIGO)
        ref_stats = self.current_stats_multi[ref_lang_code]
        # dim = ref_stats["dim"] # Usar dim/tau do input do usuário para o texto novo
        # tau = ref_stats["tau"]
        centroid = ref_stats["centroid"]
        std_hs = ref_stats["std_hs"]
        std_y  = ref_stats["std_y"]

        sig = text_to_signal(texto)
        if len(sig) < dim * tau: # Verifica se o sinal é longo o suficiente para Bandt-Pompe
            messagebox.showwarning("Aviso", f"Texto muito curto para análise (min. {dim*tau} caracteres).")
            return

        if space == "bp":
            Hs_new, Y_new = bandt_pompe_complexity(sig, dim, tau)
            y_label = "C"
        else:  # 'fs'
            Hs_new, Y_new = compute_hs_f(sig, dim, tau)
            y_label = "F"

        # Distância normalizada tipo Mahalanobis diagonal
        d_Hs = (Hs_new - centroid[0]) / (std_hs + 1e-10)
        d_Y  = (Y_new - centroid[1]) / (std_y  + 1e-10)
        dist = float(np.sqrt(d_Hs**2 + d_Y**2))

        threshold = 2.0 # Limiar padrão para pertencimento
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
        self.ax.legend(loc="upper left", fontsize=8)
        self.canvas.draw()

        msg = (
            f"Texto novo: Hs={Hs_new:.4f}, {y_label}={Y_new:.4f}, dist={dist:.2f} (vs {ref_lang_code}). "
            f"{'Pertence ao idioma.' if belongs else 'Não pertence ao idioma.'}"
        )
        self.result_label.config(text=msg, foreground=("green" if belongs else "red"))

    def on_compare_bp(self):
        self._compare_text_in_space("bp")

    def on_compare_fs(self):
        self._compare_text_in_space("fs")

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

        lang_str = "_".join(selected_lang_codes) # Usa códigos para o nome do arquivo
        file_name = f"{self.current_space}_{lang_str}_m{self.dim_var.get()}_t{self.tau_var.get()}.png"
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

        lang_str = "_".join(selected_lang_codes) # Usa códigos para o nome do arquivo
        file_name = f"data_{self.current_space}_{lang_str}_m{self.dim_var.get()}_t{self.tau_var.get()}.csv"
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

                    # O histograma foi removido, então não há mais essa opção de exportação
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

        2.  Plano de Complexidade-Entropia (Bandt-Pompe):
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

        3.  Plano de Fisher-Shannon:
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
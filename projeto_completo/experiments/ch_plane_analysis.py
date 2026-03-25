# experiments/ch_plane_analysis.py

"""
Análise do Plano Complexidade-Entropia (CH Plane) para LID
Integra resultados da reprodução baseline com métricas de teoria da informação
"""

import warnings

from baseline_reproduction.config import CH_HS_THRESHOLD, CH_CJS_THRESHOLD

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from information_theory.complexity_entropy_plane import ComplexityEntropyPlane
from information_theory.fisher_shannon_plane import FisherShannonPlane
from signal_processing.text_signal import text_to_signal
from data.dataset_loader import load_dataset_sqlite
from config import *

# Configuração visual
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class CHPlaneAnalyzer:
    """
    Analisador completo do plano Complexidade-Entropia para LID
    """

    def __init__(self, database: str, embedding_dim: int = 6):
        """
        Inicializa o analisador

        Args:
            database: Caminho do banco SQLite
            embedding_dim: Dimensão de imersão (D) para Bandt-Pompe
        """
        self.database = database
        self.embedding_dim = embedding_dim

        self.ch_analyzer = ComplexityEntrencyPlane(embedding_dim)
        self.fs_analyzer = FisherShannonPlane(embedding_dim)

        self.texts = None
        self.labels = None
        self.signals = None

        self.hs_values = None
        self.cjs_values = None
        self.fi_values = None

        self.results_dir = Path("results/information_theory")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.plots_dir = Path("results/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, sample_size: int = None):
        """
        Carrega dataset do SQLite

        Args:
            sample_size: Número de textos a carregar (None = todos)
        """
        print("Carregando dataset...")
        self.texts, self.labels = load_dataset_sqlite(self.database)

        if sample_size:
            indices = np.random.choice(len(self.texts), sample_size, replace=False)
            self.texts = [self.texts[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

        print(f"✓ Carregados {len(self.texts)} textos")
        print(f"✓ Idiomas únicos: {len(set(self.labels))}")

        return self

    def compute_ch_metrics(self):
        """
        Calcula métricas HS e CJS para todos os textos
        """
        print("\nCalculando métricas HS e CJS (Bandt-Pompe)...")

        hs_list = []
        cjs_list = []

        for text in tqdm(self.texts, desc="CH Plane", total=len(self.texts)):
            signal = text_to_signal(text)
            hs, cjs = self.ch_analyzer.analyzer.compute_both_metrics(signal)

            hs_list.append(hs)
            cjs_list.append(cjs)

        self.hs_values = np.array(hs_list)
        self.cjs_values = np.array(cjs_list)

        print(f"✓ Métricas calculadas para {len(self.texts)} textos")

        return self

    def compute_fs_metrics(self):
        """
        Calcula métricas Fisher-Shannon
        """
        print("\nCalculando métricas Fisher-Shannon...")

        fi_list = []

        for text in tqdm(self.texts, desc="FS Plane", total=len(self.texts)):
            signal = text_to_signal(text)
            fi = self.fs_analyzer.compute_fisher_information(signal)
            fi_list.append(fi)

        self.fi_values = np.array(fi_list)

        print(f"✓ Métricas Fisher-Shannon calculadas")

        return self

    def create_dataframe(self) -> pd.DataFrame:
        """
        Cria DataFrame com todas as métricas
        """
        df = pd.DataFrame({
            'text_id': range(len(self.texts)),
            'language': self.labels,
            'hs': self.hs_values,
            'cjs': self.cjs_values,
            'fi': self.fi_values if self.fi_values is not None else np.nan
        })

        return df

    def plot_ch_plane(self, df: pd.DataFrame):
        """
        Plota plano Complexidade-Entropia
        """
        print("\nGerando visualizações do plano CH...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Plano Complexidade-Entropia (CH Plane) - Análise LID',
                     fontsize=16, fontweight='bold')

        # 1. Todos os idiomas
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(df['hs'], df['cjs'], c=pd.factorize(df['language']),
                              cmap='tab20', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Entropia Normalizada (HS)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Complexidade Estatística (CJS)', fontsize=11, fontweight='bold')
        ax1.set_title('Todos os Idiomas (N={})'.format(len(df)), fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Zoom em região estruturada
        ax2 = axes[0, 1]
        subset = df[(df['hs'] > 0.3) & (df['hs'] < 0.8) &
                    (df['cjs'] > 0.15) & (df['cjs'] < 0.7)]
        scatter2 = ax2.scatter(subset['hs'], subset['cjs'],
                              c=pd.factorize(subset['language']),
                              cmap='tab20', alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Entropia Normalizada (HS)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Complexidade Estatística (CJS)', fontsize=11, fontweight='bold')
        ax2.set_title('Zoom: Região Estruturada', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Centróides por idioma
        ax3 = axes[1, 0]
        centroids = df.groupby('language')[['hs', 'cjs']].mean()
        colors = plt.cm.tab20(np.linspace(0, 1, len(centroids)))

        for idx, (lang, row) in enumerate(centroids.iterrows()):
            ax3.scatter(row['hs'], row['cjs'], s=200, alpha=0.8,
                       edgecolors='black', linewidth=2, color=colors[idx], label=lang)

        ax3.set_xlabel('HS Médio', fontsize=11, fontweight='bold')
        ax3.set_ylabel('CJS Médio', fontsize=11, fontweight='bold')
        ax3.set_title('Centróides por Idioma', fontsize=12, fontweight='bold')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
        ax3.grid(True, alpha=0.3)

        # 4. Distribuição HS por idioma (top 10)
        ax4 = axes[1, 1]
        top_langs = df['language'].value_counts().head(10).index
        df_top = df[df['language'].isin(top_langs)]

        sns.boxplot(data=df_top, x='language', y='hs', ax=ax4, palette='Set2')
        ax4.set_xlabel('Idioma', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Entropia (HS)', fontsize=11, fontweight='bold')
        ax4.set_title('Distribuição HS - Top 10 Idiomas', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        filepath = self.plots_dir / 'ch_plane_analysis_complete.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {filepath}")

        plt.close()

    def plot_fs_plane(self, df: pd.DataFrame):
        """
        Plota plano Fisher-Shannon
        """
        print("\nGerando visualizações do plano Fisher-Shannon...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Plano Fisher-Shannon (FS Plane) - Análise LID',
                     fontsize=16, fontweight='bold')

        # 1. Plano FS completo
        ax1 = axes
        scatter1 = ax1.scatter(df['fi'], df['hs'], c=pd.factorize(df['language']),
                              cmap='tab20', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Informação de Fisher (FI)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Entropia Normalizada (HS)', fontsize=11, fontweight='bold')
        ax1.set_title('Plano Fisher-Shannon', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Centróides FS
        ax2 = axes[1]
        centroids_fs = df.groupby('language')[['fi', 'hs']].mean()
        colors = plt.cm.tab20(np.linspace(0, 1, len(centroids_fs)))

        for idx, (lang, row) in enumerate(centroids_fs.iterrows()):
            ax2.scatter(row['fi'], row['hs'], s=200, alpha=0.8,
                       edgecolors='black', linewidth=2, color=colors[idx], label=lang)

        ax2.set_xlabel('FI Médio', fontsize=11, fontweight='bold')
        ax2.set_ylabel('HS Médio', fontsize=11, fontweight='bold')
        ax2.set_title('Centróides - Plano Fisher-Shannon', fontsize=12, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        filepath = self.plots_dir / 'fs_plane_analysis.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {filepath}")

        plt.close()

    def analyze_regions(self, df: pd.DataFrame):
        """
        Analisa regiões no plano CH (estruturado, ruído, caótico)
        """
        print("\nAnalisando regiões no plano CH...")

        # Definir regiões
        structured = (df['hs'] < CH_HS_THRESHOLD) & (df['cjs'] > CH_CJS_THRESHOLD)
        noise = (df['hs'] > CH_HS_THRESHOLD) & (df['cjs'] < CH_CJS_THRESHOLD)
        chaotic = (df['hs'] > CH_HS_THRESHOLD) & (df['cjs'] > CH_CJS_THRESHOLD)

        print(f"\n📊 Distribuição por região:")
        print(f"  Estruturado (HS<{CH_HS_THRESHOLD}, CJS>{CH_CJS_THRESHOLD}): {structured.sum():,} ({structured.mean()*100:.1f}%)")
        print(f"  Ruído (HS>{CH_HS_THRESHOLD}, CJS<{CH_CJS_THRESHOLD}): {noise.sum():,} ({noise.mean()*100:.1f}%)")
        print(f"  Caótico (HS>{CH_HS_THRESHOLD}, CJS>{CH_CJS_THRESHOLD}): {chaotic.sum():,} ({chaotic.mean()*100:.1f}%)")

        # Salvar análise
        regions_df = pd.DataFrame({
            'region': ['structured', 'noise', 'chaotic'],
            'count': [structured.sum(), noise.sum(), chaotic.sum()],
            'percentage': [structured.mean()*100, noise.mean()*100, chaotic.mean()*100]
        })

        regions_df.to_csv(self.results_dir / 'ch_regions_analysis.csv', index=False)

        return regions_df

    def compute_centroids(self, df: pd.DataFrame) -> dict:
        """
        Calcula centróides de cada idioma no plano CH
        """
        print("\nCalculando centróides por idioma...")

        centroids = {}

        for lang in df['language'].unique():
            lang_data = df[df['language'] == lang]

            centroids[lang] = {
                'hs_mean': lang_data['hs'].mean(),
                'hs_std': lang_data['hs'].std(),
                'cjs_mean': lang_data['cjs'].mean(),
                'cjs_std': lang_data['cjs'].std(),
                'count': len(lang_data)
            }

        # Salvar centróides
        centroids_df = pd.DataFrame(centroids).T
        centroids_df.to_csv(self.results_dir / 'ch_centroids.csv')

        print(f"✓ Centróides calculados para {len(centroids)} idiomas")

        return centroids

    def classify_by_ch_distance(self, df: pd.DataFrame, centroids: dict,
                               test_split: float = 0.2) -> dict:
        """
        Classifica textos por distância euclidiana no plano CH
        """
        print("\nExecutando classificação por distância CH...")

        X_train, X_test, y_train, y_test = train_test_split(
            df.index, df['language'], test_size=test_split,
            stratify=df['language'], random_state=42
        )

        predictions = []
        confidences = []

        for idx in tqdm(X_test, desc="Classificando", total=len(X_test)):
            row = df.loc[idx]

            min_distance = float('inf')
            best_lang = None

            for lang, centroid in centroids.items():
                distance = np.sqrt(
                    (row['hs'] - centroid['hs_mean'])**2 +
                    (row['cjs'] - centroid['cjs_mean'])**2
                )

                if distance < min_distance:
                    min_distance = distance
                    best_lang = lang

            predictions.append(best_lang)
            confidence = 1.0 / (1.0 + min_distance)
            confidences.append(confidence)

        accuracy = accuracy_score(y_test, predictions)

        results = {
            'accuracy': accuracy,
            'y_test': y_test.values,
            'predictions': predictions,
            'confidences': confidences
        }

        print(f"✓ Acurácia classificação CH: {accuracy:.4f}")

        return results

    def generate_report(self, df: pd.DataFrame, centroids: dict,
                       ch_results: dict, regions_df: pd.DataFrame):
        """
        Gera relatório completo em texto
        """
        print("\nGerando relatório...")

        report_path = self.results_dir / f'ch_plane_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ANÁLISE DO PLANO COMPLEXIDADE-ENTROPIA (CH PLANE)\n")
            f.write("Identificação de Idiomas em Textos\n")
            f.write("="*70 + "\n\n")

            f.write(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"Dimensão de imersão (D): {self.embedding_dim}\n")
            f.write(f"Total de textos analisados: {len(df):,}\n")
            f.write(f"Idiomas únicos: {df['language'].nunique()}\n\n")

            f.write("-"*70 + "\n")
            f.write("ESTATÍSTICAS GLOBAIS\n")
            f.write("-"*70 + "\n\n")

            f.write(f"Entropia (HS):\n")
            f.write(f"  Média: {df['hs'].mean():.4f}\n")
            f.write(f"  Desvio padrão: {df['hs'].std():.4f}\n")
            f.write(f"  Mín/Máx: {df['hs'].min():.4f} / {df['hs'].max():.4f}\n\n")

            f.write(f"Complexidade (CJS):\n")
            f.write(f"  Média: {df['cjs'].mean():.4f}\n")
            f.write(f"  Desvio padrão: {df['cjs'].std():.4f}\n")
            f.write(f"  Mín/Máx: {df['cjs'].min():.4f} / {df['cjs'].max():.4f}\n\n")

            f.write("-"*70 + "\n")
            f.write("ANÁLISE POR REGIÃO\n")
            f.write("-"*70 + "\n\n")

            for _, row in regions_df.iterrows():
                f.write(f"{row['region'].upper()}: {row['count']:,} textos ({row['percentage']:.1f}%)\n")

            f.write("\n" + "-"*70 + "\n")
            f.write("CENTRÓIDES POR IDIOMA\n")
            f.write("-"*70 + "\n\n")

            for lang in sorted(centroids.keys()):
                c = centroids[lang]
                f.write(f"{lang.upper()}:\n")
                f.write(f"  HS: {c['hs_mean']:.4f} ± {c['hs_std']:.4f}\n")
                f.write(f"  CJS: {c['cjs_mean']:.4f} ± {c['cjs_std']:.4f}\n")
                f.write(f"  Amostras: {c['count']}\n\n")

            f.write("-"*70 + "\n")
            f.write("RESULTADOS DE CLASSIFICAÇÃO\n")
            f.write("-"*70 + "\n\n")

            f.write(f"Acurácia (distância euclidiana): {ch_results['accuracy']:.4f}\n\n")

            f.write("Confiança média por idioma:\n")
            for lang in sorted(set(ch_results['y_test'])):
                mask = np.array(ch_results['y_test']) == lang
                if mask.sum() > 0:
                    conf = np.mean(np.array(ch_results['confidences'])[mask])
                    f.write(f"  {lang}: {conf:.4f}\n")

        print(f"✓ Relatório salvo: {report_path}")

    def run_complete_analysis(self, sample_size: int = None):
        """
        Executa análise completa
        """
        print("\n" + "="*70)
        print("ANÁLISE COMPLETA DO PLANO COMPLEXIDADE-ENTROPIA")
        print("="*70 + "\n")

        # 1. Carregar dados
        self.load_data(sample_size)

        # 2. Calcular métricas
        self.compute_ch_metrics()
        self.compute_fs_metrics()

        # 3. Criar DataFrame
        df = self.create_dataframe()

        # 4. Gerar visualizações
        self.plot_ch_plane(df)
        self.plot_fs_plane(df)

        # 5. Analisar regiões
        regions_df = self.analyze_regions(df)

        # 6. Calcular centróides
        centroids = self.compute_centroids(df)

        # 7. Classificação
        ch_results = self.classify_by_ch_distance(df, centroids)

        # 8. Gerar relatório
        self.generate_report(df, centroids, ch_results, regions_df)

        print("\n" + "="*70)
        print("✓ ANÁLISE CONCLUÍDA COM SUCESSO")
        print("="*70 + "\n")

        return {
            'dataframe': df,
            'centroids': centroids,
            'ch_results': ch_results,
            'regions': regions_df
        }


def main():
    """
    Função principal
    """
    analyzer = CHPlaneAnalyzer(
        database=DATABASE,
        embedding_dim=EMBEDDING_DIM
    )

    results = analyzer.run_complete_analysis(sample_size=5000)

    print("\n📊 Resultados salvos em:")
    print(f"  - {analyzer.results_dir}")
    print(f"  - {analyzer.plots_dir}")


if __name__ == "__main__":
    main()
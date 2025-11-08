#!/usr/bin/env python3
"""
statistical_analysis_pipeline.py
Pipeline to perform statistical tests and visualizations for language metrics (Format 1).
Usage: python statistical_analysis_pipeline.py --metrics PATH_TO_METRICS.csv
If no metrics file provided, a demo dataset will be created.
Expected metrics format (one row per language):
idioma, entropia, complexidade, energia_1, ..., energia_32
Requires: pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import itertools
import random

def read_files(metrics_path=None, classification_path='/mnt/data/classificacao_idiomas.xlsx'):
    # Read classification
    class_path = Path(classification_path)
    if not class_path.exists():
        raise FileNotFoundError(f'Classification file not found: {classification_path}')
    class_df = pd.read_excel(class_path)
    # Read metrics
    if metrics_path is None or not Path(metrics_path).exists():
        print('[INFO] Metrics file not found; generating a small demo dataset. Replace with your real metrics file.')
        # Demo: generate synthetic data for 10 languages
        langs = class_df['Código'].tolist()[:12] if 'Código' in class_df.columns else ['pt','en','es','fr','de','ru','hi','fa','ta','ar','tr','az']
        rng = np.random.default_rng(42)
        rows = []
        for i,lang in enumerate(langs):
            ent = 0.7 + 0.15*rng.standard_normal()
            comp = 0.2 + 0.3*np.abs(rng.standard_normal())*(0.5 + 0.5*(i%3==0))
            energies = np.abs(0.01 + 0.01*rng.standard_normal(32) + (0.005*(i%3)))
            rows.append([lang, float(ent), float(comp)] + energies.tolist())
        cols = ['idioma','entropia','complexidade'] + [f'energia_{i+1}' for i in range(32)]
        metrics_df = pd.DataFrame(rows, columns=cols)
    else:
        metrics_df = pd.read_csv(metrics_path) if metrics_path.endswith('.csv') else pd.read_excel(metrics_path)
    return metrics_df, class_df

def merge_classification(metrics_df, class_df):
    # Normalize key names
    k = 'idioma'
    if k not in metrics_df.columns:
        raise KeyError("Metrics file must contain 'idioma' column with language codes matching the classification file 'Código'.")
    # Try to match by code column
    code_col = None
    for candidate in ['Código','codigo','code','lang','iso']:
        if candidate in class_df.columns:
            code_col = candidate
            break
    if code_col is None:
        # attempt to find language name column
        if 'Idioma' in class_df.columns:
            code_col = 'Idioma'
            class_df = class_df.rename(columns={code_col:'IdiomaName'})
            print('[WARN] Classification file has no explicit code column; matching by language names may fail.')
            # attempt join by language full name
            merged = metrics_df.merge(class_df, left_on='idioma', right_on='IdiomaName', how='left')
            return merged
        else:
            raise KeyError("Classification file must contain a column identifying the language code or name. Expected 'Código' or similar.")
    merged = metrics_df.merge(class_df, left_on='idioma', right_on=code_col, how='left')
    return merged

def check_assumptions(grouped, variable):
    # grouped: dict group_name -> array
    # Return Shapiro p-values and Levene p-value
    shapiro = {}
    for g, arr in grouped.items():
        if len(arr) < 3:
            shapiro[g] = np.nan
        else:
            try:
                shapiro[g] = stats.shapiro(arr)[1]
            except Exception:
                shapiro[g] = np.nan
    # Levene across groups
    arrays = [arr for arr in grouped.values() if len(arr)>0]
    if len(arrays) < 2:
        levene_p = np.nan
    else:
        levene_p = stats.levene(*arrays, center='median')[1]
    return shapiro, levene_p

def anova_or_kruskal(df, variable, group_col='Família', alpha=0.05):
    # group_col expected in df
    groups = df.groupby(group_col)[variable].apply(lambda x: x.dropna().values).to_dict()
    shapiro, levene_p = check_assumptions(groups, variable)
    normal_ok = all((p is not np.nan and p>0.05) for p in shapiro.values() if not np.isnan(p))
    homosced = (not np.isnan(levene_p)) and (levene_p>0.05)
    result = {'shapiro': shapiro, 'levene_p': levene_p}
    if normal_ok and homosced:
        # perform one-way ANOVA (ordinary)
        group_lists = [g for g in groups.values() if len(g)>0]
        fstat, pval = stats.f_oneway(*group_lists)
        result.update({'test':'ANOVA', 'f':float(fstat), 'p':float(pval)})
        # posthoc Tukey (requires long-form table)
        try:
            data = df[[group_col, variable]].dropna().rename(columns={group_col:'group', variable:'value'})
            tuk = pairwise_tukeyhsd(data['value'], data['group'])
            result['tukey_summary'] = str(tuk.summary())
        except Exception as e:
            result['tukey_error'] = str(e)
    else:
        # Kruskal-Wallis
        group_lists = [g for g in groups.values() if len(g)>0]
        if len(group_lists) < 2:
            result.update({'test':'KRUSKAL', 'h':np.nan, 'p':np.nan, 'posthoc':None})
        else:
            h, p = stats.kruskal(*group_lists)
            result.update({'test':'KRUSKAL', 'h':float(h), 'p':float(p)})
            # Pairwise Mann-Whitney with Bonferroni
            names = list(groups.keys())
            post = []
            for i,j in itertools.combinations(range(len(names)),2):
                a = groups[names[i]]
                b = groups[names[j]]
                if len(a)>0 and len(b)>0:
                    u,pw = stats.mannwhitneyu(a,b,alternative='two-sided')
                    post.append({'g1':names[i],'g2':names[j],'u':float(u),'p_uncorrected':float(pw)})
            # Bonferroni
            m = len(post)
            for item in post:
                item['p_bonf'] = min(1.0, item['p_uncorrected']*m)
            result['posthoc'] = post
    return result

def spearman_corr_matrix(df, vars_list):
    sub = df[vars_list].dropna()
    rho = sub.corr(method='spearman')
    return rho

def permanova(X, groups, n_permutations=999, metric='euclidean', random_state=42):
    """
    Simple PERMANOVA implementation returning pseudo-F and p-value.
    X: (n_samples, n_features) numeric matrix
    groups: array-like group labels (n_samples)
    """
    rng = np.random.default_rng(random_state)
    D = pairwise_distances(X, metric=metric)
    n = D.shape[0]
    # Convert to squared distances if euclidean to follow some conventions
    # Use classical PERMANOVA pseudo-F computed from sums-of-squares on distances
    unique_groups = np.unique(groups)
    # total sum of squares (from distance matrix)
    A = -0.5 * (D**2)
    grand_mean = A.mean()
    SST = np.sum((A - grand_mean)**2) / n  # not standard; using distance-based approach
    # compute SS among groups
    ss_between = 0.0
    for g in unique_groups:
        idx = np.where(groups==g)[0]
        if len(idx)==0: continue
        Ag = A[np.ix_(idx, idx)]
        mean_g = Ag.mean()
        ss_between += len(idx) * (mean_g - grand_mean)**2
    df_between = len(unique_groups)-1
    df_within = n - len(unique_groups)
    ms_between = ss_between / max(1, df_between)
    # approximate ms_within using SST - ss_between
    ss_within = max(1e-12, SST - ss_between)
    ms_within = ss_within / max(1, df_within)
    pseudoF = ms_between / ms_within if ms_within>0 else np.inf
    # permutations
    perm_stats = []
    for i in range(n_permutations):
        perm = rng.permutation(groups)
        ssb = 0.0
        for g in np.unique(perm):
            idx = np.where(perm==g)[0]
            Ag = A[np.ix_(idx, idx)]
            mean_g = Ag.mean()
            ssb += len(idx) * (mean_g - grand_mean)**2
        msb = ssb / max(1, df_between)
        ssw = max(1e-12, SST - ssb)
        msw = ssw / max(1, df_within)
        perm_stats.append(msb/msw if msw>0 else np.inf)
    perm_stats = np.array(perm_stats)
    pvalue = (np.sum(perm_stats >= pseudoF) + 1) / (len(perm_stats)+1)
    return {'pseudoF':float(pseudoF), 'pvalue':float(pvalue), 'permutation_stats':perm_stats}

def pca_and_lda_plots(X, y, labels, out_prefix='/mnt/data/analysis_plot'):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    pc = pca.fit_transform(Xs)
    plt.figure(figsize=(8,6))
    for lab in np.unique(y):
        idx = np.where(y==lab)[0]
        plt.scatter(pc[idx,0], pc[idx,1], label=str(lab), s=80)
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('PCA (2 components)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_prefix + '_pca.png', dpi=200)
    plt.close()
    # LDA (if more than 1 class and samples >= n_classes)
    unique = np.unique(y)
    if len(unique) > 1 and X.shape[0] >= len(unique):
        try:
            lda = LDA(n_components=min(len(unique)-1,2))
            ld = lda.fit_transform(Xs, y)
            plt.figure(figsize=(8,6))
            for lab in unique:
                idx = np.where(y==lab)[0]
                plt.scatter(ld[idx,0], ld[idx,1] if ld.shape[1]>1 else np.zeros_like(ld[idx,0]), label=str(lab), s=80)
            plt.xlabel('LD1'); 
            if ld.shape[1]>1: plt.ylabel('LD2')
            plt.title('LDA projection')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(out_prefix + '_lda.png', dpi=200)
            plt.close()
        except Exception as e:
            print('[WARN] LDA failed:', e)

def main(args):
    metrics_path = args.metrics
    metrics_df, class_df = read_files(metrics_path, args.classification)
    merged = merge_classification(metrics_df, class_df)
    # Ensure family column exists: try common names
    fam_col = None
    for candidate in ['Família','familia','Family','family']:
        if candidate in merged.columns:
            fam_col = candidate
            break
    if fam_col is None:
        # ask user mapping: try 'Sub‑família' -> fallback group by subfamily
        for candidate in ['Sub‑família','Subfamilia','subfamília','Sub-família','Subfamily']:
            if candidate in merged.columns:
                fam_col = candidate
                break
    if fam_col is None:
        # create a dummy family column from Codigo first letter
        merged['Família'] = merged['idioma'].apply(lambda x: str(x)[:2])
        fam_col = 'Família'
        print('[WARN] No family column found; a dummy Famíla column was created from idioma codes. For best results, ensure classification file has a "Família" column.')
    else:
        # rename to consistent name
        merged = merged.rename(columns={fam_col:'Família'})
        fam_col = 'Família'
    # variables to analyze
    energy_cols = [c for c in merged.columns if str(c).lower().startswith('energia')]
    vars_basic = ['entropia','complexidade']
    vars_all = vars_basic + energy_cols
    # Save merged
    merged.to_csv('/mnt/data/merged_metrics_classification.csv', index=False)
    print('[INFO] Merged file saved to /mnt/data/merged_metrics_classification.csv')
    # 1) Test entropia
    res_ent = anova_or_kruskal(merged, 'entropia', group_col='Família')
    res_comp = anova_or_kruskal(merged, 'complexidade', group_col='Família')
    # 2) Spearman correlation among basic vars
    rho_basic = spearman_corr_matrix(merged, vars_basic)
    # 3) PERMANOVA on full feature set
    feat_df = merged[vars_all].dropna()
    # align groups for PERMANOVA
    feat_idx = feat_df.index.values
    groups = merged.loc[feat_idx, 'Família'].values
    X = feat_df.values
    perm = permanova(X, groups, n_permutations=999)
    # 4) PCA and LDA plots
    pca_and_lda_plots(X, groups, merged.loc[feat_idx, 'idioma'].values, out_prefix='/mnt/data/analysis_plot')
    # 5) Write results
    import json
    summary = {
        'entropia_test': res_ent,
        'complexidade_test': res_comp,
        'spearman_basic': rho_basic.to_dict() if isinstance(rho_basic, pd.DataFrame) else None,
        'permanova': {'pseudoF':perm['pseudoF'],'pvalue':perm['pvalue']}
    }
    Path('/mnt/data').joinpath('analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('[INFO] Summary saved to /mnt/data/analysis_summary.json')
    print('[INFO] PCA and LDA plots saved to /mnt/data/analysis_plot_pca.png and _lda.png (if available).')
    print('[INFO] Full merged table: /mnt/data/merged_metrics_classification.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Statistical analysis pipeline for language metrics (Format 1).')
    parser.add_argument('--metrics', type=str, default=None, help='Path to metrics file (CSV or XLSX) in Format 1. If omitted, demo data is created.')
    parser.add_argument('--classification', type=str, default='/mnt/data/classificacao_idiomas.xlsx', help='Path to classification Excel file.')
    args = parser.parse_args()
    main(args)

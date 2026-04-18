import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

# --------------------------------------------------------------------
# Ajuste do sys.path
# Sobe dois níveis a partir de baseline_reproduction/:
#   baseline_reproduction/ -> projeto_completo/
# Assim todos os módulos do projeto ficam acessíveis diretamente
# --------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]  # projeto_completo/
sys.path.insert(0, str(ROOT_DIR))

import time
import numpy as np
from datetime import datetime

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import *
from data.dataset_loader import load_dataset_sqlite
from data.ti_features_loader import load_ti_features_from_db
from lid_pipeline import LIDPipeline
from spacing_experiment import apply_spacing, apply_spacing_between_two_words

from utils.logger import setup_logger, log_final_results
from evaluation.save_results import save_results
from evaluation.plot_results import plot_results
from evaluation.confusion import save_confusion
from evaluation.statistics import compute_statistics

from utils.system_info import print_and_log_system_info, SystemMonitor, print_and_log_monitor_results

DATABASE_REF = '..\\' + DATABASE
DATABASE_TI_REF = '..\\' + DATABASE_TI

# Função auxiliar para dividir os dados, incluindo raw_labels
def custom_train_test_split(X_spaced, y, raw_labels, test_size, stratify, random_state):
    # train_test_split retorna os índices se X for um range
    indices = np.arange(len(X_spaced))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=stratify, random_state=random_state)

    X_train_spaced = [X_spaced[i] for i in train_idx]
    X_test_spaced = [X_spaced[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    raw_train_labels = [raw_labels[i] for i in train_idx]
    raw_test_labels = [raw_labels[i] for i in test_idx]

    return X_train_spaced, X_test_spaced, y_train, y_test, raw_train_labels, raw_test_labels


def run_experiment():

    logger = setup_logger()

    print("\n====================================================")
    print("REPRODUÇÃO DO EXPERIMENTO DO ARTIGO")
    print("A Signal Processing Method for Text Language Identification")
    print("====================================================\n")

    start_datetime = datetime.now()
    start_time = time.time()

    print_and_log_system_info(logger)

    monitor = SystemMonitor(interval=1)
    monitor.start()

    logger.info("====================================================")
    logger.info("INÍCIO DA EXECUÇÃO DA PIPELINE")
    logger.info("====================================================")
    logger.info(f"Início: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')}")
    logger.info("")

    print("Data/Hora início:", start_datetime.strftime("%d/%m/%Y %H:%M:%S"))
    print()

    logger.info("Inicio da execução")

    print("Carregando dataset do banco SQLite...")

    # load_dataset_sqlite agora retorna texts, labels, unique_langs, raw_labels
    texts, labels, unique_langs, raw_labels = load_dataset_sqlite(DATABASE_REF)

    print("Total de textos carregados:", len(texts))
    print("Total de idiomas:", len(set(labels)))
    print()

    logger.info(f"Total textos: {len(texts)}")
    logger.info(f"Idiomas: {len(set(labels))}")

    train_percent = (1 - TEST_SPLIT) * 100
    test_percent = TEST_SPLIT * 100

    print("Configuração experimental:")
    print(f"Clusters: {N_CLUSTERS}")
    print(f"Execuções por experimento: {N_RUNS}")
    print(f"Treino: {train_percent:.0f}%")
    print(f"Teste: {test_percent:.0f}%")
    if USE_TI_FEATURES:
        print(f"Features de TI ativadas. Usando 'space' = '{TI_FEATURE_SPACE_VALUE}'")
    else:
        print("Features de TI desativadas.")
    print()

    results = {}

    last_y_test = None
    last_preds = None

    # Lógica para carregar features de TI uma única vez no início do experimento
    ti_features_data = None
    use_ti_features_current_run = USE_TI_FEATURES # Variável local para controle
    if use_ti_features_current_run:
        print(f"Carregando features de Teoria da Informação para 'space' = '{TI_FEATURE_SPACE_VALUE}'...")
        ti_features_data = load_ti_features_from_db(DATABASE_TI_REF)
        if not ti_features_data:
            print("Aviso: Não foi possível carregar features de TI. Desativando o uso de TI para esta execução.")
            logger.info("Aviso: Nao foi possivel carregar features de TI. Desativando o uso de TI para esta execucao.")
            use_ti_features_current_run = False
        else:
            print(f"Features de TI carregadas para {len(ti_features_data)} idiomas.")
            logger.info(f"Features de TI carregadas para {len(ti_features_data)} idiomas.")


    for spacing in SPACING_LEVELS:

        print("\n----------------------------------------------------")
        print(f"Experimento com {spacing} espaço(s) entre palavras")
        print("----------------------------------------------------")

        logger.info(f"Experimento {spacing} espaços")

        spacing_start = time.time()

        print("Aplicando espaçamento nos textos...")

        spaced_texts = [
            #apply_spacing(t, spacing) for t in texts
            apply_spacing(t, spacing) for t in texts
        ]

        acc_runs = []

        for run in range(N_RUNS):

            print(f"\nExecução {run+1}/{N_RUNS}")

            logger.info(f"Execução {run+1}/{N_RUNS}")

            # Dividir os textos espaçados, labels e raw_labels
            X_train_spaced, X_test_spaced, y_train, y_test, raw_train_labels, raw_test_labels = custom_train_test_split(
                spaced_texts,
                labels,
                raw_labels, # Passa os raw_labels
                test_size=TEST_SPLIT,
                stratify=labels,
                random_state=RANDOM_STATE
            )

            print("Treinando pipeline...")

            # Passar ti_features_data para o pipeline
            pipeline = LIDPipeline(N_CLUSTERS, ti_features_data if use_ti_features_current_run else None)

            train_start = time.time()

            # O fit do pipeline precisa dos raw_labels para o idioma.
            pipeline.fit(X_train_spaced, y_train, raw_train_labels if use_ti_features_current_run else None)

            train_time = time.time() - train_start

            print(f"Treinamento concluído em {train_time:.2f} segundos")

            print("Executando predições no conjunto de teste...")

            preds = []

            for i, text_spaced in enumerate(tqdm(X_test_spaced, desc="Classificando textos")):
                # O predict do pipeline precisa do raw_test_labels[i] para o idioma.
                preds.append(pipeline.predict(text_spaced, raw_test_labels[i] if use_ti_features_current_run else None))

            acc = np.mean(np.array(preds) == np.array(y_test))

            print(f"Acurácia execução {run+1}: {acc:.4f}")

            logger.info(f"Acuracia execucao {run+1}: {acc:.4f}")

            acc_runs.append(acc)

            last_y_test = y_test
            last_preds = preds

        spacing_time = time.time() - spacing_start

        stats = compute_statistics(acc_runs)

        results[spacing] = stats

        print("\nResultado médio para", spacing, "espaço(s):")
        print("Acurácia média:", "{:.4f}".format(results[spacing]["mean"]))
        print(f"Tempo do experimento: {spacing_time/60:.2f} minutos")

        logger.info(f"Acuracia media {spacing}: {results[spacing]["mean"]:.4f}")

    print("\n====================================================")
    print("RESULTADOS FINAIS")
    print("====================================================\n")

    total_mean = 0
    global_mean = 0
    for k, v in results.items():
        print(f"{k} espaço(s) → {v['mean']:.4f}")
        total_mean += v['mean']

    if total_mean > 0:
        global_mean = total_mean/len(results)

    print(f"\nMédia global: {global_mean:.4f}")

    log_final_results(logger, results, global_mean)

    end_time = datetime.now()

    total_time = end_time - start_datetime

    logger.info("====================================================")
    logger.info("TEMPO TOTAL DE EXECUÇÃO")
    logger.info("====================================================")

    logger.info(f"Fim: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Tempo total: {str(total_time)}")
    logger.info("")

    monitor.stop()
    stats = monitor.get_stats()
    print_and_log_monitor_results(stats, logger)

    save_results(results)
    plot_results(results)

    if last_y_test is not None:
        save_confusion(last_y_test, last_preds)

    end_datetime = datetime.now()
    total_time = time.time() - start_time

    print("\n====================================================")
    print("EXECUÇÃO FINALIZADA")
    print("====================================================")

    print("Data/Hora início:", start_datetime.strftime("%d/%m/%Y %H:%M:%S"))
    print("Data/Hora término:", end_datetime.strftime("%d/%m/%Y %H:%M:%S"))

    print(f"Tempo total de execução: {total_time/60:.2f} minutos")

    logger.info("Fim da execução")

if __name__ == "__main__":
    run_experiment()
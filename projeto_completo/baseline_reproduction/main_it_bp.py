# projeto_completo/main_it_bp.py

import warnings
warnings.filterwarnings('ignore')

import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import time
import numpy as np
from datetime import datetime

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import *
from data.dataset_loader import load_dataset_sqlite
from lid_pipeline_bp import LIDPipelineBP # Importa a nova pipeline
from spacing_experiment import apply_spacing

from utils.logger import setup_logger, log_final_results
from evaluation.save_results import save_results
from evaluation.plot_results import plot_results
from evaluation.confusion import save_confusion
from evaluation.statistics import compute_statistics

from utils.system_info import print_and_log_system_info, SystemMonitor, print_and_log_monitor_results

DATABASE_REF = os.path.join(ROOT_DIR, DATABASE) # Caminho absoluto para o DB principal

def run_experiment_bp():

    logger = setup_logger() # Log separado

    print("\n====================================================")
    print("REPRODUÇÃO DO EXPERIMENTO COM FEATURES WAVELET + BANDT-POMPE")
    print("====================================================\n")

    start_datetime = datetime.now()
    start_time = time.time()

    print_and_log_system_info(logger)

    monitor = SystemMonitor(interval=1)
    monitor.start()

    logger.info("====================================================")
    logger.info("INÍCIO DA EXECUÇÃO DA PIPELINE (WAVELET + BP)")
    logger.info("====================================================")
    logger.info(f"Início: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')}")
    logger.info("")

    print("Data/Hora início:", start_datetime.strftime("%d/%m/%Y %H:%M:%S"))
    print()

    logger.info("Inicio da execução")

    print("Carregando dataset do banco SQLite...")

    texts, labels, unique_langs = load_dataset_sqlite(DATABASE_REF)

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
    print()

    results = {}

    last_y_test = None
    last_preds = None

    for spacing in SPACING_LEVELS:

        print("\n----------------------------------------------------")
        print(f"Experimento com {spacing} espaço(s) entre palavras (WAVELET + BP)")
        print("----------------------------------------------------")

        logger.info(f"Experimento {spacing} espaços (WAVELET + BP)")

        spacing_start = time.time()

        print("Aplicando espaçamento nos textos...")

        spaced_texts = [
            apply_spacing(t, spacing) for t in texts
        ]

        acc_runs = []

        for run in range(N_RUNS):

            print(f"\nExecução {run+1}/{N_RUNS}")

            logger.info(f"Execução {run+1}/{N_RUNS}")

            X_train, X_test, y_train, y_test = train_test_split(
                spaced_texts,
                labels,
                test_size=TEST_SPLIT,
                stratify=labels,
                random_state=None
            )

            print("Treinando pipeline (WAVELET + BP)...")

            pipeline = LIDPipelineBP(N_CLUSTERS) # Instancia a nova pipeline

            train_start = time.time()

            pipeline.fit(X_train, y_train)

            train_time = time.time() - train_start

            print(f"Treinamento concluído em {train_time:.2f} segundos")

            print("Executando predições no conjunto de teste...")

            preds = []

            for i, text in enumerate(tqdm(X_test, desc="Classificando textos")):
                # Passa o rótulo verdadeiro para o predict para buscar as features de TI
                # Isso é para fins de avaliação do potencial das features.
                preds.append(pipeline.predict(text, language=y_test[i])) 

            acc = np.mean(np.array(preds) == np.array(y_test))

            print(f"Acurácia execução {run+1}: {acc:.4f}")

            logger.info(f"Acuracia execucao {run+1}: {acc:.4f}")

            acc_runs.append(acc)

            last_y_test = y_test
            last_preds = preds

        spacing_time = time.time() - spacing_start

        stats = compute_statistics(acc_runs)

        results[spacing] = stats

        print("\nResultado médio para", spacing, "espaço(s) (WAVELET + BP):")
        print("Acurácia média:", "{:.4f}".format(results[spacing]["mean"]))
        print(f"Tempo do experimento: {spacing_time/60:.2f} minutos")

        logger.info(f"Acuracia media {spacing}: {results[spacing]["mean"]:.4f}")

    print("\n====================================================")
    print("RESULTADOS FINAIS (WAVELET + BP)")
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
    logger.info("TEMPO TOTAL DE EXECUÇÃO (WAVELET + BP)")
    logger.info("====================================================")

    logger.info(f"Fim: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Tempo total: {str(total_time)}")
    logger.info("")

    monitor.stop()
    stats = monitor.get_stats()
    print_and_log_monitor_results(stats, logger)

    save_results(results, filename_prefix="results_bp") # Salva com prefixo
    plot_results(results, filename_prefix="plot_bp") # Plota com prefixo

    if last_y_test is not None:
        save_confusion(last_y_test, last_preds, filename_prefix="confusion_bp") # Confusão com prefixo

    end_datetime = datetime.now()
    total_time = time.time() - start_time

    print("\n====================================================")
    print("EXECUÇÃO FINALIZADA (WAVELET + BP)")
    print("====================================================")

    print("Data/Hora início:", start_datetime.strftime("%d/%m/%Y %H:%M:%S"))
    print("Data/Hora término:", end_datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    print(f"Tempo total de execução: {total_time/60:.2f} minutos")

    logger.info("Fim da execução")

if __name__ == "__main__":
    run_experiment_bp()

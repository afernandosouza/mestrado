import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
from datetime import datetime

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import *
from data.dataset_loader import load_dataset_sqlite
from pipeline.lid_pipeline import LIDPipeline
from experiments.spacing_experiment import apply_spacing

from logger import setup_logger, log_final_results
from evaluation.save_results import save_results
from evaluation.plot_results import plot_results
from evaluation.confusion import save_confusion
from evaluation.statistics import compute_statistics

from utils.system_info import print_and_log_system_info, SystemMonitor, print_and_log_monitor_results


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

    texts, labels = load_dataset_sqlite(DATABASE)

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
        print(f"Experimento com {spacing} espaço(s) entre palavras")
        print("----------------------------------------------------")

        logger.info(f"Experimento {spacing} espaços")

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

            print("Treinando pipeline...")

            pipeline = LIDPipeline(N_CLUSTERS)

            train_start = time.time()

            pipeline.fit(X_train, y_train)

            train_time = time.time() - train_start

            print(f"Treinamento concluído em {train_time:.2f} segundos")

            print("Executando predições no conjunto de teste...")

            preds = []

            for text in tqdm(X_test, desc="Classificando textos"):

                preds.append(pipeline.predict(text))

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

    for k, v in results.items():
        print(f"{k} espaço(s) → {v['mean']:.4f}")

    log_final_results(logger, results)

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
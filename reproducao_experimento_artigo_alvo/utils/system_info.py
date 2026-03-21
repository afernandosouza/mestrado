import platform
import psutil
import threading
import time
from datetime import datetime

try:
    import GPUtil
except ImportError:
    GPUtil = None


def get_system_info():

    info = {}

    # Sistema operacional
    info["os"] = platform.system() + " " + platform.release()

    # CPU
    info["cpu_model"] = platform.processor()
    info["cpu_physical_cores"] = psutil.cpu_count(logical=False)
    info["cpu_logical_cores"] = psutil.cpu_count(logical=True)

    freq = psutil.cpu_freq()

    if freq:
        info["cpu_freq_mhz"] = round(freq.max, 2)

    # RAM
    ram = psutil.virtual_memory()

    info["ram_total_gb"] = round(ram.total / (1024**3), 2)

    # GPU
    gpus = []

    if GPUtil:

        try:
            gpu_list = GPUtil.getGPUs()

            for gpu in gpu_list:
                gpus.append({
                    "name": gpu.name,
                    "memory_total_gb": round(gpu.memoryTotal / 1024, 2)
                })

        except Exception:
            pass

    info["gpus"] = gpus

    return info

def print_and_log_system_info(logger=None):

    info = get_system_info()

    print("\n====================================================")
    print("INFORMAÇÕES DO SISTEMA")
    print("====================================================")

    print("Sistema operacional:", info["os"])
    print("CPU:", info["cpu_model"])
    print("Núcleos físicos:", info["cpu_physical_cores"])
    print("Núcleos lógicos:", info["cpu_logical_cores"])

    if "cpu_freq_mhz" in info:
        print("Frequência CPU:", info["cpu_freq_mhz"], "MHz")

    print("Memória RAM:", info["ram_total_gb"], "GB")

    if info["gpus"]:

        for gpu in info["gpus"]:
            print("GPU:", gpu["name"])
            print("VRAM:", gpu["memory_total_gb"], "GB")

    else:
        print("GPU: não detectada")

    print()


    # registrar no log
    if logger:

        logger.info("====================================================")
        logger.info("CONFIGURAÇÃO DO SISTEMA")
        logger.info("====================================================")

        logger.info(f"Sistema operacional: {info['os']}")
        logger.info(f"CPU: {info['cpu_model']}")
        logger.info(f"Núcleos físicos: {info['cpu_physical_cores']}")
        logger.info(f"Núcleos lógicos: {info['cpu_logical_cores']}")

        if "cpu_freq_mhz" in info:
            logger.info(f"Frequência CPU: {info['cpu_freq_mhz']} MHz")

        logger.info(f"Memória RAM: {info['ram_total_gb']} GB")

        if info["gpus"]:

            for gpu in info["gpus"]:
                logger.info(f"GPU: {gpu['name']}")
                logger.info(f"VRAM: {gpu['memory_total_gb']} GB")

        else:
            logger.info("GPU: não detectada")

        logger.info("")


class SystemMonitor:

    def __init__(self, interval=1):

        self.interval = interval
        self.cpu_usage = []
        self.ram_usage = []
        self.running = False
        self.thread = None

    def _monitor(self):

        while self.running:

            cpu = psutil.cpu_percent(interval=None)

            ram = psutil.virtual_memory().used / (1024 ** 3)

            self.cpu_usage.append(cpu)
            self.ram_usage.append(ram)

            time.sleep(self.interval)

    def start(self):

        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):

        self.running = False

        if self.thread:
            self.thread.join()

    def get_stats(self):

        if not self.cpu_usage:
            return None

        stats = {}

        stats["cpu_mean"] = sum(self.cpu_usage) / len(self.cpu_usage)
        stats["cpu_max"] = max(self.cpu_usage)

        stats["ram_mean"] = sum(self.ram_usage) / len(self.ram_usage)
        stats["ram_max"] = max(self.ram_usage)

        return stats


def print_and_log_monitor_results(stats, logger=None):

    if not stats:
        return

    print("\n====================================================")
    print("USO DE RECURSOS DURANTE O EXPERIMENTO")
    print("====================================================")

    print(f"CPU média: {stats['cpu_mean']:.2f}%")
    print(f"CPU máxima: {stats['cpu_max']:.2f}%")

    print(f"RAM média: {stats['ram_mean']:.2f} GB")
    print(f"RAM máxima: {stats['ram_max']:.2f} GB")

    print()

    if logger:

        logger.info("====================================================")
        logger.info("USO DE RECURSOS DURANTE O EXPERIMENTO")
        logger.info("====================================================")

        logger.info(f"CPU média: {stats['cpu_mean']:.2f}%")
        logger.info(f"CPU máxima: {stats['cpu_max']:.2f}%")

        logger.info(f"RAM média: {stats['ram_mean']:.2f} GB")
        logger.info(f"RAM máxima: {stats['ram_max']:.2f} GB")

        logger.info("")
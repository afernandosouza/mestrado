import numpy as np

T_VALUE_95 = 2.262  # t(0.975, 9)


def compute_statistics(acc_runs):

    acc_runs = np.array(acc_runs)

    mean = np.mean(acc_runs)

    std = np.std(acc_runs, ddof=1)

    margin = T_VALUE_95 * (std / np.sqrt(len(acc_runs)))

    ci_low = mean - margin
    ci_high = mean + margin

    return {
        "mean": mean,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high
    }
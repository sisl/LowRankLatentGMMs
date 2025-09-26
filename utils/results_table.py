# packages
import argparse
import json
import numpy as np
from tabulate import tabulate

parser = argparse.ArgumentParser()
parser.add_argument("--experiments", type=str, required=True, default="uci", choices=["uci", "images"])
parser.add_argument("--format", type=str, default="tabulate", choices=["tabulate", "latex"])
opt = parser.parse_args()

if opt.experiments == "uci":
    systems = ["HEPMASS", "MINIBOONE", "BSDS300"]
    metrics = ["epochs", "total train times", "log probs", "NFEs"]
    best_val_fun = [np.argmin, np.argmin, np.argmax, np.argmin]
    n_trial_per_system = [5, 5, 5]
elif opt.experiments == "uci":
    systems = ["fashion", "cifar10", "celeba"]
    metrics = ["epochs", "total train times", "NDB/C", "NFEs"]
    best_val_fun = [np.argmin, np.argmin, np.argmin, np.argmin]
    n_trial_per_system = [3, 3, 3]
else:
    raise ValueError("Not a recognized set of experiments.")


def calculate_mean_std(metric_values, correction, fmt):
    trimmed_values = np.array(np.trim_zeros(metric_values,"b"))
    if fmt == "f":
        if opt.format == "latex":
            return rf"{np.nanmean(correction*trimmed_values):.3f} \pm {np.nanstd(correction*trimmed_values):.3f}"
        else:
            return rf"{np.nanmean(correction*trimmed_values):.3f} +- {np.nanstd(correction*trimmed_values):.3f}" 
    elif fmt == "i":
        if opt.format == "latex":
            return rf"{int(np.nanmean(correction*trimmed_values)):d} \pm {int(np.nanstd(correction*trimmed_values)):d}"
        else:
            return rf"{int(np.nanmean(correction*trimmed_values)):d} +- {int(np.nanstd(correction*trimmed_values)):d}" 

metrics_correction = [1, 1/60, 1, 1]
metrics_format = ["f", "f", "f", "f"]
methods = ["VPCFM-MPPCA", "OTCFM-MPPCA", "VPCFM-Normal", "OTCFM-Normal"]

results_path = "./results/"
experiments = {s:{m:{} for m in methods} for s in systems}
for s in systems:
    for m in methods:
        path = results_path+s+"/"+m+"/results.json"
        try:
            experiments[s][m] = json.load(open(path, "r"))
        except:
            experiments[s][m] = None



for n_trials, e in zip(n_trial_per_system,experiments):
    print(f"Results for experiment: {e}")
    
    header_row = ["", ] + metrics
    result_rows = []
    
    if opt.format == "latex":
        for m in methods:
            if experiments[e][m] is not None:
                result_rows.append([""]+[m]+[calculate_mean_std(experiments[e][m][met][:n_trials], metrics_correction[i], metrics_format[i]) for i,met in enumerate(metrics)])
                
            else:
                result_rows.append([""]+[m]+["---" for met in metrics])

        vals = np.zeros((4,4))
        for j in range(2,6):
            for i in range(4):
                vals[i,j-2] = float(result_rows[i][j].split(r"\pm")[0])
        vals = np.array(vals)
        best_idx = []
        for i, bvf in enumerate(best_val_fun):
            bidx = bvf(vals[:,i])
            result_rows[bidx][i+2] = r"\bfseries "+result_rows[bidx][i+2]
        print(tabulate(result_rows, headers=header_row, tablefmt="latex_raw"))
        
    else:
        for m in methods:
            if experiments[e][m] is not None:
                result_rows.append([m]+[calculate_mean_std(experiments[e][m][met][:n_trials], metrics_correction[i], metrics_format[i]) for i,met in enumerate(metrics)])
            else:
                result_rows.append([m]+["None" for met in metrics])
        print(tabulate(result_rows, headers=header_row, tablefmt="grid"))
    
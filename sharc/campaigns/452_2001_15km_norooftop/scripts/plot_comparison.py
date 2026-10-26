import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

output_dir = r"C:\Users\josef\Documents\SHARC\sharc\campaigns\452_2001_15km_norooftop\output"
plot_dir = r"C:\Users\josef\.gemini\antigravity\brain\3ab96335-336c-4507-97ea-bdd0667dcdba"

models = ["P452", "P2001"]
p_values = ["p20", "p30", "p40"]

def get_latest_folder(model, p_val):
    pattern = os.path.join(output_dir, f"SatP_d15km_{model}_{p_val}_*")
    folders = glob.glob(pattern)
    folders = [f for f in folders if os.path.isdir(f)]
    if not folders:
        return None
    return sorted(folders)[-1]

plt.figure(figsize=(10, 6))
colors = {"P452": "blue", "P2001": "red"}
linestyles = {"p20": "-", "p30": "--", "p40": ":"}

for model in models:
    for p_val in p_values:
        folder = get_latest_folder(model, p_val)
        if not folder:
            continue
        file_path = os.path.join(folder, "imt_system_path_loss.csv")
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, header=None).values.flatten()
            data = np.sort(data)
            cdf = np.arange(1, len(data) + 1) / len(data)
            plt.plot(data, cdf, label=f"{model} {p_val}", color=colors[model], linestyle=linestyles[p_val])

plt.xlabel("Path Loss (dB)")
plt.ylabel("CDF")
plt.title("Path Loss Comparison (P.452 vs P.2001)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "path_loss_cdf.png"))
plt.close()

plt.figure(figsize=(10, 6))
for model in models:
    for p_val in p_values:
        folder = get_latest_folder(model, p_val)
        if not folder:
            continue
        file_path = os.path.join(folder, "system_inr.csv")
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, header=None).values.flatten()
            data = np.sort(data)
            cdf = np.arange(1, len(data) + 1) / len(data)
            plt.plot(data, cdf, label=f"{model} {p_val}", color=colors[model], linestyle=linestyles[p_val])

plt.xlabel("INR (dB)")
plt.ylabel("CDF")
plt.title("INR Comparison (P.452 vs P.2001)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "inr_cdf.png"))
plt.close()

print("Plots generated successfully.")

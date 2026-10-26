import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Caminho para os resultados
base_output_dir = r"C:\Users\josef\Documents\SHARC\sharc\campaigns\452_2001_15km\output"
out_plots_dir = os.path.join(base_output_dir, "comparison_plots")
os.makedirs(out_plots_dir, exist_ok=True)

# Cores e estilos para o plot
styles = {
    'P452': {'linestyle': '-', 'marker': ''},
    'P2001': {'linestyle': '--', 'marker': ''}
}
colors = {
    'p20': 'blue',
    'p30': 'orange',
    'p40': 'green'
}

plt.figure(figsize=(10, 7))

# Pegar as pastas mais recentes
latest_folders = {}
for folder_name in os.listdir(base_output_dir):
    folder_path = os.path.join(base_output_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue
        
    csv_file = os.path.join(folder_path, "imt_system_path_loss.csv")
    if not os.path.exists(csv_file):
        continue
        
    parts = folder_name.split('_')
    model = "Desconhecido"
    p_val = "Desconhecido"
    for part in parts:
        if part in ['P452', 'P2001']:
            model = part
        if part in ['p20', 'p30', 'p40']:
            p_val = part
            
    if model == "Desconhecido" or p_val == "Desconhecido":
        continue

    config_key = f"{model}_{p_val}"
    if config_key not in latest_folders or folder_name > latest_folders[config_key]:
        latest_folders[config_key] = folder_name

# Plotar
for config_key, folder_name in latest_folders.items():
    folder_path = os.path.join(base_output_dir, folder_name)
    csv_file = os.path.join(folder_path, "imt_system_path_loss.csv")
    
    parts = folder_name.split('_')
    for part in parts:
        if part in ['P452', 'P2001']:
            model = part
        if part in ['p20', 'p30', 'p40']:
            p_val = part

    df = pd.read_csv(csv_file)
    if 'samples' not in df.columns:
        continue
        
    pl_data = df['samples'].dropna().values
    pl_sorted = np.sort(pl_data)
    p_cdf = 1.0 * np.arange(len(pl_data)) / (len(pl_data) - 1)
    
    label_str = f"{model} ({p_val})"
    color = colors.get(p_val, 'black')
    style = styles.get(model, {}).get('linestyle', '-')
    
    plt.plot(pl_sorted, p_cdf, label=label_str, linewidth=2, color=color, linestyle=style)

plt.title('Path Loss a 15km: P.452 vs P.2001 (Sem Clutter/Terreno, 50 Snapshots)')
plt.xlabel('Path Loss entre Sistema e IMT (dB)')
plt.ylabel('CDF')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(loc='lower right')

out_file = os.path.join(out_plots_dir, "comparacao_path_loss_15km.png")
plt.savefig(out_file, bbox_inches='tight', dpi=150)
print(f"Gráfico salvo em: {out_file}")

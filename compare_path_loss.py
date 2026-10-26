import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base_dir = r"C:\Users\josef\Documents\SHARC\sharc\campaigns"
metsat_dir = os.path.join(base_dir, "MetSat")
p2001_dir = os.path.join(base_dir, "2001_clutter")
output_subdirs = ["output", "output_first", "output_dist"]
out_plots_dir = os.path.join(base_dir, "comparison_plots")

os.makedirs(out_plots_dir, exist_ok=True)

def get_latest_simulations(camp_dir, subdirs):
    sims = {}
    for sub in subdirs:
        d = os.path.join(camp_dir, sub)
        if not os.path.isdir(d):
            continue
        for folder in os.listdir(d):
            full_path = os.path.join(d, folder)
            if not os.path.isdir(full_path):
                continue
            
            # folder looks like SatP_d100km_clutter_one_2026-08-19_01
            # let's split by "_2026" or just use regex
            parts = folder.split("_202")
            if len(parts) >= 2:
                scenario = parts[0]
                date_str = "202" + parts[1]
                
                if scenario not in sims or date_str > sims[scenario]['date']:
                    sims[scenario] = {'path': full_path, 'date': date_str}
    return sims

metsat_sims = get_latest_simulations(metsat_dir, output_subdirs)
p2001_sims = get_latest_simulations(p2001_dir, output_subdirs)

common_scenarios = set(metsat_sims.keys()).intersection(set(p2001_sims.keys()))

print(f"Found {len(common_scenarios)} common scenarios for comparison.")

for scenario in sorted(common_scenarios):
    path_met = os.path.join(metsat_sims[scenario]['path'], "imt_system_path_loss.csv")
    path_2001 = os.path.join(p2001_sims[scenario]['path'], "imt_system_path_loss.csv")
    
    if not os.path.exists(path_met) or not os.path.exists(path_2001):
        print(f"Skipping {scenario}: CSV not found.")
        continue
        
    df_met = pd.read_csv(path_met)
    df_2001 = pd.read_csv(path_2001)
    
    pl_met = df_met['samples'].dropna().values
    pl_2001 = df_2001['samples'].dropna().values
    
    # Calculate CDF
    pl_met_sorted = np.sort(pl_met)
    p_met = 1. * np.arange(len(pl_met)) / (len(pl_met) - 1)
    
    pl_2001_sorted = np.sort(pl_2001)
    p_2001 = 1. * np.arange(len(pl_2001)) / (len(pl_2001) - 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'IMT-System Path Loss Comparison\nScenario: {scenario}', fontsize=14)
    
    # Plot CDF
    ax1.plot(pl_met_sorted, p_met, label='MetSat (P.452)', linewidth=2)
    ax1.plot(pl_2001_sorted, p_2001, label='2001_revisada (P.2001)', linewidth=2, linestyle='--')
    ax1.set_title('CDF (Cumulative Distribution Function)')
    ax1.set_xlabel('Path Loss (dB)')
    ax1.set_ylabel('CDF')
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.legend(loc='lower right')
    
    # Plot PDF
    bins = np.histogram_bin_edges(np.concatenate([pl_met, pl_2001]), bins=50)
    ax2.hist(pl_met, bins=bins, density=True, alpha=0.6, label='MetSat (P.452)')
    ax2.hist(pl_2001, bins=bins, density=True, alpha=0.6, label='2001_revisada (P.2001)')
    ax2.set_title('PDF (Probability Density Function)')
    ax2.set_xlabel('Path Loss (dB)')
    ax2.set_ylabel('Density')
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.legend(loc='upper right')
    
    out_file = os.path.join(out_plots_dir, f"{scenario}_path_loss_cdf_pdf.png")
    plt.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Saved plot for {scenario} -> {out_file}")

print(f"\nAll plots saved to {out_plots_dir}")

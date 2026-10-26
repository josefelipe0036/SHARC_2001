import pandas as pd
import os

base_dir = r"C:\Users\josef\Documents\SHARC\sharc\campaigns"
metsat_path = os.path.join(base_dir, "MetSat", "output", "SatP_d100km_clutter_one_2024-03-22_01", "imt_system_path_loss.csv")
p2001_path = os.path.join(base_dir, "2001_revisada", "output", "SatP_d100km_clutter_one_AZ_EL_2026-08-28_03", "imt_system_path_loss.csv")

# Let's find one folder that exists in both
metsat_dir = os.path.join(base_dir, "MetSat", "output")
p2001_dir = os.path.join(base_dir, "2001_revisada", "output")
if not os.path.exists(p2001_dir):
    p2001_dir = os.path.join(base_dir, "2001_revisada", "output_first")

print("P.452 Mean / Min / Max")
try:
    for folder in os.listdir(metsat_dir):
        if "SatP_d100km_clutter_one_AZ_EL" in folder:
            df = pd.read_csv(os.path.join(metsat_dir, folder, "imt_system_path_loss.csv"))
            print(f"P452: {df['samples'].mean():.2f} / {df['samples'].min():.2f} / {df['samples'].max():.2f}")
except Exception as e:
    pass

try:
    for folder in os.listdir(p2001_dir):
        if "SatP_d100km_clutter_one_AZ_EL" in folder:
            df = pd.read_csv(os.path.join(p2001_dir, folder, "imt_system_path_loss.csv"))
            print(f"P2001: {df['samples'].mean():.2f} / {df['samples'].min():.2f} / {df['samples'].max():.2f}")
except Exception as e:
    pass

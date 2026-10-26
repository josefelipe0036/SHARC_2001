import os
import glob
import yaml
import subprocess
import sys

dest_dir = r"C:\Users\josef\Documents\SHARC\sharc\campaigns\2001_clutter"

print("Fixing YAML output_dir...")
for f in glob.glob(os.path.join(dest_dir, "input", "*.yaml")):
    with open(f, 'r', encoding='utf8') as file:
        data = yaml.unsafe_load(file)
        
    if 'general' in data and 'output_dir' in data['general']:
        data['general']['output_dir'] = data['general']['output_dir'].replace("2001_95", "2001_clutter")
        
    with open(f, 'w', encoding='utf8') as file:
        yaml.dump(data, file, sort_keys=False, default_flow_style=False)

print(f"Re-running simulation...")
script_path = os.path.join(dest_dir, "scripts", "start_simulations_multi_thread.py")
subprocess.run([sys.executable, script_path], check=True)

print("Running graph generation...")
subprocess.run([sys.executable, r"C:\Users\josef\Documents\SHARC\compare_path_loss.py"], check=True)
subprocess.run([sys.executable, r"C:\Users\josef\Documents\SHARC\generate_detailed_pdf.py"], check=True)

print("All done!")

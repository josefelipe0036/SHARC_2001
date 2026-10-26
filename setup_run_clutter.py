import os
import shutil
import glob
import yaml
import subprocess
import sys

base_dir = r"C:\Users\josef\Documents\SHARC\sharc\campaigns"
src_dir = os.path.join(base_dir, "2001_95")
dest_dir = os.path.join(base_dir, "2001_clutter")

print(f"Creating new campaign folder: {dest_dir}")
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)
os.makedirs(dest_dir)
os.makedirs(os.path.join(dest_dir, "input"))
os.makedirs(os.path.join(dest_dir, "scripts"))
os.makedirs(os.path.join(dest_dir, "output"))

# Copy scripts and adjust them
for f in glob.glob(os.path.join(src_dir, "scripts", "*.py")):
    with open(f, 'r', encoding='utf8') as file:
        content = file.read()
    content = content.replace("2001_95", "2001_clutter")
    with open(os.path.join(dest_dir, "scripts", os.path.basename(f)), 'w', encoding='utf8') as file:
        file.write(content)

# Process YAML files
print("Processing YAML files...")
for f in glob.glob(os.path.join(src_dir, "input", "*.yaml")):
    with open(f, 'r', encoding='utf8') as file:
        data = yaml.unsafe_load(file)
        
    for link in data.get('link_collection', []):
        if 'param_p2001' in link:
            link['param_p2001']['p'] = 'RANDOM'
            link['param_p2001']['clutter_loss'] = True
            link['param_p2001']['below_rooftop'] = True

    dest_file = os.path.join(dest_dir, "input", os.path.basename(f))
    with open(dest_file, 'w', encoding='utf8') as file:
        yaml.dump(data, file, sort_keys=False, default_flow_style=False)

print(f"Setup complete. Starting simulations in multi-thread...")
script_path = os.path.join(dest_dir, "scripts", "start_simulations_multi_thread.py")
subprocess.run([sys.executable, script_path], check=True)
print("Simulation finished successfully!")

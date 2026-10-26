import os
import glob
import yaml

src_dir = r"C:\Users\josef\Documents\SHARC\sharc\campaigns\2001_95\input"
dest_dir = r"C:\Users\josef\Documents\SHARC\sharc\campaigns\2001_clutter\input"

os.makedirs(dest_dir, exist_ok=True)

for f in glob.glob(os.path.join(src_dir, "*.yaml")):
    with open(f, 'r', encoding='utf8') as file:
        data = yaml.unsafe_load(file)
        
    # Fix general output_dir
    if 'general' in data:
        if 'output_dir' in data['general']:
            data['general']['output_dir'] = data['general']['output_dir'].replace('2001_95', '2001_clutter')
            
    # Fix param_p2001
    for key in data:
        if isinstance(data[key], dict) and 'param_p2001' in data[key]:
            data[key]['param_p2001']['p'] = 'RANDOM'
            data[key]['param_p2001']['clutter_loss'] = True
            data[key]['param_p2001']['below_rooftop'] = True

    dest_file = os.path.join(dest_dir, os.path.basename(f))
    with open(dest_file, 'w', encoding='utf8') as file:
        yaml.dump(data, file, sort_keys=False, default_flow_style=False)

print("YAMLs corrected and saved to 2001_clutter/input")

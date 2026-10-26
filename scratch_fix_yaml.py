import os
import re
import glob

directory = r"C:\Users\josef\Documents\SHARC\sharc\campaigns\2001_95\input"
files = glob.glob(os.path.join(directory, "*.yaml"))

replacement_block = """  param_p2001:
    p: 0.95
    polarization: horizontal
    rx_lat: -15.47
    tx_lat: -15.46
    tx_height_agl: 18.0
    rx_height_agl: 7.0
  tx_power_density:"""

# Regex to match the block starting with param_p2001: and ending right before tx_power_density:
pattern = re.compile(r"  param_p2001:.*?\n  tx_power_density:", re.DOTALL)

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    new_content = pattern.sub(replacement_block, content)
    
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(new_content)

print(f"Fixed {len(files)} files.")

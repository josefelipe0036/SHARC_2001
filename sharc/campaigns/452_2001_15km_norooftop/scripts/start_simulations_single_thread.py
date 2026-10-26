import os
import subprocess

sharc_dir = r"C:\Users\josef\Documents\SHARC"
campaign_dir = os.path.join(sharc_dir, "sharc", "campaigns", "452_2001_15km_norooftop")
input_dir = os.path.join(campaign_dir, "input")

yaml_files = [
    "config_P2001_p20.yaml",
    "config_P2001_p30.yaml",
    "config_P2001_p40.yaml",
    "config_P452_p20.yaml",
    "config_P452_p30.yaml",
    "config_P452_p40.yaml"
]

env = os.environ.copy()
env["PYTHONPATH"] = sharc_dir

for f in yaml_files:
    file_path = os.path.join(input_dir, f)
    print(f"Running {f}...")
    subprocess.run(["python", os.path.join(sharc_dir, "sharc", "main_cli.py"), "-p", file_path], env=env)
    print(f"Finished {f}")

print("All simulations completed successfully.")

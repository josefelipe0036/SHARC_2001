import pandas as pd
import numpy as np

metsat_csv = r'C:\Users\josef\Documents\SHARC\sharc\campaigns\MetSat\output\SatP_d100km_clutter_one_2026-08-18_02\imt_system_path_loss.csv'
p2001_csv = r'C:\Users\josef\Documents\SHARC\sharc\campaigns\2001_95\output\SatP_d100km_clutter_one_2026-08-19_01\imt_system_path_loss.csv'

try:
    met = pd.read_csv(metsat_csv)['samples'].dropna()
    p2001 = pd.read_csv(p2001_csv)['samples'].dropna()
    print('MetSat (P.452) Mean:', met.mean(), 'Min:', met.min(), 'Max:', met.max())
    print('P2001 (P.2001) Mean:', p2001.mean(), 'Min:', p2001.min(), 'Max:', p2001.max())
except Exception as e:
    print(e)

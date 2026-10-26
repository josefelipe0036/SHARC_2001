import os
import glob
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

base_dir = r"C:\Users\josef\Documents\SHARC\sharc\campaigns"
metsat_dir = os.path.join(base_dir, "MetSat")
p2001_dir = os.path.join(base_dir, "2001_revisada")
plots_dir = os.path.join(base_dir, "comparison_plots")
output_subdirs = ["output", "output_first", "output_dist"]

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
            parts = folder.split("_202")
            if len(parts) >= 2:
                scenario = parts[0]
                date_str = "202" + parts[1]
                if scenario not in sims or date_str > sims[scenario]['date']:
                    sims[scenario] = {'path': full_path, 'date': date_str}
    return sims

metsat_sims = get_latest_simulations(metsat_dir, output_subdirs)
p2001_sims = get_latest_simulations(p2001_dir, output_subdirs)
common_scenarios = sorted(list(set(metsat_sims.keys()).intersection(set(p2001_sims.keys()))))

def get_yaml_params(scenario, camp_dir):
    yaml_path = os.path.join(camp_dir, "input", f"{scenario}.yaml")
    if not os.path.exists(yaml_path):
        return {}
    with open(yaml_path, 'r', encoding='utf8') as f:
        data = yaml.unsafe_load(f)
    params = {}
    
    # Try to extract interesting params
    try:
        if 'simulation' in data:
            params['snapshots'] = data['simulation'].get('snapshots', 'N/A')
        links = data.get('link_collection', [])
        if len(links) > 0:
            link = links[0]
            if 'station_a' in link:
                params['freq'] = link['station_a'].get('param_transmission', {}).get('frequency', 'N/A')
            
            p_model = link.get('param_p452') or link.get('param_p2001') or {}
            params['clutter_loss'] = p_model.get('clutter_loss', 'False')
            params['is_terrain'] = p_model.get('is_terrain', 'False')
            params['p_time'] = p_model.get('percentage_p', 'N/A')
            if 'p' in p_model: params['p_time'] = p_model['p']
    except Exception:
        pass
    return params

pdf_path = r"C:\Users\josef\Documents\SHARC\Resumo_Detalhado_SHARC.pdf"
with PdfPages(pdf_path) as pdf:
    # Cover Page
    fig = plt.figure(figsize=(12, 6.75))
    fig.text(0.1, 0.6, "Resultados Detalhados das Simulacoes\nITU-R P.452 (MetSat) vs P.2001 (2001_clutter)", fontsize=24, fontweight='bold')
    fig.text(0.1, 0.4, f"Total de Cenarios Analisados: {len(common_scenarios)}\nEste relatorio contem os parametros e resultados de cada simulacao lado a lado.", fontsize=16)
    plt.axis('off')
    pdf.savefig(fig)
    plt.close(fig)

    for scenario in common_scenarios:
        fig = plt.figure(figsize=(12, 8))
        
        # Title
        fig.text(0.05, 0.93, f"Cenario: {scenario}", fontsize=18, fontweight='bold')
        
        # Get Params
        p_met = get_yaml_params(scenario, metsat_dir)
        p_2001 = get_yaml_params(scenario, p2001_dir)
        
        param_text = (
            f"Parametros Extraidos (MetSat):\n"
            f" - Snapshots: {p_met.get('snapshots', 'N/A')}\n"
            f" - Frequencia: {p_met.get('freq', 'N/A')} MHz\n"
            f" - Clutter Loss: {p_met.get('clutter_loss', 'N/A')}\n"
            f" - Is Terrain: {p_met.get('is_terrain', 'N/A')}\n"
            f" - Prob. Tempo (p): {p_met.get('p_time', 'N/A')}\n"
        )
        fig.text(0.05, 0.78, param_text, fontsize=11, va='top', family='monospace')
        
        # Get Results
        path_met = os.path.join(metsat_sims[scenario]['path'], "imt_system_path_loss.csv")
        path_2001 = os.path.join(p2001_sims[scenario]['path'], "imt_system_path_loss.csv")
        
        mean_met, mean_2001 = "N/A", "N/A"
        try:
            df_met = pd.read_csv(path_met)['samples'].dropna()
            mean_met = f"{df_met.mean():.2f} dB (Min: {df_met.min():.2f} / Max: {df_met.max():.2f})"
        except: pass
        
        try:
            df_2001 = pd.read_csv(path_2001)['samples'].dropna()
            mean_2001 = f"{df_2001.mean():.2f} dB (Min: {df_2001.min():.2f} / Max: {df_2001.max():.2f})"
        except: pass
        
        res_text = (
            f"Resultados de Path Loss:\n"
            f" - MetSat (P.452)     : {mean_met}\n"
            f" - 2001_revisada (P.2001): {mean_2001}"
        )
        fig.text(0.45, 0.78, res_text, fontsize=12, va='top', fontweight='bold')
        
        # Plot Image
        img_path = os.path.join(plots_dir, f"{scenario}_path_loss_cdf_pdf.png")
        if os.path.exists(img_path):
            try:
                img = mpimg.imread(img_path)
                ax = fig.add_axes([0.05, 0.05, 0.9, 0.65])
                ax.imshow(img)
                ax.axis('off')
            except Exception as e:
                fig.text(0.5, 0.4, "Erro ao carregar imagem", ha='center')
        else:
            fig.text(0.5, 0.4, "Imagem nao encontrada", ha='center')
            
        pdf.savefig(fig)
        plt.close(fig)

print("PDF Detalhado Gerado em:", pdf_path)

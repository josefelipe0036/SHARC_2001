#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de validação para a implementação ITU-R P.2001-6 no SHARC.
Compara os resultados do motor local com os vetores de teste oficiais da ITU.
"""

import os
import numpy as np
import pandas as pd
import glob

# Importação do motor de cálculo do SHARC
try:
    from sharc.propagation.propagation_p2001_core import compute_P2001
except ImportError:
    print("Erro: Não foi possível encontrar 'sharc.propagation.propagation_p2001_core'.")
    print("Executa o script a partir da raiz do SHARC: python -m tests.validateP2001")
    exit(1)

# Configurações de tolerância e caminhos
tol = 1e-6
success = 0
total = 0

# Localização automática da pasta de exemplos (dentro de tests/validation_examples/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
test_profiles_dir = os.path.join(BASE_DIR, "validation_examples")

# Procura ficheiros de perfil (*_profile.csv)
search_pattern = os.path.join(test_profiles_dir, "*_profile.csv")
filenames = glob.glob(search_pattern)

if not filenames:
    print(f"Erro: Nenhum ficheiro encontrado em {test_profiles_dir}")
    print("Verifica se a pasta 'validation_examples' existe e contém os CSVs da ITU.")
    exit(1)

print(f"Iniciando validação técnica em: {test_profiles_dir}")

for filepath in filenames:
    filename = os.path.basename(filepath)
    results_file = filepath.replace("_profile.csv", "_results.csv")
    
    if not os.path.exists(results_file):
        print(f"Aviso: Resultados de referência não encontrados para {filename}. Ignorando.")
        continue

    # Carrega dados do terreno (ignora as 9 linhas de metadados da ITU)
    df_profile = pd.read_csv(filepath, skiprows=9, names=['d', 'h', 'z'])
    # Carrega os resultados oficiais da ITU para comparação
    df_results = pd.read_csv(results_file)

    print(f"\n>>> Processando: {filename}")
    GHz_prev = -1
    Lb_computed = []
    
    for i, row in df_results.iterrows():
        # Feedback de progresso por frequência
        if row.GHz > GHz_prev:
            print(f"    Frequência: {row.GHz} GHz | Porcentagem de Tempo: {row.Tpc}%")
            GHz_prev = row.GHz
        
        # Chamada ao motor compute_P2001 do SHARC
        # Passamos os arrays de perfil e os parâmetros da linha atual do teste
        res = compute_P2001(
            d_arr=df_profile.d.to_numpy(),
            h_arr=df_profile.h.to_numpy(),
            z_arr=df_profile.z.to_numpy(),
            f_ghz=row.GHz,
            p_pct=row.Tpc,
            rx_lat=row.Phirn,
            rx_lon=row.Phire,
            tx_lat=row.Phitn,
            tx_lon=row.Phite,
            rx_height_agl=row.Hrg,
            tx_height_agl=row.Htg,
            Gr_dbi=row.Grx,
            Gt_dbi=row.Gtx,
            polarization=int(row.FlagVp)
        )
        
        # Extraímos a perda básica de transmissão (Lb_dB)
        Lb_computed.append(res.Lb_dB)

    Lb_computed = np.array(Lb_computed)
    Lb_ref = df_results.Lb.to_numpy()
    
    # Diferença absoluta entre o calculado e a referência
    delta = np.abs(Lb_computed - Lb_ref)
    max_error = np.max(delta)

    if max_error < tol:
        print(f"  [SUCESSO] {filename}: Erro máximo {max_error:.2e} dB")
        success += 1
    else:
        print(f"  [FALHA] {filename}: Erro máximo {max_error:.2e} dB")
        idx_max = np.argmax(delta)
        print(f"    Maior erro no índice {idx_max}:")
        print(f"    - Calculado: {Lb_computed[idx_max]:.4f}")
        print(f"    - Referência: {Lb_ref[idx_max]:.4f}")

    total += 1

print("\n" + "="*60)
print(f"RESULTADO: {success} de {total} testes passaram.")
if success == total:
    print("CONFORMIDADE TOTAL: O motor matemático está idêntico ao padrão ITU-R P.2001-6.")
else:
    print("ALERTA: Existem divergências nos cálculos. Revisa a lógica do core.")
print("="*60)
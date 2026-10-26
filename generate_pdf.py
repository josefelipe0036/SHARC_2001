import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

# Define function to draw text on a blank figure
def draw_text_page(pdf, title, text_body):
    fig = plt.figure(figsize=(10, 5.625))  # 16:9 ratio
    fig.text(0.1, 0.85, title, fontsize=22, fontweight='bold', ha='left')
    fig.text(0.1, 0.75, text_body, fontsize=14, ha='left', va='top', wrap=True)
    plt.axis('off')
    pdf.savefig(fig)
    plt.close(fig)

with PdfPages(r"C:\Users\josef\Documents\SHARC\Resumo_SHARC_P452_vs_P2001.pdf") as pdf:
    # Slide 1: Title
    text1 = ("Objetivo: Analisar os resultados de perda de propagacao (Path Loss) entre as campanhas\n"
             "simuladas no SHARC para os modelos P.452 (MetSat) e P.2001 (2001_95)\n"
             "e explicar suas divergencias.\n\n"
             "Topicos Abordados:\n"
             "1. Resultados Observados\n"
             "2. Por que os resultados divergiram?\n"
             "3. Parametros Utilizados\n"
             "4. O que podemos fazer para alinhar e verificar os modelos?")
    draw_text_page(pdf, "Comparacao de Modelos de Propagacao:\nITU-R P.452 vs P.2001", text1)
    
    # Slide 2: Results
    fig2 = plt.figure(figsize=(10, 5.625))
    fig2.text(0.1, 0.85, "1. Resultados Observados", fontsize=22, fontweight='bold', ha='left')
    fig2.text(0.1, 0.75, "Diferenca Media Encontrada: ~57 dB\n- P.452: Media de ~221 dB de perda.\n- P.2001: Media de ~164 dB de perda.", fontsize=14, ha='left', va='top')
    try:
        img = mpimg.imread(r"C:\Users\josef\Documents\SHARC\sharc\campaigns\comparison_plots\SatP_d100km_clutter_one_path_loss_cdf_pdf.png")
        ax = fig2.add_axes([0.1, 0.1, 0.8, 0.5])
        ax.imshow(img)
        ax.axis('off')
    except Exception as e:
        fig2.text(0.1, 0.5, "Grafico nao encontrado", fontsize=14)
    pdf.savefig(fig2)
    plt.close(fig2)
    
    # Slide 3: Reasons
    text3 = ("A divergencia ocorre por diferencas na integracao dos modelos no SHARC:\n\n"
             "1. Clutter Loss (Obstaculos Visuais):\n"
             "O script integrador do P.452 soma as perdas de ambiente no final da conta.\n"
             "O integrador do P.2001 nao soma essa perda extra (~30-50 dB).\n\n"
             "2. Perfil de Terreno (Difracao):\n"
             "O SHARC gera elevacoes aleatorias apenas para a P.452 (se is_terrain: true).\n"
             "A P.2001 rodou em 'Terra Plana' sem as montanhas que causam atenuacao por difracao.\n\n"
             "3. Building Entry Loss:\n"
             "A atenuacao para estacoes indoor e ativamente somada no P.452, mas foi omitida no P.2001.")
    draw_text_page(pdf, "2. Por que eles divergiram?", text3)
    
    # Slide 4: Parameters
    text4 = ("A configuracao enviada pela interface/YAML foi a mesma (100km, 5000 snapshots, etc.),\n"
             "mas o SHARC repassou coisas diferentes para o motor base de cada modelo.\n\n"
             "- Probabilidade de Tempo (p): 0.95% (Igual nos dois)\n"
             "- Terreno: P.452 gerou dinamicamente | P.2001 usou plano (d_arr = None)\n"
             "- Clutter Loss: P.452 aplicou pelo P.2108 | P.2001 ignorou\n"
             "- Alturas de Antenas: 18m e 7m (Normal em ambos)")
    draw_text_page(pdf, "3. Parametros Utilizados", text4)
    
    # Slide 5: Action Plan
    text5 = ("Para comparar os dois modelos perfeitamente, precisamos alinhar os wrappers em\n"
             "Python para que o P.2001 receba os mesmos modificadores externos do SHARC.\n\n"
             "Plano de Acao:\n"
             "1. Alterar sharc/propagation/propagation_p2001.py para:\n"
             "   - Incluir a geracao aleatoria de perfil de terreno (reutilizando a logica do P.452).\n"
             "   - Instanciar e somar PropagationClutterLoss.\n"
             "   - Instanciar e somar PropagationBuildingEntryLoss para nos indoor.\n"
             "2. Rodar novamente a campanha 2001_95.\n"
             "3. Re-gerar a CDF/PDF. As curvas passarao a ser comparaveis.")
    draw_text_page(pdf, "4. O que fazer para alinhar e verificar?", text5)

print("PDF gerado!")

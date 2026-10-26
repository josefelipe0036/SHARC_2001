import os

base_dir = r"C:\Users\josef\Documents\SHARC"
compare_file = os.path.join(base_dir, "compare_path_loss.py")
pdf_file = os.path.join(base_dir, "generate_detailed_pdf.py")

for file in [compare_file, pdf_file]:
    with open(file, 'r', encoding='utf8') as f:
        content = f.read()
    # Replace the p2001 directory from 2001_95 to 2001_revisada
    content = content.replace('"2001_95"', '"2001_revisada"')
    # Also update the labels inside the files if there are any old labels
    content = content.replace('2001_95 (P.2001)', '2001_revisada (P.2001)')
    content = content.replace('2001_clutter (P.2001)', '2001_revisada (P.2001)')
    
    with open(file, 'w', encoding='utf8') as f:
        f.write(content)

print("Updated python files to use 2001_revisada")

import json, sys
sys.stdout.reconfigure(encoding='utf-8')
for nb_path in ['notebook/apriori.ipynb', 'notebook/apriori_vs_gnn_comparison.ipynb']:
    nb = json.load(open(nb_path, encoding='utf-8'))
    print(f'########## {nb_path} ##########')
    for i, c in enumerate(nb['cells']):
        src = ''.join(c['source'])
        ct = c['cell_type']
        print(f'[Cell {i} - {ct}]')
        print(src)
        print()

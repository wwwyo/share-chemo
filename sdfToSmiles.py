import pandas as pd
import numpy as np
from rdkit import Chem

# 1. sdfファイルの名前を指定
# ex. ChemCupid_Namiki.sdf
filename = 'ChemCupid_Namiki'

# -----------------------------------------


def main():
    mols = [mol for mol in Chem.SDMolSupplier(
        f'{filename}.sdf') if mol is not None]
    smiles = [Chem.MolToSmiles(mol) for mol in mols]
    df = pd.DataFrame({'SMILES': smiles})
    print(f'output to {filename}.csv')
    df.to_csv(f'{filename}.csv')


main()

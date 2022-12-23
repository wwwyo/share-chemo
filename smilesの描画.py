import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# 1. smilesファイルの名前を指定
# ex. ChemCupid_Namiki.csv
filename = './data/structure_creation3/smiles'

# 2. smilesが入っているカラム名を指定
column = 'SMILES'

# --------------------------------------------------------


def main():
    df = pd.read_csv(f'{filename}.csv', index_col=0)
    [draw(smiles) for smiles in df[column]]


def draw(smiles):
    print(smiles)
    mol = Chem.MolFromSmiles(smiles)
    display(Draw.MolToImage(mol, size=(150, 150)))


main()

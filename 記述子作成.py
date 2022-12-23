import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Avalon import pyAvalonTools
from mordred import Calculator, descriptors
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import rdMHFPFingerprint

BIT = 2048

# 1. 記述子に変換したいデータセットのファイル名
filename = 'dataset'

# 2. fingerprintの方法を選択する。*[]内のものを全て実行
finger_methods = [
    'morgan',
    'morgan_feature',
    'maccs',
    'rdkit',
    'minhash',
    'avalon',
    # 'atom', # 重い
    # 'donor', # 重い
    'mordred_2d',
    # 'mordred_3d',
]

# print(df.columns)
# fingerprintで変換するカラム名を選択
finger_columns = ['R1-', 'organocatalyst']


# ---------------------------------------------
def main():
    df = pd.read_csv(f'{filename}.csv', index_col=0)
    for finger_method in finger_methods:
        print(finger_method)
        _df = toFinger(df, finger_columns, finger_method)
        print(f'output to {finger_method}.csv')
        _df.to_csv(f'{finger_method}.csv')


def toFinger(df, columns, finger_method):
    df_copy = df.drop(columns=columns)
    for column in columns:
        fingerprints = toFingerFromSmiles(df[column], finger_method)
        if finger_method in ['mordred_2d', 'mordred_3d']:
            fingerprints_df = fingerprints.add_suffix(
                '_'+column).set_axis(df.index, axis='index')
        else:
            column_names = list(
                map(lambda x: str(x)+'_'+column, range(len(fingerprints[0]))))
            fingerprints_df = pd.DataFrame(
                fingerprints, index=df.index, columns=column_names)

        fingerprints_df = fingerprints_df[fingerprints_df.columns[~fingerprints_df.isnull(
        ).any()]]
        df_copy = pd.merge(df_copy, fingerprints_df,
                           left_index=True, right_index=True)
    return df_copy


def toFingerFromSmiles(series, method):
    encoder = rdMHFPFingerprint.MHFPEncoder()
    mols = []
    for smile in series:
        if smile in ['-', 0]:
            smile = ''
        mols.append(Chem.MolFromSmiles(smile))

    if (method == 'mordred_2d'):
        # scaling必要
        calc_2D = Calculator(descriptors, ignore_3D=True)  # 2D記述子
        fingerprints = calc_2D.pandas(mols, quiet=False)
        for column in fingerprints.columns:
            if fingerprints[column].dtypes == object:
                fingerprints[column] = fingerprints[column].values.astype(
                    np.float32)
        return fingerprints
    elif (method == 'mordred_3d'):
        # scaling必要
        calc_3D = Calculator(descriptors, ignore_3D=False)  # 3D記述子
        fingerprints = calc_3D.pandas(mols, quiet=False)
        for column in fingerprints.columns:
            if fingerprints[column].dtypes == object:
                fingerprints[column] = fingerprints[column].values.astype(
                    np.float32)
        return fingerprints
    else:
        fingerprints = []
        for mol_idx, mol in enumerate(mols):
            try:
                # listに直してる。
                if (method == 'morgan'):
                    fingerprint = [
                        x for x in AllChem.GetMorganFingerprintAsBitVect(mol, 2, BIT)]
                elif (method == 'morgan_feature'):
                    fingerprint = [x for x in AllChem.GetMorganFingerprintAsBitVect(
                        mol, 2, BIT, useFeatures=True)]
                elif (method == 'maccs'):
                    fingerprint = [
                        x for x in AllChem.GetMACCSKeysFingerprint(mol)]
                elif (method == 'rdkit'):
                    fingerprint = [x for x in Chem.RDKFingerprint(mol)]
                elif (method == 'minhash'):
                    # scaling必要
                    fingerprint = [x for x in encoder.EncodeMol(mol)]
                elif (method == 'avalon'):
                    fingerprint = [x for x in pyAvalonTools.GetAvalonFP(mol)]
                elif (method == 'atom'):
                    fingerprint = [
                        x for x in Pairs.GetAtomPairFingerprintAsBitVect(mol)]
                elif (method == 'donor'):
                    print('donor')
                    fingerprint = [x for x in GetBPFingerprint(mol)]
                else:
                    print('method error')
                fingerprints.append(fingerprint)
            except Exception as e:
                print(e)
                print("Error", mol_idx)
                break
        return fingerprints


main()

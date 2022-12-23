import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. スケーリングしたいデータセットを取得
# ex. dataset.csv
filename = 'dataset'

# 2. スケーリングしたいカラム名を設定
scaling_columns = ['organocatalyst(mol%)', 'temp(℃)', 'time(h)']

# ----------------------------------------------


def main():
    df = pd.read_csv(f'{filename}.csv', index_col=0)
    _df = scaling(df, scaling_columns)
    print(f'output to {filename}_scaled.csv')
    _df.to_csv(f'{filename}_scaled.csv')


def scaling(df, scaling_columns):
    scaler = StandardScaler()
    scaler.fit(df[scaling_columns])
    df_scaled = pd.concat([df.drop(columns=scaling_columns),
                           pd.DataFrame(scaler.transform(
                               df[scaling_columns]), index=df.index, columns=scaling_columns)
                           ], axis=1, join='inner')
    print('shape: ', df_scaled.shape)
    return df_scaled


main()

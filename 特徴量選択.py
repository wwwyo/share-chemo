import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

# 1. 特徴量選択したいデータセットのファイル名を設定
# ex. avalon.csv
filename = 'avalon'

# 2. borutaの特徴量選択を追加で行うか
# デフォルトでは分散と相関係数に基づいて特徴量選択を行う
isBoruta = True  # or False


# ---------------------------------------------------
def main():
    df = pd.read_csv(f'{filename}.csv', index_col=0)
    _df = removeNoImpactData(df)
    if (isBoruta):
        print('exec boruta')
        _df = boruta(df.drop(columns=['yield']), df['yield'])

    print(f'output to {filename}_selected.csv')
    _df.to_csv(f'{filename}_selected.csv')


def removeNoImpactData(df):
    print(df.head(5))
    X = df.drop(columns=['yield'])
    y = df['yield']
    # ベルヌーイ分布に基づく分散の閾値は0.16
    select = VarianceThreshold(threshold=(.8 * (1 - .8)))
    select.fit_transform(X)
    selected_columns = X.columns[select.get_support()]
    selected_X = pd.DataFrame(X[selected_columns], columns=selected_columns)

    # 相関が0.95以上のカラムを削除
    X_new = deleteHighCorrColumn(selected_X)

    df_new = pd.concat([
        y,
        X_new,
    ], axis=1)
    print('before_shape: ', df.shape)
    print('new_shape: ', df_new.shape)
    return df_new


def deleteHighCorrColumn(df):
    threshold = 0.95
    df_corr = df.corr()
    df_corr = abs(df_corr)
    columns = df_corr.columns

    # 対角線の値を0にする
    for i in range(0, len(columns)):
        df_corr.iloc[i, i] = 0

    while True:
        columns = df_corr.columns
        max_corr = 0.0
        query_column = None
        target_column = None

        df_max_column_value = df_corr.max()
        max_corr = df_max_column_value.max()
        query_column = df_max_column_value.idxmax()
        target_column = df_corr[query_column].idxmax()

        if max_corr < threshold:
            # しきい値を超えるものがなかったため終了
            break
        else:
            # しきい値を超えるものがあった場合
            delete_column = None
            saved_column = None

            # その他との相関の絶対値が大きい方を除去
            if sum(df_corr[query_column]) <= sum(df_corr[target_column]):
                delete_column = target_column
                saved_column = query_column
            else:
                delete_column = query_column
                saved_column = target_column

            # 除去すべき特徴を相関行列から消す（行、列）
            df_corr.drop([delete_column], axis=0, inplace=True)
            df_corr.drop([delete_column], axis=1, inplace=True)

    return df[df_corr.columns]


def boruta(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    corr_list = []
    for n in range(10000):
        shadow_features = np.random.rand(X_scaled.shape[0]).T
        corr = np.corrcoef(X_scaled, shadow_features, rowvar=False)[-1]
        corr = abs(corr[corr < 0.95])
        corr_list.append(corr.max())

    corr_array = np.array(corr_list)
    perc = 100 * (1-corr_array.max())

    # RandomForestRegressorでBorutaを実行
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
    feat_selector = BorutaPy(
        rf,
        n_estimators='auto',
        verbose=2,
        alpha=0.05,  # 有意水準
        max_iter=100,  # 試行回数
        perc=perc,  # ランダム生成変数の重要度の何％を基準とするか
        random_state=1
    )
    feat_selector.fit(X_scaled, y.values)

    # 選択された特徴量を確認
    selected = feat_selector.support_
    print('選択された特徴量の数: %d' % np.sum(selected))
    print(selected)
    print(X.columns[selected])

    # 選択した特徴量で学習
    X_selected = X[X.columns[selected]]
    return X_selected


main()

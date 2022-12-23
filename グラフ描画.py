import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = "DejaVu Serif"

# 1. 実測値のy
y_obs = pd.read_csv('y_obs.csv')

# 2. 予測したy
y_pred = pd.read_csv('y_pred.csv')


# 3. グラフのタイトル
title = 'yy plot'

# --------------------------------------------


def main(y_obs, y_pred, title):
    yvalues = np.concatenate([y_obs.values, y_pred.values])
    ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues), np.ptp(yvalues)

    plt.figure(figsize=(8, 8))
    plt.scatter(y_obs, y_pred)
    plt.plot([ymin - yrange * 0.01, ymax + yrange * 0.01],
             [ymin - yrange * 0.01, ymax + yrange * 0.01])
    plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.xlabel('Observed Y', fontsize=24)
    plt.ylabel('Predicted Y', fontsize=24)
    plt.title(title, fontsize=24)
    plt.tick_params(labelsize=16)
    plt.show()


main(y_obs, y_pred, title)

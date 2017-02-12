import numpy as np
from matplotlib import pyplot as plt
import math


def plot_imgs(imgs, steerings):
    plt.subplots_adjust()
    count = imgs.shape[0]
    n_col = 5
    n_row = math.ceil(count / n_col)
    n_row = 2 if n_row == 1 else n_row
    fig, ax = plt.subplots(n_row, n_col, figsize=[16, 3 * n_row/2] )
    for i in range(n_row):
        for j in range(n_col):
            ij = j + i * n_col
            if ij < count:
                ax[i,j].axis('off')
                ax[i,j].set_title(steerings[ij])
                ax[i,j].imshow(imgs[ij])
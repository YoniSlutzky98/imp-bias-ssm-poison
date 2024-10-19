import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

def plot_func(evals, stamps, title):
    font_big = 14
    font_small = 12
    n_evals = len(evals[0])

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    cmap = plt.get_cmap('viridis')
    colors = iter(cmap(np.linspace(0, 1, n_evals)))
    for i in range(n_evals):
        color = next(colors)
        ax.plot(stamps, evals[:, i], label=f'a_{i + 1}', color=color)
    ax.set_xlabel('Iteration', fontsize=font_big)
    ax.set_ylabel('Entry magnitude', fontsize=font_big)
    ax.set_title(title, fontsize=font_big, y=1.02)
    # ax.legend(fontsize=font_small)
    ax.grid(True, which='both', linewidth=0.5, color='0.875')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(3, 4))
    ax.tick_params(axis='both', which='major', labelsize=font_small)
    ax.xaxis.get_offset_text().set_size(font_small)
    plt.tight_layout()
    plt.show()

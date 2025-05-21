import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

def effective_rank(elements):
    elements = np.abs(elements)
    probs = elements/np.sum(elements)
    entropy = -np.sum(probs * np.log(probs, where=(probs > 0)))
    return np.exp(entropy)

def plot_func(evals, gammas, stamps, n_evals, title):
    font_big = 14
    font_small = 12

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

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    eranks = np.array([effective_rank(vector) for vector in evals])
    ax.plot(stamps, eranks, label=f'erank')
    ax.set_ylim(bottom=0, top=np.max(eranks) * 1.2)
    ax.set_xlabel('Iteration', fontsize=font_big)
    ax.set_ylabel('Effective rank', fontsize=font_big)
    ax.set_title(title, fontsize=font_big, y=1.02)
    # ax.legend(fontsize=font_small)
    ax.grid(True, which='both', linewidth=0.5, color='0.875')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(3, 4))
    ax.tick_params(axis='both', which='major', labelsize=font_small)
    ax.xaxis.get_offset_text().set_size(font_small)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    cmap = plt.get_cmap('viridis')
    d = 1
    colors = iter(cmap(np.linspace(0, 1, 1)))
    for i in range(d):
        color = next(colors)
        ax.plot(stamps, gammas[:, i], label=f'gamma_{i}', color=color)
    ax.set_xlabel('Iteration', fontsize=font_big)
    ax.set_ylabel('Gamma magnitude', fontsize=font_big)
    ax.set_title(title, fontsize=font_big, y=1.02)
    # ax.legend(fontsize=font_small)
    ax.grid(True, which='both', linewidth=0.5, color='0.875')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(3, 4))
    ax.tick_params(axis='both', which='major', labelsize=font_small)
    ax.xaxis.get_offset_text().set_size(font_small)
    plt.tight_layout()
    plt.show()
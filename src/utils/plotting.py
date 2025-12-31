import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm


def parallel_coordinates_objectives(Y):
    fig, axs = plt.subplots(1, 4, figsize=(12, 5), sharey=True, constrained_layout=True)
    colors = cm.viridis(np.linspace(0, 1, len(Y)))
    x = np.arange(4)

    plt.rcParams['font.family'] = 'Times New Roman'
    for i in range(4):
        ax = axs[i]
        order = np.argsort(-Y[f"f{i + 1}"].values)
        df_sorted = Y.iloc[order]

        for _, row in df_sorted.iterrows():
            ax.plot(x, row.values, color=cm.viridis(row[f"f{i + 1}"]), alpha=0.3)

        ax.set_title(rf"Ordered and Colored by $f_{i + 1}$", fontfamily="Times New Roman")

        ax.set_xticks(x)
        ax.set_xticklabels(['$f_1$', '$f_2$', '$f_3$', '$f_4$'])
    fig.suptitle(r"Parallel Coordinates Ordered and Colored by Each Objective ($f_1$–$f_4$)")

    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation='horizontal', shrink=0.4, pad=0.07, aspect=30)
    cbar.set_label('Normalized Objective Value')
    cbar.ax.tick_params(labelsize=12)
    # plt.show()
    return plt


def histogram_sensitivity(df_sobol_first, df_sobol_total, df_shapley, title):
    # df_sobol_first, df_sobol_total, df_shapley: pd.DataFrame.
    first, total, shapley = df_sobol_first.values, df_sobol_total.values, df_shapley.values
    var_names, obj_names = df_sobol_first.index.tolist(), df_sobol_first.columns.tolist()

    n_obj = len(obj_names)

    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axes = plt.subplots(1, n_obj, figsize=(12, 4), sharey=True, constrained_layout=True)

    if n_obj == 1:
        axes = [axes]

    x = np.arange(len(var_names))
    width = 0.25

    for j, ax in enumerate(axes):
        ax.bar(x - width, first[:, j], width, label='First-order')
        ax.bar(x, total[:, j], width, label='Total-order')
        ax.bar(x + width, shapley[:, j], width, label='Shapley')

        # ax.set_title(f"Sensitivity for {obj_names[j]}")
        ax.set_title(rf"Sensitivity for $f_{j + 1}$", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(['$x_1$', '$x_2$', '$x_3$', '$x_4$', '$x_5$', '$x_6$'])
        ax.set_ylim(0, 1.1)

        if j == n_obj - 1:
            ax.legend(fontsize=14)

        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both')

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(title)
    # plt.show()
    return plt


def scatter_preference_vs_classical(F_classical, results):
    import matplotlib as mpl
    obj_names = ['f1', 'f2', 'f3', 'f4']

    f_raw = np.array([res['f_raw'] for res in results])
    f_raw_mat = np.array([[float(d[k]) for k in obj_names] for d in f_raw])

    g_key = np.round(f_raw_mat, decimals=3)
    tuples = [tuple(row) for row in g_key]
    unique, counts = np.unique(tuples, axis=0, return_counts=True)
    probs_map = {tuple(u): c / len(f_raw_mat) for u, c in zip(unique, counts)}
    probs = np.array([probs_map[tuple(row)] for row in g_key])

    order = np.argsort(probs)
    f_plot = f_raw_mat[order]
    p_plot = probs[order]

    norm = mpl.colors.Normalize(vmin=p_plot.min(), vmax=p_plot.max())
    cmap = "viridis"
    mpl.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.family'] = 'Times New Roman'

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # f1 vs f2
    axes[0].scatter(F_classical[:, 0], F_classical[:, 1], color='grey', alpha=0.3, s=5)
    axes[0].scatter(f_plot[:, 0], f_plot[:, 1], c=p_plot, cmap=cmap, norm=norm, s=70, marker="*")
    axes[0].set_xlabel("$f_1$")
    axes[0].set_ylabel("$f_2$")
    axes[0].set_title("$f_1 \ $vs$ \ f_2$")
    axes[0].tick_params(axis='x', labelsize=12)

    # f2 vs f3
    axes[1].scatter(F_classical[:, 1], F_classical[:, 2], color='grey', alpha=0.5, s=5)
    axes[1].scatter(f_plot[:, 1], f_plot[:, 2], c=p_plot, cmap=cmap, norm=norm, s=70, marker="*")
    axes[1].set_xlabel("$f_2$")
    axes[1].set_ylabel("$f_3$")
    axes[1].set_title("$f_2 \ $vs$ \ f_3$")
    axes[1].tick_params(axis='x', labelsize=12)

    # f3 vs f4
    axes[2].scatter(F_classical[:, 2], F_classical[:, 3], color='grey', alpha=0.5, s=5, label="Classical")
    axes[2].scatter(f_plot[:, 2], f_plot[:, 3], c=p_plot, cmap=cmap, norm=norm, s=70, marker="*", label="Preference")
    axes[2].set_xlabel("$f_3$")
    axes[2].set_ylabel("$f_4$")
    axes[2].set_title("$f_3 \ $vs$ \ f_4$")
    axes[2].legend()
    axes[2].tick_params(axis='x', labelsize=12)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, label="Probability")
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.15, top=0.82, wspace=0.35)

    plt.suptitle("Comparison of Pareto-optimal objective vectors: Classical vs Preference-aware")
    return plt
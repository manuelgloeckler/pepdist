import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import numpy as np

def remove_equals(scores, exact=1.0):
    """ Removes equal matches  """
    while True:
        try:
            scores.remove(exact)
        except BaseException:
            break
    return scores

def hist(immu, non_immu, ax=plt, density=1, bins=20):
    """Histogram of two datasets """
    ax.hist([immu, non_immu], color=['dodgerblue', 'orange'], edgecolor='black',
            bins=bins, density=density, label=["Immunogenic", "Non-Immunogenic"])
    ax.legend()
    ax.xlabel('Score')
    if density == 1:
        ax.ylabel('Normalized Frequencies')
    else:
        ax.ylabel('Frequencies')


def distplot(immu, non_immu, ax=plt, equals=False):
    if not equals:
        immu = remove_equals(immu)
        non_immu = remove_equals(non_immu)

    sns.distplot(immu, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label="Immunogenic", ax=ax)
    sns.distplot(non_immu, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label="Non-Immunogenic", ax=ax)
    ax.set(xlabel='Score', ylabel='Density')


def cum_distplot(immu, non_immu, ax=plt, equals=False, bins=100, density=1):
    """ Cumulative Density Plot """
    if not equals:
        immu = remove_equals(immu)
        non_immu = remove_equals(non_immu)

    counts1, bin_edges1 = np.histogram(immu, bins=bins, density=density)
    cdf1 = np.cumsum(counts1)
    l1, = ax.plot(bin_edges1[1:], cdf1 / cdf1[-1])
    counts2, bin_edges2 = np.histogram(non_immu, bins=bins, density=density)
    cdf2 = np.cumsum(counts2)
    l2, = ax.plot(bin_edges2[1:], cdf2 / cdf2[-1])
    ax.legend((l1, l2), ("Immunogenic", "Non-immunogenic"))
    ax.set(xlabel='Score', ylabel='Probability')


def plot_all(
    immu,
    non_immu,
    plot,
    size=(
        3,
        3),
        figsize=(
            20,
            10),
    titles=[
        "A01:01",
        "A02:01",
        "A03:01",
        "A11:01",
        "A24:02",
        "B07:02",
        "B15:01",
        "B44:02",
        "B49:01"]):
    f, axes = plt.subplots(size[0], size[1], figsize=figsize)
    f.subplots_adjust(
        left=0.125,
        bottom=0.1,
        right=0.9,
        top=0.9,
        wspace=0.2,
        hspace=0.8)
    i = 0
    for axe in axes:
        for ax in axe:
            if i > len(immu) - 1:
                break
            ax.title.set_text(titles[i] +
                              "\n" +
                              str(len(remove_equals(immu[i]))) +
                              " immunogenic and " +
                              str(len(remove_equals(non_immu[i]))) +
                              " non-immunogenic Peptides")
            plot(immu[i], non_immu[i], ax=ax)
            i += 1


def ks_test(immu, non_immu):
    """ ks test """
    statistic, p_val = scipy.stats.ks_2samp(immu, non_immu)
    return p_val

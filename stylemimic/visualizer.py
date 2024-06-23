#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:48:58 2017

@author: Amine Laghaout
"""

import pandas as pd
import os
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import numpy as np


def plot_tsne(
    data,
    features,
    target,
    n_components=2,
    verbose=0,
    random_state=0,
    nrows=None,
    title="",
    filename="t-SNE.pdf",
):
    from sklearn.manifold import TSNE

    x = data[features].iloc[:nrows]
    y = data[target].iloc[:nrows]

    tsne = TSNE(
        n_components=n_components, verbose=verbose, random_state=random_state
    )
    z = tsne.fit_transform(x)

    df = pd.DataFrame({"y": y, "Component 1": z[:, 0], "Component 2": z[:, 1]})

    sns.scatterplot(
        x="Component 1",
        y="Component 2",
        hue=df.y.tolist(),
        palette=sns.color_palette("hls", len(set(y))),
        data=df,
    ).set(title=title)
    # print('KL-divergence:', tsne.kl_divergence_)

    plt.savefig(filename)

    return tsne


def plot_correlation_matrix(
    corr, dir_path=".", filename="correlation_matrix.pdf"
):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )

    if filename is not None:
        plt.savefig(os.path.join(dir_path, filename), bbox_inches="tight")


def plot_pairgrid(
    data,
    dir_path=".",
    filename="pairgrid.pdf",
    hue=None,
    alpha=1,
    annotate_distance_corr=True,
):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.rcParams["axes.labelsize"] = 18
    g = sns.PairGrid(
        data,
        diag_sharey=False,
        # TODO: Derive a boolean from the overall label.
        hue=hue,
    )

    # Rotate the labels.
    for ax in g.axes.flatten():
        # rotate x axis labels
        ax.set_xlabel(ax.get_xlabel(), rotation=45)
        # rotate y axis labels
        ax.set_ylabel(ax.get_ylabel(), rotation=0)
        # set y labels alignment
        ax.yaxis.get_label().set_horizontalalignment("right")
        ax.xaxis.get_label().set_horizontalalignment("right")

    g.map_lower(plt.scatter, linewidths=1, edgecolor="w", s=90, alpha=alpha)
    if annotate_distance_corr:
        g.map_lower(corrfunc)
    g.map_diag(sns.histplot, kde=True, lw=4, legend=False)
    g.map_upper(sns.kdeplot, cmap="Blues_d", warn_singular=False)
    # g.map_upper(sns.histplot, cumulative=True, legend=False)
    g.add_legend()

    if filename is not None:
        plt.savefig(os.path.join(dir_path, filename), bbox_inches="tight")


def plot_correlations(corr, dir_path=".", filename="correlations.pdf"):
    try:
        from sequana.viz import corrplot
    except BaseException:
        print("WARNING: Could not import sequana.viz.")
        return None

    c = corrplot.Corrplot(corr)
    c.plot(
        method="ellipse",
        cmap="PRGn_r",
        shrink=1,
        rotation=45,
        upper="text",
        lower="ellipse",
    )
    fig = plt.gcf()
    fig.set_size_inches(10, 8)

    if filename is not None:
        plt.savefig(os.path.join(dir_path, filename), bbox_inches="tight")


def plot_time_series(
    x,
    y_dict,
    fontsize=16,
    markersize=3,
    xlabel="",
    ylabel="",
    loc="upper left",
    bbox_to_anchor=(1, 1),
    title=None,
    linewidth=3,
    xtick_frequency=10,
    rotation=45,
    save_as=None,
    adjust_xticks=True,
    log=(False, False),
    legend=True,
):
    """Plot the data stored in the dictionary ``y_dict`` versus ``x``."""

    plt.figure()

    for y_key in y_dict.keys():
        plt.plot(
            x,
            y_dict[y_key],
            label=y_key,
            marker="o",
            markersize=markersize,
            linewidth=linewidth,
        )
        plt.xlabel(r"%s" % xlabel, fontsize=fontsize)
        plt.ylabel(r"%s" % ylabel, fontsize=fontsize)
        plt.setp(plt.xticks()[1], rotation=rotation)
        #        ax = plt.gca()

        # Use logarithmic scale?
        if log[0]:
            plt.xscale("log")  # , nonposy = 'clip'
        if log[1]:
            plt.yscale("log")  # , nonposy = 'clip'

            # if len(x) > xtick_frequency and adjust_xticks:
            #     ticks_indices = np.arange(0, len(x),
            #                               int(len(x)/xtick_frequency))
            #     plt.xticks(ticks_indices)
            #     try:
            #         ax.set_xticklabels(x[ticks_indices])
            #     except Exception:
            #         pass

        if legend:
            plt.legend(
                loc=loc, bbox_to_anchor=bbox_to_anchor, fontsize=fontsize
            )

        if title is not None:
            plt.title(r"%s" % title, fontsize=fontsize)

    plt.grid()

    if save_as is not None:
        plt.savefig(save_as, bbox_inches="tight")

    plt.show()


def plot_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title="Confusion matrix",
    cmap=None,
    save_as=None,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.figure()

    if cmap is None:
        cmap = plt.cm.Blues

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    if save_as is not None:
        plt.savefig(save_as, bbox_inches="tight")

    plt.show()
    plt.clf()


def dist_corr(X, Y, pval=True, nruns=2000):
    """Distance correlation with p-value from bootstrapping"""
    import dcor

    dc = dcor.distance_correlation(X, Y)
    pv = dcor.independence.distance_covariance_test(
        X, Y, exponent=1.0, num_resamples=nruns
    )[0]
    if pval:
        return (dc, pv)
    else:
        return dc


def corrfunc(x, y, **kws):
    d, p = dist_corr(x, y)
    # print("{:.4f}".format(d), "{:.4f}".format(p))
    if p > 0.1:
        pclr = "Darkgray"
    else:
        pclr = "Darkblue"
    ax = plt.gca()
    ax.annotate(
        "DC = {:.2f}".format(d),
        xy=(0.1, 0.99),
        xycoords=ax.transAxes,
        color=pclr,
        fontsize=14,
    )


def AUC_ROC_curve(actual, predicted, save_as=None):
    """
    Plot the ROC curve and return the AUC.

    Parameters
    ----------
    actual : str
        Actual labels
    predicted : str
        Predicted labels
    save_as : str
        Pathname where to store the ROC curve.

    Returns
    -------
    AUC : float
        Area Under the Curve
    """

    import sklearn as skl
    from sklearn.metrics import RocCurveDisplay

    # Plot the ROC curve.
    RocCurveDisplay.from_predictions(actual, predicted)
    if save_as is not None:
        plt.savefig(save_as)
    plt.show()
    plt.close()

    # Comput the AUC.
    AUC = skl.metrics.roc_auc_score(actual, predicted)

    return AUC


def plot2D(
    x,
    y,
    linewidth=3,
    show=True,
    marker=None,
    legend=None,
    xlabel="",
    ylabel="",
    title="",
    fontsize=16,
    smooth=None,
    save_as=None,
    axis_range=None,
    grid=True,
    log=(False, False),
    markersize=None,
    xticks=None,
    drawstyle=None,
    fill_between=False,
):
    from scipy.signal import savgol_filter

    x = x
    y = y

    if not show:
        plt.ioff()

    plt.figure()

    # Font
    # plt.rc('font', family = fontFamily)

    # Plot several curves or...
    if isinstance(y, tuple):
        if legend in ["", None]:
            legend = [""] * len(y)

        for y_element_index, y_element in enumerate(y):
            plt.plot(
                x,
                y_element,
                label=legend[y_element_index],
                marker=marker,
                markersize=markersize,
                drawstyle=drawstyle,
                linewidth=linewidth,
            )

    # ... a single curve?
    else:
        plt.plot(
            x,
            y,
            label=legend,
            marker=marker,
            linewidth=linewidth,
            markersize=markersize,
            drawstyle=drawstyle,
        )

        # Superimpose a smoothed line?
        if smooth is not None:
            yhat = savgol_filter(y, smooth[0], smooth[1])
            plt.plot(x, yhat, color="red")

        if fill_between:
            plt.fill_between(x, 0, y, step="mid")

    # Use logarithmic scale?
    if log[0]:
        plt.xscale("log")  # , nonposy = 'clip'
    if log[1]:
        plt.yscale("log")  # , nonposy = 'clip'

    # Labels, legend, and title
    plt.xlabel(r"%s" % xlabel, fontsize=fontsize)
    plt.ylabel(r"%s" % ylabel, fontsize=fontsize)
    plt.title(r"%s" % title, fontsize=fontsize)
    if legend is not None and legend:
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=fontsize)

    # Abcissa labels
    if xticks is not None:
        if not isinstance(xticks, tuple):
            xticks = (xticks, "horizontal")
        plt.xticks(x, xticks[0], rotation=xticks[1])

    # Axis
    if axis_range is None and not isinstance(y, tuple):
        axis_range = (min(x), max(x), min(y), max(y))

    elif axis_range is None and isinstance(y, tuple):
        min_y = np.inf
        max_y = -np.inf

        for y_element in y:
            # print('>>> y_element:', y_element, 'y:', y)  # TODO remove

            if min(y_element) < min_y:
                min_y = min(y_element)
            if max(y_element) > max_y:
                max_y = max(y_element)

        axis_range = (min(x), max(x), min_y, max_y)

    plt.axis(axis_range)

    # Grid
    if grid is not False:
        if grid is True:
            plt.grid()
        else:
            plt.grid(**grid)

    if save_as is not None:
        plt.savefig(save_as, bbox_inches="tight")

    if show:
        plt.show()

    plt.clf()


# %%


def read(data):
    data = pd.read_csv(data, delimiter="\t", nrows=1000)

    columns = data.columns
    desc = data.describe()
    print(data.head())

    return desc, columns, data

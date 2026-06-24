"""
plot.py — reusable plotting functions for vocalization analysis
(spontaneous + playback)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from . import utils


def _despine(ax):
    ax.spines[['right', 'top']].set_visible(False)


def _save(fig, savedir, filename):
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        fig.savefig(os.path.join(savedir, filename), transparent=True, bbox_inches='tight', dpi=500)


# ---------------------------------------------------------------------------
# 1. bar_treat_anim
#    Summary barplot: group-level bar (left) + per-animal bars (right)
#    Used for: prop trials responded, call rate by treatment
# ---------------------------------------------------------------------------

def bar_treat_anim(
    df,
    y,
    hue_palette,                     # dict: {hue_val: color}
    ylabel,
    savedir=None,
    filename='bar_treat_anim.png',
    x_anim='anim',
    x_treat='treatment',
    figsize=(3, 2.5),
    width_ratios=(1.5, 3),
    wspace=0.15,
    estimator='mean',
    errorbar='se',
    legend_bbox=None,
    legend_title=None,
    suptitle=None,
):
    """
    Two-panel barplot: left panel shows group-level summary per treatment,
    right panel shows per-animal values dodged by treatment.
    """
    hue = x_treat
    palette = hue_palette
    ax0_ylabel = f'mean {ylabel}' if estimator == 'mean' else ylabel

    fig, axes = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={'width_ratios': list(width_ratios), 'wspace': wspace},
        sharey=True,
    )

    sns.barplot(df, x=x_treat, y=y,
                hue=x_treat, palette=palette,
                estimator=estimator, errorbar=errorbar, width=0.5,
                ax=axes[0])

    sns.barplot(df, x=x_anim, y=y,
                hue=hue, palette=palette,
                estimator=estimator, dodge=True, errorbar=errorbar, width=0.5,
                ax=axes[1])

    axes[0].set_xlabel('all')
    axes[0].set_xticklabels('')
    axes[0].set_ylabel(ax0_ylabel)

    axes[1].set_xlabel('')
    axes[1].set_ylabel('')

    for ax in axes:
        _despine(ax)

    axes[1].legend_.remove()
    legend_kw = {'bbox_to_anchor': legend_bbox} if legend_bbox is not None else {}
    fig.legend(loc='upper right', frameon=False, title=legend_title, **legend_kw)

    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.tight_layout()
    _save(fig, savedir, filename)
    return fig, axes


# ---------------------------------------------------------------------------
# 2. strip_box_anim
#    Per-recording/trial strip + boxplot, dodged by treatment, per animal
#    Used for: call rate by recording
# ---------------------------------------------------------------------------

def strip_box_anim(
    df,
    y,
    hue_palette,                     # dict: {hue_val: color}
    ylabel,
    savedir=None,
    filename='strip_box_anim.png',
    x='anim',
    hue='treatment',
    figsize=(3.5, 2.5),
    strip_size=3,
    box_width=0.3,
    legend_bbox=None,
    legend_title=None,
    suptitle=None,
):
    """
    Strip + box plot per animal, dodged by treatment.
    Stripplot shows individual recording/trial values; boxplot shows
    median, IQR, and 1.5×IQR whiskers.
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.stripplot(df, x=x, y=y,
                  hue=hue, palette=hue_palette,
                  dodge=True, size=strip_size, jitter=False,
                  legend=False, ax=ax)

    sns.boxplot(df, x=x, y=y,
                hue=hue, palette=hue_palette,
                dodge=True, width=box_width,
                fliersize=0,          # outliers already shown by stripplot
                ax=ax)

    ax.set_ylabel(ylabel)
    ax.legend_.remove()
    _despine(ax)

    legend_kw = {'bbox_to_anchor': legend_bbox} if legend_bbox is not None else {}
    fig.legend(loc='upper right', frameon=False, title=legend_title, **legend_kw)

    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.tight_layout()
    _save(fig, savedir, filename)
    return fig, ax


# ---------------------------------------------------------------------------
# 3. strip_box
#    Per-recording/trial strip + boxplot split by treatment (no per-animal x).
# ---------------------------------------------------------------------------

def strip_box(
    df,
    y,
    hue_palette,                     # dict: {hue_val: color}
    ylabel,
    savedir=None,
    filename='strip_box.png',
    x='treatment',
    figsize=(2.5, 2.5),
    strip_size=3,
    strip_alpha=0.7,
    jitter=True,
    # jitter_width=0.2,
    dodge=False,
    box_width=0.4,
    box_alpha=0.4,
    box_fill=False,
    legend_bbox=None,
    legend_title=None,
    suptitle=None,
    ax=None,
):
    """
    Strip + box plot with treatment on the x-axis (no per-animal split).
    Stripplot shows individual recording/trial values; boxplot shows
    median, IQR, and 1.5×IQR whiskers.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.stripplot(df, x=x, y=y,
                  hue=x, palette=hue_palette,
                  size=strip_size, alpha=strip_alpha,
                  jitter=jitter,
                  legend=False, ax=ax)

    sns.boxplot(df, x=x, y=y,
                hue=x, palette=hue_palette,
                width=box_width,
                dodge=dodge,
                fliersize=0,
                fill=box_fill,
                ax=ax)
    
    for patch in ax.patches:
        patch.set_alpha(box_alpha)

    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    if ax.get_legend():
        ax.get_legend().remove()
    _despine(ax)

    legend_kw = {'bbox_to_anchor': legend_bbox} if legend_bbox is not None else {}
    fig.legend(loc='upper right', frameon=False, title=legend_title, **legend_kw)

    if own_fig:
        if suptitle is not None:
            fig.suptitle(suptitle)
        fig.tight_layout()
        _save(fig, savedir, filename)
    return fig, ax


# ---------------------------------------------------------------------------
# 3. line_scatter_sess
#    Per-animal lines across sorted sessions + colored scatter + optional
#    polynomial regression over the group.
#    Used for: call rate / prop responded across sessions
# ---------------------------------------------------------------------------

def line_scatter_sess(
    df,
    y,
    hue_palette,                     # dict: {hue_val: color}
    ylabel,
    savedir=None,
    filename='line_scatter_sess.png',
    x='sess_n_sort',
    hue='treatment',
    unit='anim',
    reg=True,
    reg_order=3,
    xtick_labels=('S-1', 'M-1', 'S-2', 'M-2'),
    figsize=(3.5, 2.4),
    legend_bbox=None,
    legend_title=None,
    title=None,
    suptitle=None,
):
    """
    Lines connecting each animal's sessions (gray, one style per animal),
    scatter points colored by treatment, and optional polynomial regression
    over the full group.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # gray per-animal lines
    sns.lineplot(df, x=x, y=y,
                 units=unit, estimator=None,
                 color='gray', style=unit,
                 alpha=0.7, linewidth=1.5, legend=True,
                 ax=ax)

    # treatment-colored scatter
    sns.scatterplot(df, x=x, y=y,
                    hue=hue, palette=hue_palette,
                    s=50, alpha=0.7, legend=False,
                    ax=ax)

    # optional polynomial regression over the full group
    if reg:
        sns.regplot(df, x=x, y=y,
                    color='cornflowerblue',
                    line_kws={'linewidth': 1.7},
                    order=reg_order, label='reg.', ci=None,
                    scatter=False,
                    ax=ax)

    ax.set_xlabel('session [sorted]')
    ax.set_xticks(range(1, len(xtick_labels) + 1))
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel(ylabel)

    legend_kw = {'bbox_to_anchor': legend_bbox} if legend_bbox is not None else {'bbox_to_anchor': (1.1, 1)}
    ax.legend(loc='upper left', frameon=False, title=legend_title, **legend_kw)
    _despine(ax)

    if title is not None:
        ax.set_title(title)

    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.tight_layout()
    _save(fig, savedir, filename)
    return fig, ax


# ---------------------------------------------------------------------------
# 4. hist_hue
#    Overlapping histograms by hue (e.g. treatment), optionally split into
#    subplots by a second column (e.g. condition, call type).
#    Used for: duration, ICI, spectral parameter distributions.
#
#    title behaviour:
#      - single panel  → ax.set_title(title)
#      - multi-panel   → fig.suptitle(title); each panel gets its own title
#                        from split_by values (or split_labels)
# ---------------------------------------------------------------------------

def hist_hue(
    df,
    x,
    hue_palette,                     # dict: {hue_val: color}
    xlabel,
    ylabel=None,                     # default: 'density' if density=True, 'count' if False
    savedir=None,
    filename='hist_hue.png',
    hue='treatment',
    split_by=None,                   # column for subplots; None → single axis
    split_labels=None,               # dict: {val: label} for subplot titles
    split_order=None,                # list to control subplot order
    log=False,                       # log10-transform x after scaling
    binsize=None,                    # default: 1.0 (linear) or 0.1 (log)
    scale=1.0,                       # multiply x values before plotting (e.g. 1000 for s→ms)
    tick_vals=None,                  # explicit tick positions in display units (after scale, before log)
    alpha=0.7,
    histtype='bar',                  # 'bar', 'step', 'stepfilled'
    density=True,
    density_norm='hue',              # 'hue': each group sums to 1; 'all': all groups sum to 1
    xlim=None,
    sharey=True,
    figsize=None,
    title=None,                      # single plot: ax.set_title; multi-panel: fig.suptitle
    legend_bbox=None,
    legend_title=None,
):
    """
    Overlapping histograms of `x` coloured by `hue`, optionally split into
    side-by-side subplots by a second column (e.g. condition, call type).

    When split_by is None, a single axis is produced and `title` is set as a
    regular ax title.  When split_by is set, each panel gets its own title
    from the split values (or split_labels dict), and `title` becomes the
    figure suptitle.

    When log=True, values are log10-transformed. Tick labels are
    back-transformed to display units via FuncFormatter so the axis remains
    readable. Use tick_vals (in display units, i.e. after scale) to pin ticks
    to meaningful values, e.g. tick_vals=[0.3, 1, 3, 10] for ms durations.
    """
    if ylabel is None:
        ylabel = 'count' if (not density and density_norm == 'hue') else 'density'
    if binsize is None:
        binsize = 0.1 if log else 1.0

    if not isinstance(hue_palette, dict):
        if hasattr(df[hue], 'cat'):
            hue_vals_in_data = [v for v in df[hue].cat.categories if v in df[hue].values]
        else:
            hue_vals_in_data = sorted(df[hue].dropna().unique())
        hue_palette = dict(zip(hue_vals_in_data,
                               sns.color_palette(hue_palette, n_colors=len(hue_vals_in_data))))

    hue_vals = list(hue_palette.keys())

    # --- build figure ---
    if split_by is not None:
        if split_order is not None:
            panels = split_order
        elif hasattr(df[split_by], 'cat'):
            panels = [v for v in df[split_by].cat.categories if v in df[split_by].values]
        elif split_by == hue and isinstance(hue_palette, dict):
            panels = [v for v in hue_palette if v in df[split_by].values]
        else:
            panels = sorted(df[split_by].dropna().unique())
        n_panels = len(panels)
        fig_w = figsize[0] if figsize else 2.5 * n_panels
        fig_h = figsize[1] if figsize else 2.5
        fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), sharey=sharey)
        axes = np.atleast_1d(axes)
    else:
        panels = [None]
        fig, ax_single = plt.subplots(figsize=figsize or (3, 2.5))
        axes = np.atleast_1d(ax_single)

    # --- plot ---
    ax_xlabel = f'{xlabel} (log scale)' if log else xlabel
    plotted_hue_vals = []

    for ax, panel_val in zip(axes, panels):
        panel_df = df[df[split_by] == panel_val] if split_by is not None else df

        if density_norm == 'all':
            all_vals = np.array(panel_df[x], dtype=float) * scale
            all_vals = all_vals[~np.isnan(all_vals)]
            n_panel = len(all_vals)
            shared_vals = np.log10(all_vals) if log else all_vals
            shared_bins = utils.gen_bins(shared_vals, binsize=binsize)

        for hue_val in hue_vals:
            vals = np.array(panel_df.loc[panel_df[hue] == hue_val, x], dtype=float) * scale
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            if hue_val not in plotted_hue_vals:
                plotted_hue_vals.append(hue_val)
            if log:
                vals = np.log10(vals)
                vals = vals[np.isfinite(vals)]
            if density_norm == 'all':
                ax.hist(vals, bins=shared_bins, color=hue_palette[hue_val], alpha=alpha,
                        density=False, histtype=histtype,
                        weights=np.full(len(vals), 1.0 / (n_panel * binsize)))
            else:
                bins = utils.gen_bins(vals, binsize=binsize)
                ax.hist(vals, bins=bins, color=hue_palette[hue_val], alpha=alpha,
                        density=density, histtype=histtype)

        if log:
            if tick_vals is not None:
                ax.set_xticks([np.log10(v) for v in tick_vals])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{10**x:.3g}'))

        ax.set_xlabel(ax_xlabel)
        if xlim is not None:
            ax.set_xlim(xlim)
        _despine(ax)

        # per-panel title only in multi-panel mode
        if split_by is not None:
            panel_title = (split_labels.get(panel_val, panel_val)
                           if split_labels else str(panel_val))
            ax.set_title(panel_title)

    axes[0].set_ylabel(ylabel)

    # --- title ---
    if title is not None:
        if split_by is None:
            axes[0].set_title(title)
        else:
            fig.suptitle(title)

    handles = [Patch(color=hue_palette[v], label=str(v)) for v in plotted_hue_vals]
    legend_kw = {'bbox_to_anchor': legend_bbox} if legend_bbox is not None else {}
    fig.legend(handles=handles, loc='upper right', frameon=False, title=legend_title, **legend_kw)

    fig.tight_layout()
    _save(fig, savedir, filename)
    return (fig, axes[0]) if split_by is None else (fig, axes)


# ---------------------------------------------------------------------------
# 5. scatter_umap_metric
#    UMAP scatter coloured by a binned metric (left) + countplot of bins
#    (right).  Legend placed outside to the right.
# ---------------------------------------------------------------------------

def scatter_umap_metric(
    df,
    metric,                          # column base; bins expected in metric+'_bin'
    metric_lab,                      # display label (axis, legend title)
    proj='umap',                     # projection prefix; uses proj+'_1' / proj+'_2'
    palette='Blues',
    countplot_hue=True,              # False → plain countplot (avoids seaborn hue warnings)
    x_formatter=None,                # formatter for countplot x-axis; set_powerlimits applied internally
    x_scale_label=None,              # string appended to x-axis label, e.g. '×10⁻²'
    y_formatter=None,                # FuncFormatter for countplot y-axis; None → default
    legend_num_format=None,          # format string for numeric bin labels, e.g. '{:,.0f}' or '{:.2e}'
    legend_scale_label=None,         # string appended to legend title, e.g. '$\\times 10^{-2}$'
    figsize=(6, 3.2),
    alpha=0.7,
    savedir=None,
    filename='scatter_umap_metric.png',
    legend_bbox=(0.95, 1),
    suptitle=None,
):
    """
    Left panel: UMAP scatter coloured by binned metric.
    Right panel: count of calls per bin.
    Legend stacked to the right of the figure.
    """
    bin_col = metric + '_bin'

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sns.scatterplot(df, x=proj + '_1', y=proj + '_2',
                    hue=bin_col, palette=palette,
                    edgecolor='k', alpha=alpha,
                    ax=axes[0])

    count_kw = {'hue': bin_col} if countplot_hue else {}
    sns.countplot(data=df, x=bin_col,
                  palette=palette, 
                  legend=False, edgecolor='#636363',
                  ax=axes[1], **count_kw)

    axes[1].set_xticks(axes[1].get_xticks()[0:-1:2])
    xlabel_scale = f' {x_scale_label}' if x_scale_label is not None else ''
    axes[1].set_xlabel(metric_lab + xlabel_scale + '\n(bin upper bound)')
    axes[1].set_ylabel('count')

    if x_formatter is not None:
        x_formatter.set_powerlimits((3, 3))
        axes[1].xaxis.set_major_formatter(x_formatter)

    if y_formatter is not None:
        axes[1].yaxis.set_major_formatter(y_formatter)

    for ax in axes:
        _despine(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend_.remove()

    if legend_num_format is not None:
        def _fmt(s):
            try:
                return legend_num_format.format(float(s))
            except (ValueError, TypeError):
                return s
        labels = [_fmt(l) for l in labels]

    legend_scale = f' {legend_scale_label}' if legend_scale_label is not None else ''
    legend_kw = {'bbox_to_anchor': legend_bbox} if legend_bbox is not None else {}
    fig.legend(handles=handles, labels=labels,
               loc='upper left', frameon=False,
               title=metric_lab + legend_scale + '\n(bin upper bound)',
               **legend_kw)

    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.tight_layout()
    _save(fig, savedir, filename)
    return fig, axes


# ---------------------------------------------------------------------------
# 6. unity_line
#    Draw a y=x dashed line on an axis from 0 to the max of the input data.
# ---------------------------------------------------------------------------

def unity_line(ax, data, linestyle='--', color='k', **kwargs):
    """Draw a unity (y = x) line on *ax* from 0 to max(*data*)."""
    vmax = np.max(data)
    ax.plot([0, vmax], [0, vmax], linestyle=linestyle, color=color, **kwargs)
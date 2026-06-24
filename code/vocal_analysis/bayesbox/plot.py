"""
plot.py — Plotting functions for Bayesian distribution-parameter inference.

Supports gamma (α, β) and gaussian (μ, σ) results interchangeably. Plot
functions that accept a results_dict auto-detect the distribution from the
stored 'dist' field; functions taking raw arrays take a `dist` kwarg that
defaults to 'gamma' for backward compatibility.

Style matches npxtoolbox.plot: despined axes, frameon=False legends,
tight_layout, optional save via _save().
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Distribution label lookup
# ---------------------------------------------------------------------------

# _DIST_LABELS = {

#     'gamma':    (r'$\alpha$ [shape)', r'$\beta$ (scale)'),
#     'gaussian': (r'$\mu$ (mean)',     r'$\sigma$ (std)'),
# }
_DIST_LABELS = {
    'gamma':    (r'$\alpha$', r'$\beta$'),
    'gaussian': (r'$\mu$',     r'$\sigma$'),
}
_DIST_TITLE = {'gamma': 'Gamma', 'gaussian': 'Gaussian'}

def _param_labels(dist):
    """Return (param1_label, param2_label) for a given distribution name."""
    return _DIST_LABELS.get(dist, _DIST_LABELS[dist])


def _dist_from_results(results_dict, fallback='gamma'):
    """Pull the `dist` field out of the first result in a results_dict."""
    if not results_dict:
        return fallback
    first = next(iter(results_dict.values()))
    return first.get('dist', fallback)


# ---------------------------------------------------------------------------
# Helpers (mirrors npxtoolbox.plot)
# ---------------------------------------------------------------------------

def _despine(ax):
    ax.spines[['right', 'top']].set_visible(False)


def _save(fig, savedir, filename):
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        fig.savefig(os.path.join(savedir, filename),
                    transparent=True, bbox_inches='tight', dpi=200)


# ---------------------------------------------------------------------------
# Axis-limit utility
# ---------------------------------------------------------------------------

def _nice_lims(lo, hi):
    """
    Snap lo down and hi up to the nearest round number appropriate for
    the magnitude of the values.  Step size = 10^floor(log10(center)),
    so the rounding adapts automatically to scale:
        center ~1.3  → step 1   → (0, 2)
        center ~1350 → step 1000 → (1000, 2000)
    """
    center = (lo + hi) / 2
    if center <= 0:
        step = 10 ** np.floor(np.log10(hi - lo)) if hi > lo else 1
    else:
        step = 10 ** np.floor(np.log10(center))
    return float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step)


def auto_lims(slice_results, buffer=0.15, nice=True):
    """
    Compute axis limits for a and b that cover all HPD credible regions
    across treatments in *slice_results*, with a fractional buffer, clipped
    to the inference grid extent.

    Parameters
    ----------
    slice_results : dict
        {treatment: result_dict} as returned by inference.run_inference.
    buffer : float
        Fractional padding added on each side of the combined credible range.
        E.g. 0.15 adds 15 % of the span on each side.
    nice : bool
        If True (default), snap limits to round numbers scaled to the
        magnitude of the values (e.g. [1.1, 1.5] → [0, 2];
        [1120, 1586] → [1000, 2000]).

    Returns
    -------
    a_lim : (float, float)
    b_lim : (float, float)
    """
    bounds_a_list = []
    bounds_b_list = []
    grid_a = grid_b = None

    for res in slice_results.values():
        bounds_a_list.append(res['bounds_a'])
        bounds_b_list.append(res['bounds_b'])
        if grid_a is None:
            grid_a = res['a_values']
            grid_b = res['b_values']

    a_lo = min(b[0] for b in bounds_a_list)
    a_hi = max(b[1] for b in bounds_a_list)
    b_lo = min(b[0] for b in bounds_b_list)
    b_hi = max(b[1] for b in bounds_b_list)

    a_span = a_hi - a_lo
    b_span = b_hi - b_lo

    a_lim = (max(float(grid_a.min()), a_lo - buffer * a_span),
             min(float(grid_a.max()), a_hi + buffer * a_span))
    b_lim = (max(float(grid_b.min()), b_lo - buffer * b_span),
             min(float(grid_b.max()), b_hi + buffer * b_span))

    if nice:
        a_lim = _nice_lims(*a_lim)
        b_lim = _nice_lims(*b_lim)

    return a_lim, b_lim


# ---------------------------------------------------------------------------
# 1. posterior_2d
#    2-D posterior P(a, b | data) as a filled heatmap.
# ---------------------------------------------------------------------------

def posterior_2d(prob_matrix, a_values, b_values,
                 ax=None,
                 cmap='Blues',
                 title=None,
                 a_lim=None, b_lim=None,
                 dist='gamma',
                 savedir=None,
                 filename='posterior_2d.png'):
    """
    Heatmap of the 2-D joint posterior P(param1, param2 | data).

    Parameter-2 is placed on the x-axis, parameter-1 on the y-axis.
    prob_matrix[i, j] = P(param1_i, param2_j | data), shape (n_a, n_b).
    Axis labels are chosen from `dist`:
        'gamma'    → α (shape) / β (scale)
        'gaussian' → μ (mean)  / σ (stdev)

    Parameters
    ----------
    prob_matrix : ndarray, shape (n_a, n_b)
        Normalised probability matrix from inference.ll_to_probability.
    a_values : 1-D array
        Parameter-1 grid values — rows of prob_matrix.
    b_values : 1-D array
        Parameter-2 grid values — columns of prob_matrix.
    ax : matplotlib Axes, optional
        Axes to draw into.  If None a new figure is created and saved.
    cmap : str
        Colormap passed to pcolormesh.
    title : str, optional
        Axes title.
    a_lim : (float, float), optional
        y-axis limits for parameter-1.  Defaults to full grid range.
    b_lim : (float, float), optional
        x-axis limits for parameter-2.  Defaults to full grid range.
    dist : {'gamma', 'gaussian'}
        Controls axis labels only.
    savedir, filename : str
        If savedir is not None the figure is saved there (only when ax is None).

    Returns
    -------
    fig, ax : Figure, Axes
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(3.5, 3))
    else:
        fig = ax.figure

    # pcolormesh: x → param2, y → param1
    pcm = ax.pcolormesh(b_values, a_values, prob_matrix,
                        cmap=cmap, shading='auto')
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Probability', fontsize=9)
    
    p1_label, p2_label = _param_labels(dist)
    ax.set_xlabel(p2_label)
    ax.set_ylabel(p1_label)
    if b_lim is not None:
        ax.set_xlim(b_lim)
    if a_lim is not None:
        ax.set_ylim(a_lim)
    _despine(ax)

    if title is not None:
        ax.set_title(title)

    if own_fig:
        fig.tight_layout()
        _save(fig, savedir, filename)

    return fig, ax


# ---------------------------------------------------------------------------
# 2. posterior_2d_grid
#    Side-by-side 2-D posteriors for each treatment (or any label set).
# ---------------------------------------------------------------------------

def posterior_2d_grid(results_dict, a_values, b_values,
                      treatment_order=None,
                      cmap='Blues',
                      title=None,
                      figsize=None,
                      a_lim=None, b_lim=None,
                      dist=None,
                      savedir=None,
                      filename='posterior_2d_grid.png'):
    """
    Side-by-side 2-D posteriors, one panel per key in results_dict.

    Calls posterior_2d for each key and shares the parameter-1 y-axis across
    panels. The distribution (for axis labels) is auto-detected from the
    first entry in results_dict unless overridden via `dist`.

    Parameters
    ----------
    results_dict : dict
        {label: result_dict} where result_dict contains 'prob' and optionally
        'dist' as returned by inference.run_inference.
    a_values, b_values : 1-D arrays
        Parameter grids (passed through to posterior_2d).
    treatment_order : list, optional
        Panel order.  Defaults to sorted keys.
    cmap : str
        Colormap for all panels.
    title : str, optional
        Figure suptitle.
    figsize : (float, float), optional
        Override figure size.  Defaults to (3.2 * n_panels, 3.0).
    a_lim, b_lim : (float, float), optional
        Shared axis limits for parameter-1 and parameter-2 across panels.
    dist : {'gamma', 'gaussian'}, optional
        Axis-label override; defaults to the 'dist' field in results_dict.
    savedir, filename : str
        Save destination.

    Returns
    -------
    fig, axes : Figure, ndarray of Axes
    """
    keys = treatment_order if treatment_order is not None else sorted(results_dict)
    keys = [k for k in keys if k in results_dict]
    n = len(keys)

    if dist is None:
        dist = _dist_from_results(results_dict)

    fig_w = figsize[0] if figsize else 3.2 * n
    fig_h = figsize[1] if figsize else 3.0
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, key in zip(axes, keys):
        prob = results_dict[key]['prob']
        _, _ = posterior_2d(prob, a_values, b_values, ax=ax, cmap=cmap,
                            title=key, a_lim=a_lim, b_lim=b_lim, dist=dist)

    p1_label, _ = _param_labels(dist)
    axes[0].set_ylabel(p1_label)
    for ax in axes[1:]:
        ax.set_ylabel('')

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    _save(fig, savedir, filename)
    return fig, axes


# ---------------------------------------------------------------------------
# 3. marginal_posteriors
#    Overlaid marginal distributions for a and b across treatments.
#    Shaded region marks the HPD credible interval.
# ---------------------------------------------------------------------------

def marginal_posteriors(a_values, b_values, results_dict,
                        treatment_palette,
                        credibility=0.95,
                        treatment_order=None,
                        title=None,
                        linewidth=1.8,
                        alpha_fill=0.15,
                        figsize=(5.5, 2.4),
                        legend_bbox=None,
                        legend=True,
                        a_lim=None, b_lim=None,
                        dist=None,
                        savedir=None,
                        filename='marginal_posteriors.png'):
    """
    Two-panel figure of marginal posteriors for parameter-1 and parameter-2.

    Treatments are overlaid as coloured lines; the HPD credible interval for
    each treatment is shown as a shaded span. Distribution is auto-detected
    from the 'dist' field in results_dict unless `dist` is passed explicitly.

    Parameters
    ----------
    a_values, b_values : 1-D arrays
        Parameter grids (parameter-1 and parameter-2).
    results_dict : dict
        {treatment: result_dict} with keys 'prob_a', 'prob_b',
        'bounds_a', 'bounds_b' as returned by inference.run_inference.
    treatment_palette : dict
        {treatment: colour}.
    credibility : float
        Used in the legend title only; shading uses the stored HPD bounds.
    treatment_order : list, optional
        Plot order.  Defaults to sorted keys.
    title : str, optional
        Figure suptitle.
    linewidth : float
        Line width for posterior curves.
    alpha_fill : float
        Opacity of the HPD shaded region.
    figsize : (float, float)
        Figure size.
    legend_bbox : (float, float), optional
        bbox_to_anchor for the figure legend.  Useful for moving it outside
        the axes (e.g. (1.15, 1.0)).
    a_lim, b_lim : (float, float), optional
        x-axis limits for the parameter-1 and parameter-2 panels respectively.
    dist : {'gamma', 'gaussian'}, optional
        Axis-label override; defaults to the 'dist' field in results_dict.
    savedir, filename : str
        Save destination.

    Returns
    -------
    fig, axes : Figure, ndarray of Axes (length 2)
    """
    keys = treatment_order if treatment_order is not None else sorted(results_dict)
    keys = [k for k in keys if k in results_dict]

    if dist is None:
        dist = _dist_from_results(results_dict)
    p1_label, p2_label = _param_labels(dist)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    param_specs = [
        (p1_label, a_values, 'prob_a', 'bounds_a'),
        (p2_label, b_values, 'prob_b', 'bounds_b'),
    ]

    for ax, (param_label, param_vals, prob_key, bounds_key) in zip(axes, param_specs):
        for key in keys:
            res = results_dict[key]
            color = treatment_palette.get(key, 'gray')
            posterior = res[prob_key]
            bounds    = res[bounds_key]

            ax.plot(param_vals, posterior,
                    color=color, linewidth=linewidth, label=key)
            ax.axvspan(bounds[0], bounds[1],
                       color=color, alpha=alpha_fill)

        ax.set_xlabel(param_label)
        ax.set_ylabel('Probability')
        _despine(ax)

    if a_lim is not None:
        axes[0].set_xlim(a_lim)
        axes[0].set_xticks(np.linspace(a_lim[0],a_lim[1],4), np.array(np.linspace(a_lim[0],a_lim[1],4),int))
        # axes[0].set_xticks(np.linspace(a_lim[0],a_lim[1],3), np.round(np.linspace(a_lim[0],a_lim[1],3),2))

    if b_lim is not None:
        axes[1].set_xlim(b_lim)
        axes[1].set_xticks(np.linspace(b_lim[0],b_lim[1],3), np.array(np.linspace(b_lim[0],b_lim[1],3),int))
        # axes[1].set_xticks(np.linspace(b_lim[0],b_lim[1],3), np.round(np.linspace(b_lim[0],b_lim[1],3),2))

    # Credible-interval annotation on right panel y-axis label
    axes[1].set_ylabel('')
    if legend is True:
        handles, labels = axes[0].get_legend_handles_labels()
        legend_kw = {'bbox_to_anchor': legend_bbox} if legend_bbox is not None else {}
        fig.legend(handles, labels,
               loc='upper right', frameon=False,
               title=f'{int(credibility*100)}% HPD (shaded)', **legend_kw)

    if title is not None:
        fig.suptitle(title, y=1.02)

    fig.tight_layout()
    _save(fig, savedir, filename)
    return fig, axes


# ---------------------------------------------------------------------------
# 4. credible_region_summary
#    Bar/line plot comparing credible interval widths across slices.
# ---------------------------------------------------------------------------

def credible_region_summary(summary_df,
                             param='a',
                             treatment_palette=None,
                             ylabel_map=None,
                             figsize=(5, 2.5),
                             dist='gamma',
                             savedir=None,
                             filename='credible_region_summary.png'):
    """
    Horizontal error-bar plot of HPD credible intervals across data slices.

    Each row shows one (slice, treatment) combination. The dot marks the
    posterior mode; error bars span the HPD credible interval.

    Parameters
    ----------
    summary_df : DataFrame
        Columns: 'slice', 'treatment', 'lower', 'upper', 'center'.
        One row per (slice, treatment) pair, typically produced by iterating
        over inference.run_inference results.
    param : {'a', 'b'}
        Which parameter axis to plot — 'a' is parameter-1 (α or μ),
        'b' is parameter-2 (β or σ).
    treatment_palette : dict, optional
        {treatment: colour}.  Falls back to steelblue if not provided.
    ylabel_map : dict, optional
        {slice_name: display_label} to replace raw slice keys with readable
        labels on the y-axis.
    figsize : (float, float)
        Figure size.
    dist : {'gamma', 'gaussian'}
        Chooses the x-axis label (e.g. 'Gamma α (shape)' vs 'Gaussian μ (mean)').
    savedir, filename : str
        Save destination.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    slices = summary_df['slice'].unique()
    fig, ax = plt.subplots(figsize=figsize)

    y_tick_labels = []
    y_pos = 0
    y_ticks = []

    for slice_name in slices:
        sub = summary_df[summary_df['slice'] == slice_name]
        for _, row in sub.iterrows():
            color = (treatment_palette.get(row['treatment'], 'gray')
                     if treatment_palette else 'steelblue')
            xerr = np.array([[row['center'] - row['lower']],
                              [row['upper'] - row['center']]])
            ax.errorbar(row['center'], y_pos,
                        xerr=xerr,
                        fmt='o', color=color, capsize=3, linewidth=1.5)
            label = ylabel_map.get(slice_name, slice_name) if ylabel_map else slice_name
            # y_tick_labels.append(f"{label}\n{row['treatment']}")
            y_tick_labels.append(f"{label}")
            y_ticks.append(y_pos)
            y_pos += 1
        y_pos += 0.5  # gap between slices

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=8)
    p1_label, p2_label = _param_labels(dist)
    param_label = p1_label if param == 'a' else p2_label
    ax.set_xlabel(f'{_DIST_TITLE.get(dist, dist.title())} {param_label}')
    _despine(ax)
    fig.tight_layout()
    _save(fig, savedir, filename)
    return fig, ax

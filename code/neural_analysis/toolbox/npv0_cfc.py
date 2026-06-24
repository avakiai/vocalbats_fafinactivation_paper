"""
npv0_cfc.py — Cross-frequency coupling (CFC) analysis for LFP data.

Two methods are implemented:

  1. Mean Vector Length (MVL) modulation index — time-resolved.
     Canolty et al. (2006): complex product exp(iφ)·A, averaged in sliding windows.
     Encodes both coupling strength (|MI|) and preferred phase (∠MI).

  2. Tort mutual information (MI) index — scalar per band pair.
     Tort et al. (2010): phase bins → mean amplitude per bin → MI from entropy.

All CFC functions operate on a single 1-D LFP trace. For multi-channel data,
select a representative channel or average good channels before calling —
the high channel correlation in this dataset makes redundant computation on all
channels unnecessary and expensive.
"""

import numpy as np
from scipy import signal as sg
import matplotlib.pyplot as plt
from matplotlib import colors, ticker


# ---------------------------------------------------------------------------
# Band definition
# ---------------------------------------------------------------------------

def make_phase_bands(start=1, stop=8, step=1, width=2):
    """
    (2, n_bands) array of [low, high] band edges for phase (Hz).

    Defaults: 1-9 Hz in 1 Hz steps with 2 Hz bandwidth — matches notebook.
    """
    lows = np.arange(start, stop, step)
    return np.vstack((lows, lows + width))


def make_amp_bands(start=15, stop=120, step=5, width=10):
    """
    (2, n_bands) array of [low, high] band edges for amplitude (Hz).

    Defaults: 15-120 Hz in 5 Hz steps with 10 Hz bandwidth.
    """
    lows = np.arange(start, stop, step)
    return np.vstack((lows, lows + width))


# ---------------------------------------------------------------------------
# Phase / amplitude extraction
# ---------------------------------------------------------------------------

def extract_phase(lfp_1d, bands, Fs, order=4):
    """
    Bandpass-filter and extract instantaneous phase for each band.

    Parameters
    ----------
    lfp_1d : 1-D array
    bands  : ndarray, shape (2, n_bands)   row 0 = low edges, row 1 = high edges
    Fs     : float
    order  : int   Butterworth order

    Returns
    -------
    phase_dict : {f_low: phase_array}   instantaneous phase in radians (−π to π)
    """
    nyq = 0.5 * Fs
    phase_dict = {}
    for i in range(bands.shape[1]):
        lo, hi = bands[0, i], bands[1, i]
        sos = sg.butter(order, [lo, hi] / nyq, btype='band', output='sos')
        filt = sg.sosfiltfilt(sos, lfp_1d)
        phase_dict[lo] = np.angle(sg.hilbert(filt))
    return phase_dict


def extract_amplitude(lfp_1d, bands, Fs, order=4):
    """
    Bandpass-filter and extract instantaneous amplitude envelope for each band.

    Returns
    -------
    amp_dict : {f_low: amplitude_array}
    """
    nyq = 0.5 * Fs
    amp_dict = {}
    for i in range(bands.shape[1]):
        lo, hi = bands[0, i], bands[1, i]
        sos = sg.butter(order, [lo, hi] / nyq, btype='band', output='sos')
        filt = sg.sosfiltfilt(sos, lfp_1d)
        amp_dict[lo] = np.abs(sg.hilbert(filt))
    return amp_dict


# ---------------------------------------------------------------------------
# Modulation index — time-resolved
# ---------------------------------------------------------------------------

def MI_windowed(product_ts, win_len_s, win_step_s, Fs):
    """
    Sliding-window modulation index from a complex product time series.

    Parameters
    ----------
    product_ts  : 1-D complex array
        exp(iφ)·A − mean(exp(iφ)·A) for one phase–amplitude pair.
    win_len_s   : float   window length in seconds
    win_step_s  : float   window step in seconds
    Fs          : float

    Returns
    -------
    MI_ts : 1-D complex array
        Complex MI per window. |MI| = coupling strength, ∠MI = preferred phase.
    x_ts  : 1-D int array   window start positions in samples
    """
    win_samp = int(win_len_s * Fs)
    step_samp = int(win_step_s * Fs)
    n = len(product_ts)
    starts = np.arange(0, n, step_samp)
    MI_ts = np.array([np.nanmean(product_ts[s:min(s + win_samp, n)]) for s in starts])
    return MI_ts, starts


def compute_cfc(lfp_1d, phase_bands, amp_bands, Fs,
                phase_dict=None, amp_dict=None,
                timeresolved=True, win_len_s=2.5, win_step_s=0.5, 
                order=4):
    """
    Time-resolved CFC over a grid of phase × amplitude band pairs.

    Parameters
    ----------
    lfp_1d       : 1-D array   single LFP channel
    phase_bands  : ndarray, shape (2, n_phase_bands)   from make_phase_bands
    amp_bands    : ndarray, shape (2, n_amp_bands)     from make_amp_bands
    Fs           : float
    win_len_s    : float   sliding window length (s)
    win_step_s   : float   sliding window step (s)
    order        : int     Butterworth order for bandpass filters

    Returns
    -------
    ModInd_mat : nested dict
        ModInd_mat[f_phase][f_amp] = {'MI_ts': complex array, 'x_ts': int array}
    phase_dict : {f_low: phase_array}
    amp_dict   : {f_low: amp_array}
    """
    if phase_dict is None:
        phase_dict = extract_phase(lfp_1d, phase_bands, Fs, order=order)
    if amp_dict is None:
        amp_dict = extract_amplitude(lfp_1d, amp_bands, Fs, order=order)

    ModInd_mat = {}
    for fpha, iphase in phase_dict.items():
        ModInd_mat[fpha] = {}
        for fAmp, iamp in amp_dict.items():
            product_ts = np.exp(1j * iphase) * iamp - np.mean(np.exp(1j * iphase) * iamp)
            if timeresolved:
                MI_ts, x_ts = MI_windowed(product_ts, win_len_s, win_step_s, Fs)
                ModInd_mat[fpha][fAmp] = {'MI_ts': MI_ts, 'x_ts': x_ts}
            else:
                MI_ts = np.nanmean(product_ts)
                ModInd_mat[fpha][fAmp] = {'MI_ts': MI_ts,'x_ts': None}


    return ModInd_mat, phase_dict, amp_dict


# ---------------------------------------------------------------------------
# Tort mutual information index — scalar
# ---------------------------------------------------------------------------

def tort_mi(phase_1d, amp_1d, n_bins=18):
    """
    Tort MI index for one phase–amplitude pair.

    Bins phase into n_bins equal bins over [0, 2π], computes mean amplitude
    per bin, normalises to a probability distribution, then:
        MI = (log N − H) / log N
    where H is the Shannon entropy. MI = 0 means uniform (no coupling),
    MI = 1 means all amplitude concentrated in one phase bin.

    Parameters
    ----------
    phase_1d : 1-D array   instantaneous phase (any range; converted to [0, 2π])
    amp_1d   : 1-D array   instantaneous amplitude
    n_bins   : int         number of phase bins (default 18 → 20° bins)

    Returns
    -------
    mi          : float          MI in [0, 1]
    mean_amp_p  : ndarray (n_bins,)   normalised amplitude distribution P
    """
    phase_2pi = phase_1d % (2 * np.pi)
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)

    mean_amp = np.array([
        np.mean(amp_1d[(phase_2pi >= bin_edges[b]) & (phase_2pi < bin_edges[b + 1])])
        for b in range(n_bins)
    ])
    mean_amp = np.nan_to_num(mean_amp)
    p = mean_amp / mean_amp.sum() if mean_amp.sum() > 0 else np.ones(n_bins) / n_bins

    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.nansum(p * np.log(p + 1e-12))
    mi = (np.log(n_bins) - entropy) / np.log(n_bins)
    return float(mi), p


def compute_cfc_tort(phase_dict, amp_dict, n_bins=18):
    """
    Apply tort_mi over all phase × amplitude band pairs.

    Parameters
    ----------
    phase_dict : {f_low: phase_array}   from extract_phase
    amp_dict   : {f_low: amp_array}     from extract_amplitude
    n_bins     : int

    Returns
    -------
    MI_mat : nested dict
        MI_mat[f_phase][f_amp] = {'mi': float, 'p': ndarray (n_bins,)}
    """
    MI_mat = {}
    for fpha, phase in phase_dict.items():
        MI_mat[fpha] = {}
        for fAmp, amp in amp_dict.items():
            mi, p = tort_mi(phase, amp, n_bins=n_bins)
            MI_mat[fpha][fAmp] = {'mi': mi, 'p': p}
    return MI_mat


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cfc_polar(MI_ts, x_ts, Fs, title=None, ax=None, savepath=None):
    """
    Polar scatter of time-resolved MI values, colored by time.

    Each point is one sliding window: position encodes preferred phase and
    coupling strength; colour encodes time within recording.

    Parameters
    ----------
    MI_ts    : 1-D complex array   from mvl_windowed or ModInd_mat[fp][fa]['MI_ts']
    x_ts     : 1-D int array       window start positions in samples
    Fs       : float
    title    : str, optional
    ax       : matplotlib Axes, optional
    savepath : str, optional
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    t_min = x_ts / Fs / 60
    sc = ax.scatter(MI_ts.real, MI_ts.imag, c=t_min, cmap='viridis', s=8, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='time [min]')

    ax.axvline(0, color='k', lw=0.5)
    ax.axhline(0, color='k', lw=0.5)

    mean_vec = np.nanmean(MI_ts)
    ax.plot([0, mean_vec.real], [0, mean_vec.imag], 'r-', lw=2)

    # r = max(np.abs(MI_ts.real).max(), np.abs(MI_ts.imag).max()) * 1.1
    # for label, (x, y), ha, va in [
    #     ('0',    ( r,  0), 'left',   'center'),
    #     ('π/2',  ( 0,  r), 'center', 'bottom'),
    #     ('±π',   (-r,  0), 'right',  'center'),
    #     ('-π/2', ( 0, -r), 'center', 'top'),
    # ]:
    #     ax.annotate(label, xy=(x, y), ha=ha, va=va, fontsize=11)

    # theta = np.linspace(0, 2 * np.pi, 200)
    # ax.plot(r * np.cos(theta), r * np.sin(theta), 'k--', lw=0.5, alpha=0.4)
    # ax.set_xlim(-r * 1.2, r * 1.2)
    # ax.set_ylim(-r * 1.2, r * 1.2)

    ax.set_xlabel('real  [cos φ · A]')
    ax.set_ylabel('imag  [sin φ · A]')

    pref_phase = np.angle(mean_vec)
    mvl = np.abs(mean_vec)
    ax.set_title(title or fr'pref. phase = {pref_phase:.2f} rad,  |MI| = {mvl:.3f}')

    if own_fig:
        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, bbox_inches='tight', transparent=True)
    return fig, ax


def plot_comodulogram(MI_mat, phase_bands, amp_bands, kind='mi',
                      cmap='inferno', title=None, ax=None,
                      xlabel=True, ylabel=True,
                      colorbar=True, cb_label=None, cb_powerlimits=None, cb_x_lab=None,
                      savepath=None):
    """
    Comodulogram heatmap of CFC over the phase × amplitude frequency grid.

    Parameters
    ----------
    MI_mat      : nested dict   from compute_cfc or compute_cfc_tort
    phase_bands : ndarray, shape (2, n_phase_bands)
    amp_bands   : ndarray, shape (2, n_amp_bands)
    kind        : {'strength', 'phase', 'mi'}
        'strength' → |mean(MI_ts)|, 'phase' → ∠mean(MI_ts), 'mi' → Tort MI
    cmap        : str
    title       : str, optional
    ax          : matplotlib Axes, optional
    xlabel, ylabel : bool
        Set False to suppress axis label text.
    colorbar    : bool
        Set False to skip the per-axes colorbar (use shared_colorbar() instead).
    cb_label    : str or None
        Label on the per-axes colorbar. None (default) = no label.
    cb_powerlimits : tuple (a, b) or None
        If given, passed to colorbar tick formatter as scilimits=(a, b).
        E.g. (0, 0) forces scientific notation on all ticks.
    savepath    : str, optional

    Returns
    -------
    fig, ax, pcm : Figure, Axes, QuadMesh
        pcm is the mappable — pass it to shared_colorbar() for a shared scale.
    """
    f_pha = phase_bands[0]
    f_amp = amp_bands[0]
    f_pha_centers = (phase_bands[0] + phase_bands[1]) / 2
    f_amp_centers = (amp_bands[0] + amp_bands[1]) / 2

    matrix = np.zeros((len(f_pha), len(f_amp)))
    for i, fp in enumerate(f_pha):
        for j, fa in enumerate(f_amp):
            entry = MI_mat[fp][fa]
            if kind == 'strength':
                matrix[i, j] = np.abs(np.nanmean(entry['MI_ts']))
            elif kind == 'phase':
                matrix[i, j] = np.angle(np.nanmean(entry['MI_ts']))
            elif kind == 'mi':
                matrix[i, j] = entry['mi']
            elif kind == 'mi_phase':
                matrix[i,j] = np.argmax(entry['p'])

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.figure

    pcm = ax.pcolormesh(f_pha_centers, f_amp_centers, matrix.T, cmap=cmap, shading='nearest')

    if xlabel:
        ax.set_xlabel('phase frequency [Hz]')
    if ylabel:
        ax.set_ylabel('amplitude frequency [Hz]')
    if title is not None:
        ax.set_title(title)

    if colorbar:
        cb = fig.colorbar(pcm, ax=ax, label=cb_label or '')
        if cb_powerlimits is not None:
            fmt = ticker.ScalarFormatter(useMathText=True)
            fmt.set_powerlimits(cb_powerlimits)
            cb.ax.yaxis.set_major_formatter(fmt)
            if cb_x_lab is not None:
                ot = cb.ax.yaxis.get_offset_text()
                ot.set_x(cb_x_lab)
                ot.set_ha('left')
    if own_fig:
        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, bbox_inches='tight', transparent=True)
    return fig, ax, pcm


def shared_colorbar(fig, pcms, axes=None, cax=None, label='MI', symmetric=False,
                    powerlimits=(0, 0), fraction=0.02, pad=0.04, shrink=1.0,
                    cb_x_lab=None):
    """
    Add a single shared colorbar normalised across all panels.

    The range covers the full min/max across all supplied pcms.

    Parameters
    ----------
    fig      : Figure
    pcms     : list of QuadMesh   returned from plot_comodulogram calls
    axes     : Axes or list of Axes, optional
        Axes to steal space from for the colorbar. Defaults to all axes in fig,
        which places the bar to the right of the entire grid.
    cax      : Axes, optional
        Pre-created axes to draw the colorbar into (overrides axes/fraction/pad).
    label    : str
    symmetric : bool
        Force vmin = -vmax (useful for difference colormaps like PuOr).
    powerlimits : (int, int)
        (0, 0) forces scientific notation; use (-4, 4) for auto.
    fraction : float   fraction of axes width used for colorbar (default 0.02)
    pad      : float   gap between axes and colorbar (default 0.04)
    shrink   : float   fraction of axes height to shrink colorbar to (default 1.0)
    offset_label_x : float or None
        X position (in colorbar axes coordinates) for the scientific-notation
        exponent label (e.g. "×10⁻²") that appears at the top of the colorbar.
        Default None leaves matplotlib's placement unchanged. Values > 1 shift
        the label further right, away from the figure panels (e.g. 1.5).

    Returns
    -------
    cbar : Colorbar
    """

    vmin = min(pcm.get_array().min() for pcm in pcms)
    vmax = max(pcm.get_array().max() for pcm in pcms)

    if symmetric:
        lim = max(abs(vmin), abs(vmax))
        norm = colors.Normalize(vmin=-lim, vmax=lim)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    for pcm in pcms:
        pcm.set_norm(norm)

    if cax is not None:
        cbar_kw = {'cax': cax}
    else:
        if axes is not None:
            target_axes = axes
        else:
            # exclude colorbar axes (they have no subplot spec)
            target_axes = [a for a in fig.axes if a.get_subplotspec() is not None]
        cbar_kw = {'ax': target_axes, 'fraction': fraction,
                   'pad': pad, 'shrink': shrink}

    cbar = fig.colorbar(pcms[0], label=label, **cbar_kw)
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits(powerlimits)
    cbar.ax.yaxis.set_major_formatter(fmt)

    if cb_x_lab is not None:
        # Draw once so the offset text exists, then reposition it.
        fig.canvas.draw()
        ot = cbar.ax.yaxis.get_offset_text()
        ot.set_x(cb_x_lab)
        ot.set_ha('left')

    return cbar

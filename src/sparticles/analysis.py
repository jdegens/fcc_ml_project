import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sparticles.plotting import (
    get_event_dataframe,
    set_plot_style,
    SIGNAL_COLOR,
    BACKGROUND_COLOR,
    COMPLEX_VAR_CONFIG,
)

# ---------------------------------------------------------------------------
# Scale factors (event weights)
# Each simulated event represents this many real expected events at FCC-hh
# luminosity, accounting for cross-section, k-factor, matching efficiency,
# and the number of events generated.
#
# signal:     scale = (xs × k × eff × lumi) / N_generated
# background: scale = (xs × k × eff × lumi) / N_generated
# ---------------------------------------------------------------------------
SIGNAL_SCALE     = 0.5322900454815167
BACKGROUND_SCALE =  13166.215355474176 # 777.3105506024549


def _build_event_table(dataset):
    """Build and cache the event DataFrame on the dataset object."""
    if not hasattr(dataset, '_event_table'):
        dataset._event_table = get_event_dataframe(dataset)
    return dataset._event_table


def summarise(dataset):
    """Print the range of each variable for signal and background events."""
    events = _build_event_table(dataset)
    sig    = events[events['label'] == 1]
    bkg    = events[events['label'] == 0]
    vars_  = [c for c in events.columns if c != 'label']

    print(f"Loaded {len(sig):,} signal events and {len(bkg):,} background events.\n")
    print(f"{'Variable':<12} {'Sig min':>10} {'Sig mean':>10} {'Sig max':>10}"
          f"  {'Bkg min':>10} {'Bkg mean':>10} {'Bkg max':>10}")
    print("-" * 76)
    for v in vars_:
        print(f"{v:<12}"
              f" {sig[v].min():>10.1f} {sig[v].mean():>10.1f} {sig[v].max():>10.1f}"
              f"  {bkg[v].min():>10.1f} {bkg[v].mean():>10.1f} {bkg[v].max():>10.1f}")


def apply_cuts(dataset, cuts, 
               signal_weight=SIGNAL_SCALE,
               background_weight=BACKGROUND_SCALE,
               save_dir='Saved Figures/CutAnalysis',
               show=True):
    """
    Apply cuts to the derived variables, calculate a physically weighted
    S/sqrt(B) significance, and plot the cut regions.

    Parameters
    ----------
    dataset : EventsDataset
    cuts : dict
        { 'variable_name': (min, max) }  --  use None for no limit.
        e.g. { 'm_bb': (100, 140), 'm_HH': (None, 1000) }
    signal_weight : float
        Scale factor per signal event (cross-section × k-factor × lumi / N_gen).
    background_weight : float
        Scale factor per background event.
    save_dir : str
        Directory for saved figures.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    float
        Weighted S/sqrt(B) after cuts.
    """
    events    = _build_event_table(dataset)
    sig_events = events[events['label'] == 1]
    bkg_events = events[events['label'] == 0]

    # Unweighted totals
    S_total_raw = len(sig_events)
    B_total_raw = len(bkg_events)

    signal_weight = signal_weight * (233994/S_total_raw)
    background_weight = background_weight * (56377/B_total_raw)

    # Weighted totals (physically expected event yields)
    S_total_w = S_total_raw * signal_weight 
    B_total_w = B_total_raw * background_weight

    # Apply cuts
    mask = pd.Series([True] * len(events), index=events.index)
    for var, ranges in cuts.items():
        if var not in events.columns:
            raise ValueError(
                f"Variable '{var}' not in dataset. "
                f"Available: {[c for c in events.columns if c != 'label']}"
            )
        # Each variable can have multiple allowed ranges joined by OR
        # (an event passes if it falls in ANY of the listed ranges)
        var_mask = pd.Series([False] * len(events), index=events.index)
        for (lo, hi) in ranges:
            range_mask = pd.Series([True] * len(events), index=events.index)
            if lo is not None:
                range_mask &= events[var] >= lo
            if hi is not None:
                range_mask &= events[var] <= hi
            var_mask |= range_mask
        mask &= var_mask

    passing   = events[mask]
    S_raw     = (passing['label'] == 1).sum()
    B_raw     = (passing['label'] == 0).sum()

    # Weighted counts after cuts
    S_w = S_raw * signal_weight
    B_w = B_raw * background_weight

    baseline_w  = S_total_w / np.sqrt(B_total_w)
    baseline_sys_w  = S_total_w / (np.sqrt(B_total_w) + 0.02 * B_total_w)


    sig_after_w = S_w / np.sqrt(B_w) if B_w > 0 else float('nan')
    sig_after_sys_w = S_w / (np.sqrt(B_w) + 0.02 * B_w) if B_w > 0 else float('nan')

    # ---- Printout ----
    print("=" * 52)
    print("  Cut-based analysis results")
    print("=" * 52)
    for var, ranges in cuts.items():
        for (lo, hi) in ranges:
            lo_str = f"{lo:.1f}" if lo is not None else "-∞"
            hi_str = f"{hi:.1f}" if hi is not None else "+∞"
            print(f"  {var}: {lo_str}  →  {hi_str}")
    print("-" * 52)
    print(f"  Events passing cuts: {S_raw:,} signal,  {B_raw:,} background")
    print(f"  Signal efficiency:   {100*S_raw/S_total_raw:.1f}%")
    print(f"  Background rejection:{100*(1 - B_raw/B_total_raw):.1f}%")
    print("-" * 52)
    print(f"  Weighted expected signal:     {S_w:>10.1f} events")
    print(f"  Weighted expected background: {B_w:>10.1f} events")
    print("-" * 52)
    if B_w == 0:
        print("  No background passed D: -- try relaxing the cuts.")
    else:
        print(f"  Baseline S/√B (no cuts): {baseline_w:.2f}, Baseline S/√B (with 2% sys): {baseline_sys_w:.2f}")
        print(f"  Your    S/√B (with cuts): {sig_after_w:.2f}, Your S/√B (with 2% sys): {sig_after_sys_w:.2f}")
        if sig_after_w > baseline_w:
            print(f"  Improved by {sig_after_w - baseline_w:.2f} :)"
                  f"  ({100*(sig_after_w/baseline_w - 1):.1f}% better)")
        else:
            print(f"  Worse than no cuts by {baseline_w - sig_after_w:.2f} :(")
        print()
    print("=" * 52)

    # ---- Plot ----
    vars_to_plot = list(cuts.keys())
    n = len(vars_to_plot)
    if n == 0:
        return sig_after_w
 
    set_plot_style()
    cols = 2
    rows = (n + cols - 1) // cols   # ceiling division
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows))
    # Flatten to a 1-D list regardless of grid shape, then hide any
    # unused axes in the last row if n is odd
    axes = np.array(axes).flatten()
    for ax in axes[n:]:
        ax.set_visible(False)
 
    for ax, var in zip(axes, vars_to_plot):
        config  = COMPLEX_VAR_CONFIG.get(var, dict(xlim=None, bins=100, label=var))
        ranges  = cuts[var]
        all_vals = pd.concat([sig_events[var], bkg_events[var]]).dropna()
        edges   = np.histogram(all_vals, bins=config['bins'])[1]
 
        ax.hist(sig_events[var].dropna(), edges,
                color=SIGNAL_COLOR,     histtype='step',
                density=True, linewidth=2, label='Signal', zorder=10)
        ax.hist(bkg_events[var].dropna(), edges,
                color=BACKGROUND_COLOR, histtype='step',
                density=True, linewidth=2, label='Background')
 
        for i, (lo, hi) in enumerate(ranges):
            x_min = edges[0]  if lo is None else lo
            x_max = edges[-1] if hi is None else hi
            label = 'Passing region' if i == 0 else None  # only one legend entry
            ax.axvspan(x_min, x_max, alpha=0.12, color='grey', label=label)
 
        if config['xlim']:
            ax.set_xlim(*config['xlim'])
        ax.set_xlabel(config['label'])
        ax.set_ylabel('Event Density')
        ax.legend(fontsize=16)
 
    fig.suptitle(
        f"Cut regions   |   S/√B: {baseline_w:.2f} → {sig_after_w:.2f}",
        y=1.02
    )
    fig.tight_layout()
 
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, 'cut_analysis.png'), bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'cut_analysis.pdf'), bbox_inches='tight')
 
    if show:
        plt.show()
    else:
        plt.close(fig)
 
    return sig_after_w

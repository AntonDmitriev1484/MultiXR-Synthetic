"""
CFAR-Adaptive Velocity Gate for UWB Range Outlier Detection
=============================================================
Replaces the hard velocity threshold with a CFAR-style adaptive threshold
on the delta sequence  d[i] = range[i] - range[i-1].

ANALOGY TO RADAR CFAR
-----------------------
Radar CFAR:
  Cell Under Test (CUT) : current range bin power
  Reference cells        : surrounding bins (excluding guard cells)
  Noise floor estimate   : statistic of reference cells (mean, percentile, ...)
  Threshold              : noise_floor * scale_factor  →  constant false alarm rate

Delta-sequence CFAR:
  Cell Under Test        : |d[i]| or d[i]  (current range step)
  Reference cells        : surrounding deltas (excluding guard cells)
  Noise floor estimate   : local typical delta magnitude  (= local motion speed)
  Threshold              : noise_floor * scale_factor
  → adapts to how fast the person is moving at each moment

WHY THIS IS BETTER THAN A HARD THRESHOLD
------------------------------------------
Hard threshold: flags d[i] > fixed_value.
  Problem: when the person is stationary, small NLoS steps go undetected.
           When moving fast, legitimate steps trigger false alarms.

CFAR threshold: flags d[i] > k * local_noise_estimate.
  Stationary person: local deltas ≈ 0, threshold is very low → catches
                     even small NLoS steps.
  Fast motion:       local deltas are large, threshold rises → fewer
                     false alarms on legitimate motion.

ONE-SIDED
----------
Only POSITIVE deltas are tested (range increases only).
NLoS bias is always positive; a recovery (large negative step) is accepted.

CFAR VARIANTS IMPLEMENTED
---------------------------
CA-CFAR  (Cell Averaging):
  noise = mean(|d| in reference window)
  threshold = k * noise
  Works well when the delta distribution is stationary within the window.
  Sensitive to clutter edges (e.g. transition from slow to fast motion).

OS-CFAR  (Order Statistics):
  noise = sorted(|d| in reference window)[rank]   rank ∈ (0,1)
  threshold = k * noise
  More robust to a few large values in the reference window.
  rank=0.75 is typical (75th percentile as noise estimate).

GOCA-CFAR  (Greatest-Of Cell Averaging):
  Split reference window into left half and right half.
  noise = max(mean_left, mean_right)
  Protects against clutter edges — uses the larger of the two sides,
  so a burst of large deltas on one side raises the threshold and
  prevents false alarms at the leading/trailing edge of fast motion.

SOCA-CFAR  (Smallest-Of Cell Averaging):
  noise = min(mean_left, mean_right)
  More sensitive — uses the quieter side, so outliers surrounded by
  calm motion on one side are more easily caught.
  Risk: at the leading edge of a fast-motion burst, left side is calm
  and threshold is low → may flag the first legitimate fast step.

All variants compared head-to-head against hard threshold and moving median.

USAGE
-----
  python uwb_cfar_sweep.py file1.json file2.json ...

OUTPUT
------
  cfar_vs_hard_<dataset>.png   — MAE comparison: CFAR variants vs hard threshold
  cfar_scale_sweep_<dataset>.png — sweep over scale factor k per CFAR type
  cfar_trace_<dataset>.png     — example link: adaptive threshold over time
  cfar_results.csv             — full numerical results
"""

import sys
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
import os
OUT_DIR = os.environ.get("OUT_DIR", "cfar_output4")

# CFAR window parameters
GUARD     = 1        # guard cells on each side of CUT (excluded from estimate)
REF_HALF  = 3        # reference cells on each side (total ref window = 2*REF_HALF)
OS_RANK   = 0.75     # OS-CFAR: percentile of reference cells to use as estimate

# Scale factor (k) sweep
K_VALUES  = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]

# Hard threshold sweep for comparison [m/s]
VTHRESH   = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# Reference baseline
MEDIAN_W  = 3

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def gt_range(r):
    return float(np.linalg.norm(
        np.array(r["optitrack_src_tx_position"]) -
        np.array(r["optitrack_dst_tx_position"])))

def moving_median(x, w=MEDIAN_W):
    x = np.asarray(x, float)
    n = len(x); y = x.copy(); h = w // 2
    for i in range(n):
        y[i] = np.median(x[max(0,i-h):min(n,i+h+1)])
    return y

def interp_accepted(times, values, reject_mask):
    """Linear interpolation over rejected samples."""
    out = values.copy().astype(float)
    for i in np.where(reject_mask)[0]:
        l = next((j for j in range(i-1,-1,-1)          if not reject_mask[j]), None)
        r = next((j for j in range(i+1,len(reject_mask)) if not reject_mask[j]), None)
        if   l is not None and r is not None:
            out[i] = out[l]+(out[r]-out[l])*(times[i]-times[l])/(times[r]-times[l])
        elif l is not None: out[i] = out[l]
        elif r is not None: out[i] = out[r]
    return out

# ══════════════════════════════════════════════════════════════════════════════
# CFAR DETECTORS
# All operate on the VELOCITY sequence  v[i] = (range[i]-range[i-1]) / dt[i]
# and return a boolean reject mask on the RANGE array (same length).
# First sample is never rejected.
# ══════════════════════════════════════════════════════════════════════════════

def _ref_cells(d_abs, i, guard, ref_half):
    """
    Return the reference cell values for position i in array d_abs.
    Guard cells [i-guard : i+guard+1] are excluded.
    Reference cells are taken from both sides beyond the guard band.
    """
    n   = len(d_abs)
    lo_ref_end   = max(0, i - guard)
    lo_ref_start = max(0, i - guard - ref_half)
    hi_ref_start = min(n, i + guard + 1)
    hi_ref_end   = min(n, i + guard + 1 + ref_half)
    left  = d_abs[lo_ref_start:lo_ref_end]
    right = d_abs[hi_ref_start:hi_ref_end]
    return left, right

def ca_cfar(times, values, k, guard=GUARD, ref_half=REF_HALF):
    """
    CA-CFAR on velocity sequence.
    Noise estimate = mean of all reference cells (left + right combined).
    Threshold = k * noise_estimate.
    """
    times   = np.asarray(times,  float)
    values  = np.asarray(values, float)
    n       = len(values)
    reject  = np.zeros(n, dtype=bool)
    thresh  = np.zeros(n)          # store for visualisation

    dt    = np.diff(times); dt = np.concatenate([[dt[0]], dt])
    vel   = np.concatenate([[0], np.diff(values)]) / np.maximum(dt, 1e-6)
    v_abs = np.abs(vel)

    for i in range(1, n):
        if vel[i] <= 0:            # one-sided: only positive steps
            thresh[i] = np.inf
            continue
        left, right = _ref_cells(v_abs, i, guard, ref_half)
        ref = np.concatenate([left, right])
        if len(ref) == 0:
            thresh[i] = np.inf
            continue
        noise     = np.mean(ref)
        thresh[i] = k * max(noise, 1e-6)
        if vel[i] > thresh[i]:
            reject[i] = True

    return reject, thresh

def os_cfar(times, values, k, rank=OS_RANK, guard=GUARD, ref_half=REF_HALF):
    """
    OS-CFAR on velocity sequence.
    Noise estimate = rank-th percentile of reference cells.
    More robust than CA when some reference cells contain outliers.
    """
    times   = np.asarray(times,  float)
    values  = np.asarray(values, float)
    n       = len(values)
    reject  = np.zeros(n, dtype=bool)
    thresh  = np.zeros(n)

    dt    = np.diff(times); dt = np.concatenate([[dt[0]], dt])
    vel   = np.concatenate([[0], np.diff(values)]) / np.maximum(dt, 1e-6)
    v_abs = np.abs(vel)

    for i in range(1, n):
        if vel[i] <= 0:
            thresh[i] = np.inf
            continue
        left, right = _ref_cells(v_abs, i, guard, ref_half)
        ref = np.concatenate([left, right])
        if len(ref) == 0:
            thresh[i] = np.inf
            continue
        noise     = np.percentile(ref, rank * 100)
        thresh[i] = k * max(noise, 1e-6)
        if vel[i] > thresh[i]:
            reject[i] = True

    return reject, thresh

def goca_cfar(times, values, k, guard=GUARD, ref_half=REF_HALF):
    """
    GOCA-CFAR: Greatest-Of Cell Averaging.
    Noise estimate = max(mean_left, mean_right).
    Protects against clutter edges — threshold stays high when either
    side of the window contains large legitimate motion.
    """
    times   = np.asarray(times,  float)
    values  = np.asarray(values, float)
    n       = len(values)
    reject  = np.zeros(n, dtype=bool)
    thresh  = np.zeros(n)

    dt    = np.diff(times); dt = np.concatenate([[dt[0]], dt])
    vel   = np.concatenate([[0], np.diff(values)]) / np.maximum(dt, 1e-6)
    v_abs = np.abs(vel)

    for i in range(1, n):
        if vel[i] <= 0:
            thresh[i] = np.inf
            continue
        left, right = _ref_cells(v_abs, i, guard, ref_half)
        noise_l   = np.mean(left)  if len(left)  > 0 else 0.0
        noise_r   = np.mean(right) if len(right) > 0 else 0.0
        noise     = max(noise_l, noise_r)
        thresh[i] = k * max(noise, 1e-6)
        if vel[i] > thresh[i]:
            reject[i] = True

    return reject, thresh

def soca_cfar(times, values, k, guard=GUARD, ref_half=REF_HALF):
    """
    SOCA-CFAR: Smallest-Of Cell Averaging.
    Noise estimate = min(mean_left, mean_right).
    More sensitive than GOCA — flags outliers that are large relative
    to the quieter side of the window.  Higher false-alarm risk at
    the start of fast-motion bursts.
    """
    times   = np.asarray(times,  float)
    values  = np.asarray(values, float)
    n       = len(values)
    reject  = np.zeros(n, dtype=bool)
    thresh  = np.zeros(n)

    dt    = np.diff(times); dt = np.concatenate([[dt[0]], dt])
    vel   = np.concatenate([[0], np.diff(values)]) / np.maximum(dt, 1e-6)
    v_abs = np.abs(vel)

    for i in range(1, n):
        if vel[i] <= 0:
            thresh[i] = np.inf
            continue
        left, right = _ref_cells(v_abs, i, guard, ref_half)
        noise_l   = np.mean(left)  if len(left)  > 0 else 0.0
        noise_r   = np.mean(right) if len(right) > 0 else 0.0
        noise     = min(noise_l, noise_r) if (len(left)>0 and len(right)>0) \
                    else max(noise_l, noise_r)
        thresh[i] = k * max(noise, 1e-6)
        if vel[i] > thresh[i]:
            reject[i] = True

    return reject, thresh

def hard_velocity_gate(times, values, vmax):
    """Hard threshold baseline for comparison."""
    times  = np.asarray(times, float)
    values = np.asarray(values, float)
    reject = np.zeros(len(values), dtype=bool)
    for i in range(1, len(values)):
        dt = max(times[i]-times[i-1], 1e-6)
        if (values[i]-values[i-1])/dt > vmax:
            reject[i] = True
    return reject

CFAR_VARIANTS = {
    "CA-CFAR":   ca_cfar,
    "OS-CFAR":   os_cfar,
    "GOCA-CFAR": goca_cfar,
    "SOCA-CFAR": soca_cfar,
}
CFAR_COLORS = {
    "CA-CFAR":   "#1f77b4",
    "OS-CFAR":   "#2ca02c",
    "GOCA-CFAR": "#d62728",
    "SOCA-CFAR": "#9467bd",
}

# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset(path):
    with open(path) as f:
        raw = json.load(f)
    pairs = defaultdict(list)
    for r in raw:
        pairs[(r["src"], r["id"])].append(r)
    for k in pairs:
        pairs[k].sort(key=lambda r: r["t"])
    return pairs

def dataset_arrays(pairs):
    t_list, m_list, g_list = [], [], []
    for key in sorted(pairs.keys()):
        recs = pairs[key]
        t_list.append(np.array([r["t"]      for r in recs]))
        m_list.append(np.array([r["range"]  for r in recs]))
        g_list.append(np.array([gt_range(r) for r in recs]))
    return t_list, m_list, g_list

def eval_method(t_list, m_list, g_list, detector_fn, **kwargs):
    """Apply detector, interpolate, compute MAE/RMSE/replace-rate."""
    errors, rrs = [], []
    for t, m, g in zip(t_list, m_list, g_list):
        rej = detector_fn(t, m, **kwargs)
        if isinstance(rej, tuple): rej = rej[0]   # cfar returns (reject, thresh)
        y   = interp_accepted(t, m, rej)
        errors.extend((y - g).tolist())
        rrs.append(rej.mean())
    e = np.array(errors)
    return np.mean(np.abs(e)), np.sqrt(np.mean(e**2)), np.mean(rrs)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data_paths = ["4/multi2_human_nlos.json", "4/multi2_object_nlos.json", "4/multi2_trip_loss.json", 
                "4/opti_multi1_free_circle.json", "4/opti_multi1_passing.json"]
    Path(OUT_DIR).mkdir(exist_ok=True)
    csv_rows = [["dataset","method","k_or_vmax","MAE_m","RMSE_m","replace_pct","delta_vs_raw_pct"]]

    for data_path in data_paths:
        dname = Path(data_path).stem
        print(f"\n{'='*65}")
        print(f"Dataset: {dname}")
        print(f"{'='*65}")

        pairs              = load_dataset(data_path)
        t_list, m_list, g_list = dataset_arrays(pairs)

        # Baselines
        raw_mae = np.mean(np.abs(np.concatenate([m-g for m,g in zip(m_list,g_list)])))
        # Moving median doesn't use detector+interp pattern — evaluate directly
        med_errors = np.concatenate([moving_median(m, MEDIAN_W) - g
                                    for m, g in zip(m_list, g_list)])
        med_mae = np.mean(np.abs(med_errors))

        print(f"  Raw MAE: {raw_mae:.4f} m   "
            f"Moving Median (w={MEDIAN_W}): {med_mae:.4f} m")

        # ── Sweep each CFAR variant over k ───────────────────────────────────────
        cfar_results = {name: [] for name in CFAR_VARIANTS}
        for name, fn in CFAR_VARIANTS.items():
            for k in K_VALUES:
                mae, rmse, rr = eval_method(t_list, m_list, g_list, fn, k=k)
                cfar_results[name].append((k, mae, rmse, rr))
                csv_rows.append([dname, name, k,
                                f"{mae:.4f}", f"{rmse:.4f}", f"{rr*100:.1f}",
                                f"{100*(mae-raw_mae)/raw_mae:.1f}"])

        # ── Sweep hard velocity gate over vmax ───────────────────────────────────
        hard_results = []
        for vmax in VTHRESH:
            mae, rmse, rr = eval_method(t_list, m_list, g_list,
                                        hard_velocity_gate, vmax=vmax)
            hard_results.append((vmax, mae, rmse, rr))
            csv_rows.append([dname, "hard_velocity_gate", vmax,
                            f"{mae:.4f}", f"{rmse:.4f}", f"{rr*100:.1f}",
                            f"{100*(mae-raw_mae)/raw_mae:.1f}"])

        # Print best per method
        print(f"\n  {'Method':<20}  {'Best k/vmax':>12}  {'MAE':>8}  {'replace%':>9}  {'Δ%':>7}")
        print(f"  {'-'*60}")
        for name in CFAR_VARIANTS:
            best = min(cfar_results[name], key=lambda x: x[1])
            print(f"  {name:<20}  k={best[0]:>9.1f}  {best[1]:>8.4f}  "
                f"{best[3]*100:>8.1f}%  {100*(best[1]-raw_mae)/raw_mae:>+6.1f}%")
        best_hard = min(hard_results, key=lambda x: x[1])
        print(f"  {'Hard vgate':<20}  v={best_hard[0]:>8.1f}m/s  {best_hard[1]:>8.4f}  "
            f"{best_hard[3]*100:>8.1f}%  {100*(best_hard[1]-raw_mae)/raw_mae:>+6.1f}%")
        print(f"  {'Moving Median':<20}  {'—':>12}  {med_mae:>8.4f}  "
            f"{'100.0':>8}%  {100*(med_mae-raw_mae)/raw_mae:>+6.1f}%")

        # ════════════════════════════════════════════════════════════════════════
        # Figure 1 — MAE vs k: all CFAR variants + hard threshold + median
        # ════════════════════════════════════════════════════════════════════════
        fig1, ax1 = plt.subplots(figsize=(12, 5.5))
        fig1.suptitle(f"CFAR vs Hard Threshold — MAE vs Scale Factor k\n"
                    f"Dataset: {dname}   Guard={GUARD}  RefHalf={REF_HALF}",
                    fontsize=12)

        for name, color in CFAR_COLORS.items():
            ks   = [r[0] for r in cfar_results[name]]
            maes = [r[1] for r in cfar_results[name]]
            best_k = ks[np.argmin(maes)]
            ax1.plot(ks, maes, color=color, lw=2.0, marker="o", ms=6, label=name)
            ax1.scatter([best_k], [min(maes)],
                        color=color, s=150, zorder=6, marker="*")

        # Hard threshold as secondary x-axis overlay (normalised to same range)
        hard_maes = [r[1] for r in hard_results]
        hard_vmaxs = [r[0] for r in hard_results]
        # Scale hard vmax to same x-axis as k for visual overlay
        k_scaled = np.interp(hard_vmaxs,
                            [min(hard_vmaxs), max(hard_vmaxs)],
                            [min(K_VALUES),   max(K_VALUES)])
        ax1.plot(k_scaled, hard_maes, color="black", lw=1.5, ls="--",
                marker="^", ms=6, alpha=0.7,
                label=f"Hard velocity gate (x-axis re-scaled for overlay)")
        ax1.scatter([k_scaled[np.argmin(hard_maes)]], [min(hard_maes)],
                    color="black", s=150, zorder=6, marker="*")

        ax1.axhline(raw_mae, color="dimgray", lw=1.0, ls=":", label=f"Raw ({raw_mae:.4f}m)")
        ax1.axhline(med_mae, color="#1f77b4", lw=1.8, ls="-.",
                    label=f"Moving Median w={MEDIAN_W} ({med_mae:.4f}m)")

        ax1.set_xlabel("Scale factor k  (threshold = k × local noise estimate)")
        ax1.set_ylabel("MAE [m]")
        ax1.set_title("Lower k = more aggressive detection   Higher k = more permissive\n"
                    "★ = best k per method")
        ax1.legend(fontsize=8, loc="upper right")
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        fig1.savefig(f"{OUT_DIR}/cfar_vs_hard_{dname}.png", dpi=150, bbox_inches="tight")
        plt.close(fig1)
        print(f"\n  Saved: cfar_vs_hard_{dname}.png")

        # ════════════════════════════════════════════════════════════════════════
        # Figure 2 — Adaptive threshold over time on example link
        # Shows what the adaptive threshold looks like vs the hard line
        # ════════════════════════════════════════════════════════════════════════
        # Use the link with the most NLoS events
        link_keys = sorted(pairs.keys())
        example_key = max(link_keys,
                        key=lambda k: np.mean(np.abs(
                            np.array([r["range"] for r in pairs[k]]) -
                            np.array([gt_range(r) for r in pairs[k]]))))
        recs   = pairs[example_key]
        t_ex   = np.array([r["t"]     for r in recs])
        m_ex   = np.array([r["range"] for r in recs])
        g_ex   = np.array([gt_range(r) for r in recs])
        t_rel  = t_ex - t_ex[0]

        # Pick representative k for each CFAR type (best k from sweep)
        best_ks = {name: cfar_results[name][np.argmin([r[1] for r in cfar_results[name]])][0]
                for name in CFAR_VARIANTS}

        # Compute velocity sequence
        dt_ex  = np.concatenate([[np.diff(t_ex)[0]], np.diff(t_ex)])
        vel_ex = np.concatenate([[0], np.diff(m_ex)]) / np.maximum(dt_ex, 1e-6)

        # Best hard vmax
        best_vmax = hard_results[np.argmin([r[1] for r in hard_results])][0]

        fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10),
                                gridspec_kw={"height_ratios": [2.5, 2, 2]})
        fig2.suptitle(f"Adaptive Threshold Over Time — {dname}\n"
                    f"Link src={example_key[0]}→id={example_key[1]}",
                    fontsize=12)

        # Panel A: range + GT + NLoS shading
        ax_r = axes2[0]
        err_ex = m_ex - g_ex
        nlos_ex = err_ex > 0.25
        for i in np.where(nlos_ex)[0]:
            ax_r.axvspan(t_rel[i]-0.15, t_rel[i]+0.15, color="mistyrose", alpha=0.6, lw=0)
        ax_r.plot(t_rel, g_ex, color="black",     lw=2.0, label="GT")
        ax_r.plot(t_rel, m_ex, color="lightgray", lw=2.0, label="Raw", alpha=0.9)
        ax_r.scatter(t_rel, m_ex, color="dimgray", s=16, zorder=3)
        # Overlay cleaned range for best CFAR
        best_cfar_name = min(CFAR_VARIANTS,
                            key=lambda n: min(r[1] for r in cfar_results[n]))
        bk = best_ks[best_cfar_name]
        rej_best, _ = CFAR_VARIANTS[best_cfar_name](t_ex, m_ex, k=bk)
        y_best = interp_accepted(t_ex, m_ex, rej_best)
        ax_r.plot(t_rel, y_best, color=CFAR_COLORS[best_cfar_name], lw=1.8,
                label=f"{best_cfar_name} k={bk} (best CFAR)")
        ax_r.set_ylabel("Range [m]"); ax_r.set_title("Range — pink = GT NLoS")
        ax_r.legend(fontsize=8); ax_r.grid(True, alpha=0.25)

        # Panel B: velocity + adaptive thresholds
        ax_v = axes2[1]
        ax_v.plot(t_rel, vel_ex, color="steelblue", lw=1.3, alpha=0.8,
                label="Range velocity d[i]/dt")
        ax_v.axhline(0, color="k", lw=0.7, ls="--")

        for name, color in CFAR_COLORS.items():
            bk_n = best_ks[name]
            _, thresh_n = CFAR_VARIANTS[name](t_ex, m_ex, k=bk_n)
            thresh_plot = np.where(thresh_n == np.inf, np.nan, thresh_n)
            ax_v.plot(t_rel, thresh_plot, color=color, lw=1.5, ls="--", alpha=0.85,
                    label=f"{name} threshold (k={bk_n})")

        ax_v.axhline(best_vmax, color="black", lw=1.5, ls=":",
                    label=f"Hard gate ({best_vmax} m/s)")
        ax_v.set_ylabel("Velocity [m/s]")
        ax_v.set_title("Adaptive thresholds (dashed) vs hard threshold (dotted)")
        ax_v.legend(fontsize=7, loc="upper right"); ax_v.grid(True, alpha=0.25)

        # Panel C: detections per method
        ax_det = axes2[2]
        y_offset = 0
        det_labels, det_yticks = [], []
        for name, color in CFAR_COLORS.items():
            bk_n  = best_ks[name]
            rej_n = CFAR_VARIANTS[name](t_ex, m_ex, k=bk_n)[0]
            ax_det.scatter(t_rel[rej_n], np.full(rej_n.sum(), y_offset),
                        color=color, s=60, marker="|", linewidths=2.0)
            ax_det.axhline(y_offset, color=color, lw=0.4, alpha=0.3)
            det_labels.append(f"{name} k={bk_n}")
            det_yticks.append(y_offset)
            y_offset += 1

        rej_hard = hard_velocity_gate(t_ex, m_ex, vmax=best_vmax)
        ax_det.scatter(t_rel[rej_hard], np.full(rej_hard.sum(), y_offset),
                    color="black", s=60, marker="|", linewidths=2.0)
        ax_det.axhline(y_offset, color="black", lw=0.4, alpha=0.3)
        det_labels.append(f"Hard gate {best_vmax}m/s")
        det_yticks.append(y_offset)

        # GT NLoS positions
        ax_det.scatter(t_rel[nlos_ex], np.full(nlos_ex.sum(), y_offset+1),
                    color="tomato", s=60, marker="|", linewidths=2.0)
        det_labels.append("GT NLoS (reference)")
        det_yticks.append(y_offset+1)

        ax_det.set_yticks(det_yticks); ax_det.set_yticklabels(det_labels, fontsize=8)
        ax_det.set_xlabel("Time [s]")
        ax_det.set_title("Detection events per method — vertical bars = flagged samples")
        ax_det.grid(True, axis="x", alpha=0.25)

        fig2.tight_layout()
        fig2.savefig(f"{OUT_DIR}/cfar_trace_{dname}.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: cfar_trace_{dname}.png")

    # ── CSV ───────────────────────────────────────────────────────────────────────
    csv_path = f"{OUT_DIR}/cfar_results.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"\nCSV saved: {csv_path}")

    # ── Final summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("OVERALL BEST PER METHOD (concatenated across all datasets)")
    print(f"{'='*70}")
    rows = list(csv.DictReader(open(csv_path)))
    methods_all = ["CA-CFAR","OS-CFAR","GOCA-CFAR","SOCA-CFAR","hard_velocity_gate"]
    for ds in sorted(set(r["dataset"] for r in rows)):
        print(f"\n  {ds}:")
        for method in methods_all:
            sub = [r for r in rows if r["dataset"]==ds and r["method"]==method]
            if not sub: continue
            best = min(sub, key=lambda r: float(r["MAE_m"]))
            print(f"    {method:<22} k/v={best['k_or_vmax']:>6}  "
                f"MAE={best['MAE_m']} m  ({best['delta_vs_raw_pct']}%)")
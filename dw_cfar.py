"""
CFAR Benchmark — Noisy Nodes (IDs 1 & 5) Only
===============================================
Keeps only records whose anchor ID is 1 or 5.

Methods compared (all use interpolation for replacement):
  1. Raw                — no filtering
  2. Moving Median w=5  — robust baseline
  3. Hard velocity gate — constant vmax threshold (swept over VTHRESH)
  4. Mean-vel gate      — k × global_mean(|vel|) threshold (swept over K_VALUES)
  5. CA-CFAR            — Cell Averaging CFAR
  6. OS-CFAR            — Order Statistics CFAR (75th percentile)
  7. GOCA-CFAR          — Greatest-Of Cell Averaging
  8. SOCA-CFAR          — Smallest-Of Cell Averaging
  9. DW+CFAR            — DecaWave 3-state hybrid where grey-area samples are
                          resolved by GOCA-CFAR on the velocity sequence:
                            DW diff > 10 dB  → NLoS  → interpolate
                            DW diff < 6 dB   → LoS   → keep
                            6–10 dB (grey)   → GOCA-CFAR decides:
                                               flagged → interpolate
                                               clear   → keep
                          Swept over CFAR scale factor k.

USAGE
-----
  python uwb_cfar_noisy_nodes.py file1.json [file2.json ...]

OUTPUT (in OUT_DIR)
-------------------
  cfar_noisy_mae_<dataset>.png
  cfar_noisy_trace_<dataset>.png
  cfar_noisy_metrics_<dataset>.png
  cfar_noisy_results.csv
"""

import sys, json, os, csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR   = os.environ.get("OUT_DIR", "cfar_dw_noisy_output4")
KEEP_IDS  = {1, 5}
GUARD     = 1
REF_HALF  = 4
OS_RANK   = 0.75
K_VALUES  = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
VTHRESH   = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
MEDIAN_W  = 5

# DW 3-state thresholds
DW_LOS_THRESH  = 6.0    # dB — below this: LoS
DW_NLOS_THRESH = 10.0   # dB — above this: NLoS
VMAX_GATE      = 2.0    # m/s — one-sided velocity gate fallback inside DW

# ── Helpers ───────────────────────────────────────────────────────────────────
def gt_range(r):
    return float(np.linalg.norm(
        np.array(r["optitrack_src_tx_position"]) -
        np.array(r["optitrack_dst_tx_position"])))

def interp_accepted(times, values, reject):
    out = values.copy().astype(float)
    for i in np.where(reject)[0]:
        l = next((j for j in range(i-1,-1,-1)        if not reject[j]), None)
        r = next((j for j in range(i+1,len(reject))  if not reject[j]), None)
        if   l is not None and r is not None:
            out[i]=out[l]+(out[r]-out[l])*(times[i]-times[l])/(times[r]-times[l])
        elif l is not None: out[i]=out[l]
        elif r is not None: out[i]=out[r]
    return out

def moving_median(x, w=MEDIAN_W):
    x=np.asarray(x,float); n=len(x); y=x.copy(); h=w//2
    for i in range(n): y[i]=np.median(x[max(0,i-h):min(n,i+h+1)])
    return y

def vel_seq(times, values):
    dt = np.concatenate([[np.diff(times)[0]], np.diff(times)])
    return np.concatenate([[0],np.diff(values)])/np.maximum(dt,1e-6)

def ref_cells(vabs, i):
    n=len(vabs)
    return (vabs[max(0,i-GUARD-REF_HALF):max(0,i-GUARD)],
            vabs[min(n,i+GUARD+1):min(n,i+GUARD+1+REF_HALF)])

# ── DW diff (CIR-based NLoS metric) ──────────────────────────────────────────
def dw_diff(r):
    """
    RX_POWER − FP_POWER = 10·log10(maxgrowthcir·2¹⁷ / (F1²+F2²+F3²))
    < 6 dB  → LoS
    6–10 dB → grey (ambiguous)
    > 10 dB → NLoS
    """
    F1=r["firstpathamp1"]; F2=r["firstpathamp2"]; F3=r["firstpathamp3"]
    fp = F1**2+F2**2+F3**2
    if fp <= 0: return np.nan
    return 10*np.log10(r["maxgrowthcir"]*(2**17)/fp)

# ── CFAR detectors (return reject_mask, threshold_array) ─────────────────────
def hard_vgate(times, values, k):
    vel=vel_seq(times,values)
    rej=vel>k; rej[0]=False
    return rej, np.full(len(values),k)

def mean_vgate(times, values, k):
    vel=vel_seq(times,values)
    thr_v=k*max(np.mean(np.abs(vel[1:])),1e-6)
    rej=vel>thr_v; rej[0]=False
    return rej, np.full(len(values),thr_v)

def ca_cfar(times, values, k):
    vel=vel_seq(times,values); vabs=np.abs(vel)
    rej=np.zeros(len(values),bool); thr=np.full(len(values),np.inf)
    for i in range(1,len(values)):
        if vel[i]<=0: continue
        l,r=ref_cells(vabs,i); ref=np.concatenate([l,r])
        if not len(ref): continue
        thr[i]=k*max(np.mean(ref),1e-6)
        if vel[i]>thr[i]: rej[i]=True
    return rej,thr

def os_cfar(times, values, k):
    vel=vel_seq(times,values); vabs=np.abs(vel)
    rej=np.zeros(len(values),bool); thr=np.full(len(values),np.inf)
    for i in range(1,len(values)):
        if vel[i]<=0: continue
        l,r=ref_cells(vabs,i); ref=np.concatenate([l,r])
        if not len(ref): continue
        thr[i]=k*max(np.percentile(ref,OS_RANK*100),1e-6)
        if vel[i]>thr[i]: rej[i]=True
    return rej,thr

def goca_cfar(times, values, k):
    vel=vel_seq(times,values); vabs=np.abs(vel)
    rej=np.zeros(len(values),bool); thr=np.full(len(values),np.inf)
    for i in range(1,len(values)):
        if vel[i]<=0: continue
        l,r=ref_cells(vabs,i)
        nl=np.mean(l) if len(l) else 0.0; nr=np.mean(r) if len(r) else 0.0
        thr[i]=k*max(nl,nr,1e-6)
        if vel[i]>thr[i]: rej[i]=True
    return rej,thr

def soca_cfar(times, values, k):
    vel=vel_seq(times,values); vabs=np.abs(vel)
    rej=np.zeros(len(values),bool); thr=np.full(len(values),np.inf)
    for i in range(1,len(values)):
        if vel[i]<=0: continue
        l,r=ref_cells(vabs,i)
        nl=np.mean(l) if len(l) else 0.0; nr=np.mean(r) if len(r) else 0.0
        noise=min(nl,nr) if (len(l) and len(r)) else max(nl,nr)
        thr[i]=k*max(noise,1e-6)
        if vel[i]>thr[i]: rej[i]=True
    return rej,thr

def dw_cfar_detector(times, values, records, cfar_k):
    """
    DW 3-state hybrid with CFAR grey-area resolution.

    Returns (reject_mask, dw_diff_array) where reject_mask is True for
    samples that should be interpolated:
      DW diff > DW_NLOS_THRESH               → always reject
      DW diff in grey AND CFAR flags it      → reject
      DW diff < DW_LOS_THRESH or CFAR clear  → keep

    Also applies a one-sided velocity gate as fallback for spikes the
    DW diff misses entirely (diff may be in LoS zone but velocity is
    physically impossible).
    """
    times  = np.asarray(times,  float)
    values = np.asarray(values, float)
    n      = len(values)

    diff   = np.array([dw_diff(r) for r in records])
    is_los  = diff < DW_LOS_THRESH
    is_nlos = diff > DW_NLOS_THRESH
    is_grey = ~is_los & ~is_nlos

    # One-sided velocity gate fallback: upward jumps that are physically
    # impossible regardless of what the DW diff says
    for i in range(1, n):
        dt = max(times[i]-times[i-1], 1e-6)
        if (values[i]-values[i-1])/dt > VMAX_GATE:
            is_nlos[i]=True; is_los[i]=False; is_grey[i]=False

    # GOCA-CFAR flags on the velocity sequence
    cfar_flags, cfar_thr = goca_cfar(times, values, cfar_k)

    # Reject mask: DW NLoS + grey samples that CFAR also flags
    reject = is_nlos | (is_grey & cfar_flags)

    return reject, diff, cfar_thr, is_los, is_grey, is_nlos, cfar_flags

# ── Method registry ───────────────────────────────────────────────────────────
# name → (fn, param_label, param_grid, color)
# fn signature: fn(times, values, k) → (reject_mask, threshold_array)
METHODS = {
    "Hard vgate":  (hard_vgate,  "vmax [m/s]",      VTHRESH,  "black"),
    "Mean-vel":    (mean_vgate,  "k × global mean",  K_VALUES, "#8c564b"),
    "CA-CFAR":     (ca_cfar,     "k",                K_VALUES, "#1f77b4"),
    "OS-CFAR":     (os_cfar,     "k",                K_VALUES, "#2ca02c"),
    "GOCA-CFAR":   (goca_cfar,   "k",                K_VALUES, "#d62728"),
    "SOCA-CFAR":   (soca_cfar,   "k",                K_VALUES, "#9467bd"),
}
DW_CFAR_COLOR = "#e377c2"

# ── Sweep functions ───────────────────────────────────────────────────────────
def sweep(t_list, m_list, g_list, fn, params):
    """Standard sweep for methods that only need (times, values, k)."""
    rows=[]
    for p in params:
        errs,rrs=[],[]
        for t,m,g in zip(t_list,m_list,g_list):
            rej,_=fn(t,m,p)
            y=interp_accepted(t,m,rej)
            errs.extend((y-g).tolist()); rrs.append(rej.mean())
        e=np.array(errs)
        rows.append((p,np.mean(np.abs(e)),np.sqrt(np.mean(e**2)),np.mean(rrs)))
    return rows

def sweep_dw_cfar(t_list, m_list, g_list, rec_list, k_values):
    """
    Sweep DW+CFAR over CFAR scale factor k.
    rec_list: list of record arrays (parallel to t_list/m_list/g_list),
              needed to compute dw_diff per record.
    """
    rows=[]
    for k in k_values:
        errs,rrs=[],[]
        for t,m,g,recs in zip(t_list,m_list,g_list,rec_list):
            rej,_,_,_,_,_,_ = dw_cfar_detector(t,m,recs,k)
            y=interp_accepted(t,m,rej)
            errs.extend((y-g).tolist()); rrs.append(rej.mean())
        e=np.array(errs)
        rows.append((k,np.mean(np.abs(e)),np.sqrt(np.mean(e**2)),np.mean(rrs)))
    return rows


if __name__ == "__main__":
    # ── Load ──────────────────────────────────────────────────────────────────────
    data_paths = ["4/multi2_human_nlos.json", "4/multi2_object_nlos.json",
                "4/multi2_trip_loss.json",  "4/opti_multi1_free_circle.json",
                "4/opti_multi1_passing.json"]

    Path(OUT_DIR).mkdir(exist_ok=True)
    csv_rows=[["dataset","method","param","MAE_m","RMSE_m","replace_pct","delta_pct"]]

    for data_path in data_paths:
        dname=Path(data_path).stem
        with open(data_path) as f: raw_data=json.load(f)

        filtered=[r for r in raw_data if r["id"] in KEEP_IDS]
        print(f"\n{dname}: keeping id∈{sorted(KEEP_IDS)} → "
            f"{len(filtered)} records  (dropped {len(raw_data)-len(filtered)})")

        pairs=defaultdict(list)
        for r in filtered: pairs[(r["src"],r["id"])].append(r)
        for k in pairs: pairs[k].sort(key=lambda r:r["t"])
        link_keys=sorted(pairs.keys())
        if not link_keys: print("  No links remain. Skipping."); continue

        t_list  =[np.array([r["t"]      for r in pairs[k]]) for k in link_keys]
        m_list  =[np.array([r["range"]  for r in pairs[k]]) for k in link_keys]
        g_list  =[np.array([gt_range(r) for r in pairs[k]]) for k in link_keys]
        rec_list=[pairs[k]                                   for k in link_keys]

        raw_mae =np.mean(np.abs(np.concatenate([m-g for m,g in zip(m_list,g_list)])))
        med_mae =np.mean(np.abs(np.concatenate([moving_median(m)-g
                                for m,g in zip(m_list,g_list)])))
        med_rmse=np.sqrt(np.mean(np.concatenate([moving_median(m)-g
                                for m,g in zip(m_list,g_list)])**2))
        print(f"  Raw MAE: {raw_mae:.4f} m   Median w={MEDIAN_W}: {med_mae:.4f} m")

        # ── Run all sweeps ────────────────────────────────────────────────────────
        results={}
        for name,(fn,plabel,params,color) in METHODS.items():
            results[name]=sweep(t_list,m_list,g_list,fn,params)
            for p,mae,rmse,rr in results[name]:
                csv_rows.append([dname,name,p,f"{mae:.4f}",f"{rmse:.4f}",
                                f"{rr*100:.1f}",f"{100*(mae-raw_mae)/raw_mae:.1f}"])

        results["DW+CFAR"]=sweep_dw_cfar(t_list,m_list,g_list,rec_list,K_VALUES)
        for p,mae,rmse,rr in results["DW+CFAR"]:
            csv_rows.append([dname,"DW+CFAR",p,f"{mae:.4f}",f"{rmse:.4f}",
                            f"{rr*100:.1f}",f"{100*(mae-raw_mae)/raw_mae:.1f}"])

        # ── Print summary ─────────────────────────────────────────────────────────
        print(f"\n  {'Method':<18}  {'Best param':>12}  {'MAE':>8}  "
            f"{'replace%':>9}  {'Δ%':>7}")
        print(f"  {'-'*62}")
        for name,(fn,plabel,params,color) in METHODS.items():
            best=min(results[name],key=lambda x:x[1])
            print(f"  {name:<18}  {plabel[:4]}={best[0]:>7}  {best[1]:>8.4f}  "
                f"{best[3]*100:>8.1f}%  {100*(best[1]-raw_mae)/raw_mae:>+6.1f}%")
        best_dw=min(results["DW+CFAR"],key=lambda x:x[1])
        print(f"  {'DW+CFAR':<18}  k={best_dw[0]:>9.1f}  {best_dw[1]:>8.4f}  "
            f"{best_dw[3]*100:>8.1f}%  {100*(best_dw[1]-raw_mae)/raw_mae:>+6.1f}%")
        print(f"  {'Median w=5':<18}  {'—':>12}  {med_mae:>8.4f}  "
            f"{'100':>8}%  {100*(med_mae-raw_mae)/raw_mae:>+6.1f}%")

        # ── Figure 1: MAE vs parameter ────────────────────────────────────────────
        fig1,ax1=plt.subplots(figsize=(13,5.5))
        fig1.suptitle(f"CFAR vs Hard Gate vs Mean Gate vs DW+CFAR — IDs {sorted(KEEP_IDS)} Only\n"
                    f"Dataset: {dname}  |  Guard={GUARD}  RefHalf={REF_HALF}  "
                    f"Replacement: interpolation", fontsize=12)

        for name,(fn,plabel,params,color) in METHODS.items():
            maes=[r[1] for r in results[name]]
            x_n=np.linspace(0,1,len(params))
            bi=int(np.argmin(maes))
            ls="--" if name=="Hard vgate" else "-"
            ax1.plot(x_n,maes,color=color,lw=2.0,ls=ls,marker="o",ms=5,
                    label=f"{name}  best={params[bi]} → MAE={maes[bi]:.4f}m")
            ax1.scatter([x_n[bi]],[maes[bi]],color=color,s=150,zorder=6,marker="*")

        # DW+CFAR on the same normalised x-axis (k values same as K_VALUES)
        dw_maes=[r[1] for r in results["DW+CFAR"]]
        x_dw=np.linspace(0,1,len(K_VALUES))
        bi_dw=int(np.argmin(dw_maes))
        ax1.plot(x_dw,dw_maes,color=DW_CFAR_COLOR,lw=2.5,ls="-.",marker="D",ms=6,
                label=f"DW+CFAR  best k={K_VALUES[bi_dw]} → MAE={dw_maes[bi_dw]:.4f}m")
        ax1.scatter([x_dw[bi_dw]],[dw_maes[bi_dw]],
                    color=DW_CFAR_COLOR,s=180,zorder=7,marker="*")

        ax1.axhline(raw_mae,color="dimgray",lw=1.0,ls=":",label=f"Raw ({raw_mae:.4f}m)")
        ax1.axhline(med_mae,color="navy",lw=1.8,ls="-.",
                    label=f"Median w={MEDIAN_W} ({med_mae:.4f}m)")
        ax1.set_xlabel("Parameter (normalised 0→1: low=aggressive, high=permissive)\n"
                    "Hard gate: vmax [m/s]  |  DW+CFAR & others: k × noise estimate")
        ax1.set_ylabel("MAE [m]")
        ax1.set_title("★ = best parameter per method")
        ax1.legend(fontsize=7,loc="upper right"); ax1.grid(True,alpha=0.3)
        fig1.tight_layout()
        fig1.savefig(f"{OUT_DIR}/cfar_noisy_mae_{dname}.png",dpi=150,bbox_inches="tight")
        plt.close(fig1)
        print(f"\n  Saved: cfar_noisy_mae_{dname}.png")

        # ── Figure 2: Threshold trace on worst link ───────────────────────────────
        worst_key=max(link_keys,
                    key=lambda k:np.mean(np.abs(
                        np.array([r["range"] for r in pairs[k]])-
                        np.array([gt_range(r) for r in pairs[k]]))))
        idx=link_keys.index(worst_key)
        t_ex=t_list[idx]; m_ex=m_list[idx]; g_ex=g_list[idx]
        rec_ex=rec_list[idx]
        t_rel=t_ex-t_ex[0]; vel_ex=vel_seq(t_ex,m_ex)
        nlos_ex=(m_ex-g_ex)>0.25

        best_params={name:min(results[name],key=lambda x:x[1])[0] for name in METHODS}
        best_k_dw=min(results["DW+CFAR"],key=lambda x:x[1])[0]

        fig2,axes2=plt.subplots(4,1,figsize=(14,14),
                                gridspec_kw={"height_ratios":[2.5,2,1.5,2]})
        fig2.suptitle(f"Threshold Trace — IDs {sorted(KEEP_IDS)} Only — {dname}\n"
                    f"Link src={worst_key[0]}→id={worst_key[1]} (worst MAE)",fontsize=12)

        # Panel A: range
        ax_r=axes2[0]
        for i in np.where(nlos_ex)[0]:
            ax_r.axvspan(t_rel[i]-0.15,t_rel[i]+0.15,color="mistyrose",alpha=0.6,lw=0)
        ax_r.plot(t_rel,g_ex,color="black",lw=2.0,label="GT")
        ax_r.plot(t_rel,m_ex,color="lightgray",lw=2.0,alpha=0.85,label="Raw")
        ax_r.scatter(t_rel,m_ex,color="dimgray",s=14)
        # Best overall CFAR method
        all_best=[(name,min(results[name],key=lambda x:x[1])[1]) for name in METHODS]
        all_best.append(("DW+CFAR",min(results["DW+CFAR"],key=lambda x:x[1])[1]))
        best_name=min(all_best,key=lambda x:x[1])[0]
        if best_name=="DW+CFAR":
            rej_b,_,_,_,_,_,_=dw_cfar_detector(t_ex,m_ex,rec_ex,best_k_dw)
            best_color=DW_CFAR_COLOR
            best_label=f"DW+CFAR k={best_k_dw} (overall best)"
        else:
            bp=best_params[best_name]
            rej_b,_=METHODS[best_name][0](t_ex,m_ex,bp)
            best_color=METHODS[best_name][3]
            best_label=f"{best_name} param={bp} (overall best)"
        ax_r.plot(t_rel,interp_accepted(t_ex,m_ex,rej_b),
                color=best_color,lw=1.8,label=best_label)
        # Also plot DW+CFAR output
        rej_dw,dw_diff_vals,_,dw_los,dw_grey,dw_nlos,_ = \
            dw_cfar_detector(t_ex,m_ex,rec_ex,best_k_dw)
        if best_name!="DW+CFAR":
            ax_r.plot(t_rel,interp_accepted(t_ex,m_ex,rej_dw),
                    color=DW_CFAR_COLOR,lw=1.4,ls="--",
                    label=f"DW+CFAR k={best_k_dw}")
        ax_r.set_ylabel("Range [m]")
        ax_r.set_title("Range — pink=GT NLoS (err>0.25m)")
        ax_r.legend(fontsize=8); ax_r.grid(True,alpha=0.25)

        # Panel B: DW diff with state shading
        ax_dw=axes2[1]
        valid=np.isfinite(dw_diff_vals)
        ax_dw.axhspan(-5,  DW_LOS_THRESH,  color="#c8e6c9",alpha=0.30)
        ax_dw.axhspan(DW_LOS_THRESH,DW_NLOS_THRESH,color="#fff9c4",alpha=0.40)
        ax_dw.axhspan(DW_NLOS_THRESH,30,   color="#ffcdd2",alpha=0.30)
        ax_dw.axhline(DW_LOS_THRESH, color="green",lw=1.0,ls="--",
                    label=f"LoS ({DW_LOS_THRESH} dB)")
        ax_dw.axhline(DW_NLOS_THRESH,color="red",  lw=1.0,ls="--",
                    label=f"NLoS ({DW_NLOS_THRESH} dB)")
        ax_dw.plot(t_rel[valid],dw_diff_vals[valid],color="navy",lw=1.3,marker="o",ms=3)
        if dw_nlos.any():
            ax_dw.scatter(t_rel[dw_nlos],dw_diff_vals[dw_nlos],
                        color="red",s=50,zorder=5,label=f"DW NLoS ({dw_nlos.sum()})")
        if dw_grey.any():
            ax_dw.scatter(t_rel[dw_grey],dw_diff_vals[dw_grey],
                        color="goldenrod",s=30,zorder=4,
                        label=f"Grey ({dw_grey.sum()})")
        ax_dw.set_ylabel("RX−FP [dB]")
        ax_dw.set_title("DW diff — state classification for DW+CFAR method")
        ax_dw.legend(fontsize=7,loc="upper right"); ax_dw.grid(True,alpha=0.25)

        # Panel C: velocity + CFAR thresholds
        ax_v=axes2[2]
        ax_v.plot(t_rel,vel_ex,color="steelblue",lw=1.3,alpha=0.8,label="Velocity")
        ax_v.axhline(0,color="k",lw=0.7,ls="--")
        for name,(fn,plabel,params,color) in METHODS.items():
            bp_n=best_params[name]; _,thr_n=fn(t_ex,m_ex,bp_n)
            thr_p=np.where(thr_n==np.inf,np.nan,thr_n)
            ls="--" if name=="Hard vgate" else "-." if name=="Mean-vel" else "-"
            ax_v.plot(t_rel,thr_p,color=color,lw=1.3,ls=ls,alpha=0.85,
                    label=f"{name} (param={bp_n})")
        # DW+CFAR GOCA threshold (applied only to grey samples)
        _,_,cfar_thr_dw,_,_,_,_=dw_cfar_detector(t_ex,m_ex,rec_ex,best_k_dw)
        thr_dw_p=np.where(cfar_thr_dw==np.inf,np.nan,cfar_thr_dw)
        # Only show threshold where grey zone active
        thr_dw_masked=np.where(dw_grey,thr_dw_p,np.nan)
        ax_v.plot(t_rel,thr_dw_masked,color=DW_CFAR_COLOR,lw=2.0,ls="-.",
                alpha=0.9,label=f"DW+CFAR GOCA thr k={best_k_dw} (grey only)")
        ax_v.set_ylabel("Velocity [m/s]")
        ax_v.set_title("Velocity + thresholds (DW+CFAR threshold shown only in grey zone)")
        ax_v.legend(fontsize=6,loc="upper right"); ax_v.grid(True,alpha=0.25)

        # Panel D: detection events
        ax_det=axes2[3]
        y_off=0; yticks,ylabels=[],[]
        for name,(fn,plabel,params,color) in METHODS.items():
            bp_n=best_params[name]; rej_n,_=fn(t_ex,m_ex,bp_n)
            ax_det.scatter(t_rel[rej_n],np.full(rej_n.sum(),y_off),
                        color=color,s=60,marker="|",linewidths=2.0)
            ax_det.axhline(y_off,color=color,lw=0.4,alpha=0.3)
            yticks.append(y_off)
            ylabels.append(f"{name} param={bp_n}  ({rej_n.sum()} flags)")
            y_off+=1
        # DW+CFAR detections (with sub-category breakdown)
        rej_dw2,_,_,_,grey2,nlos2,cfar2=dw_cfar_detector(t_ex,m_ex,rec_ex,best_k_dw)
        ax_det.scatter(t_rel[rej_dw2],np.full(rej_dw2.sum(),y_off),
                    color=DW_CFAR_COLOR,s=60,marker="|",linewidths=2.0)
        ax_det.axhline(y_off,color=DW_CFAR_COLOR,lw=0.4,alpha=0.3)
        dw_nlos_only=(nlos2 & ~grey2)
        grey_cfar=(grey2 & cfar2)
        yticks.append(y_off)
        ylabels.append(f"DW+CFAR k={best_k_dw}  "
                    f"({rej_dw2.sum()} total: "
                    f"{dw_nlos_only.sum()} DW-NLoS + "
                    f"{grey_cfar.sum()} grey-CFAR)")
        y_off+=1
        ax_det.scatter(t_rel[nlos_ex],np.full(nlos_ex.sum(),y_off),
                    color="tomato",s=60,marker="|",linewidths=2.0)
        yticks.append(y_off); ylabels.append(f"GT NLoS ({nlos_ex.sum()})")
        ax_det.set_yticks(yticks); ax_det.set_yticklabels(ylabels,fontsize=7)
        ax_det.set_xlabel("Time [s]")
        ax_det.set_title("Detection events per method")
        ax_det.grid(True,axis="x",alpha=0.25)

        fig2.tight_layout()
        fig2.savefig(f"{OUT_DIR}/cfar_noisy_trace_{dname}.png",dpi=150,bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: cfar_noisy_trace_{dname}.png")

        # ── Figure 3: Best-param MAE/RMSE bar chart ───────────────────────────────
        fig3,axes3=plt.subplots(1,2,figsize=(14,4.5))
        fig3.suptitle(f"Best-Param Summary — IDs {sorted(KEEP_IDS)} Only — {dname}",
                    fontsize=12)
        mnames =list(METHODS.keys())+["DW+CFAR",f"Median w={MEDIAN_W}"]
        mcolors=[METHODS[n][3] for n in METHODS]+[DW_CFAR_COLOR,"navy"]
        b_maes =[min(results[n],key=lambda x:x[1])[1] for n in METHODS] + \
                [min(results["DW+CFAR"],key=lambda x:x[1])[1],med_mae]
        b_rmses=[min(results[n],key=lambda x:x[2])[2] for n in METHODS] + \
                [min(results["DW+CFAR"],key=lambda x:x[2])[2],med_rmse]

        for ax,vals,ylab in [(axes3[0],b_maes,"MAE [m]"),(axes3[1],b_rmses,"RMSE [m]")]:
            bars=ax.bar(mnames,vals,color=mcolors,edgecolor="k",lw=0.5)
            ax.axhline(raw_mae,color="dimgray",lw=1.2,ls="--",
                    label=f"Raw ({raw_mae:.4f}m)")
            ax.set_ylabel(ylab)
            ax.set_xticks(range(len(mnames)))
            ax.set_xticklabels(mnames,rotation=20,ha="right",fontsize=8)
            ax.legend(fontsize=8); ax.grid(True,axis="y",alpha=0.3)
            for bar,v in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2,v+0.001,f"{v:.4f}",
                        ha="center",va="bottom",fontsize=7)

        fig3.tight_layout()
        fig3.savefig(f"{OUT_DIR}/cfar_noisy_metrics_{dname}.png",dpi=150,bbox_inches="tight")
        plt.close(fig3)
        print(f"  Saved: cfar_noisy_metrics_{dname}.png")

    # ── CSV ───────────────────────────────────────────────────────────────────────
    csv_path=f"{OUT_DIR}/cfar_noisy_results.csv"
    with open(csv_path,"w",newline="") as f: csv.writer(f).writerows(csv_rows)
    print(f"\nCSV saved: {csv_path}")
# save as: scripts/plot_acc_delta_gtsrb_rgb_vgg_vs_cyvgg.py
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- INPUTS (replace with ResNet vs CyResNet variant if you want) ---
CSV_BASE = "results\\exports_family\\GTSRB_RGB\\micro\\family_curves"
CSV_A = os.path.join(CSV_BASE, "family_acc_vs_delta_VGG19-log.csv")
CSV_B = os.path.join(CSV_BASE, "family_acc_vs_delta_CyVGG19-log.csv")
label_A = "VGG19-log"
label_B = "CyVGG19-log"
out_png = "results\\fig\\acc_delta_gtsrb_rgb_vgglog_vs_cyvgglog.png"

os.makedirs(os.path.dirname(out_png), exist_ok=True)

def load_delta_curve(path):
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    # CASE 1: "narrow" format: delta/acc
    cand_delta = [c for c in df.columns if c.lower() in ("delta", "delta_deg", "theta", "theta_deg")]
    cand_acc   = [c for c in df.columns if c.lower() in ("acc", "accuracy", "acc_mean", "mean", "avg")]
    if cand_delta and cand_acc:
        deg = df[cand_delta[0]].values.astype(float)
        acc = df[cand_acc[0]].values.astype(float)
        # aggregation: if there are multiple rows per delta, average them
        tmp = pd.DataFrame({"deg": deg, "acc": acc}).groupby("deg", as_index=False).mean()
        return np.array(tmp["deg"]), np.array(tmp["acc"])

    # CASE 2: "wide" format: d0, d15, ..., d180
    dcols = [(int(re.sub(r"[^0-9]", "", c)), c) for c in df.columns if re.match(r"^d\d+$", c.lower())]
    if not dcols:
        raise ValueError(f"Unrecognized columns in {path}. Expected 'delta,acc' or 'd0..d180'. Got: {df.columns.tolist()}")
    dcols = sorted(dcols, key=lambda x: x[0])  # sort by degree
    degs = [d for d, _ in dcols]
    # if the file has multiple rows (e.g., several models), average across rows
    arr = df[[c for _, c in dcols]].astype(float).values
    acc = arr.mean(axis=0)
    return np.array(degs, dtype=float), np.array(acc, dtype=float)

def auc_norm(deg, acc):
    # trapezoidal integration over [0,180], normalized by 180
    order = np.argsort(deg)
    deg = deg[order]; acc = acc[order]
    # fill in endpoints if missing
    if deg[0] > 0:
        deg = np.insert(deg, 0, 0.0); acc = np.insert(acc, 0, acc[0])
    if deg[-1] < 180:
        deg = np.append(deg, 180.0); acc = np.append(acc, acc[-1])
    area = np.trapz(acc, deg)
    return float(area / 180.0)

def worst_case(acc):
    return float(np.min(acc)) if len(acc) else float("nan")

degA, accA = load_delta_curve(CSV_A)
degB, accB = load_delta_curve(CSV_B)

# (optional) slight smoothing – comment out if you don’t want it
def smooth(y, w=1):
    if w <= 1: return y
    ypad = np.pad(y, (w//2, w-1-w//2), mode="edge")
    ker = np.ones(w)/w
    return np.convolve(ypad, ker, mode="valid")

accA_s = smooth(accA, w=1)
accB_s = smooth(accB, w=1)

aucA = auc_norm(degA, accA_s)
aucB = auc_norm(degB, accB_s)
worstA = worst_case(accA_s)
worstB = worst_case(accB_s)

print(f"{label_A}: AUC_theta={aucA:.4f}, worst={worstA:.4f}")
print(f"{label_B}: AUC_theta={aucB:.4f}, worst={worstB:.4f}")

plt.figure(figsize=(7.2, 4.5), dpi=150)
plt.plot(degA, accA_s, label=f"{label_A} (AUC={aucA:.3f}, worst={worstA:.3f})")
plt.plot(degB, accB_s, label=f"{label_B} (AUC={aucB:.3f}, worst={worstB:.3f})")
plt.xlabel(r"$\Delta\theta$ [deg]")
plt.ylabel("Acc (micro)")
plt.title("GTSRB_RGB — Acc($\\Delta\\theta$): VGG19-log vs CyVGG19-log")
plt.xlim(0, 180); plt.ylim(0.0, 1.0)
plt.grid(True, alpha=0.3)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(out_png)
print(f"Saved: {out_png}")

# --- Alternative variant (ResNet vs CyResNet) ---
# CSV_A = os.path.join(CSV_BASE, "family_acc_vs_delta_ResNet56-log.csv")
# CSV_B = os.path.join(CSV_BASE, "family_acc_vs_delta_CyResNet56-log.csv")
# label_A = "ResNet56-log"; label_B = "CyResNet56-log"
# out_png = "results/fig/acc_delta_gtsrb_rgb_reslog_vs_cyreslog.png"

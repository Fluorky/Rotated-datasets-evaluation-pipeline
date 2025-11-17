# save as: scripts/plot_perf_vs_time_scatter.py
import csv, os, re
import matplotlib.pyplot as plt

# Optional: automatic label de-overlap (pip install adjustText)
try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except Exception:
    HAS_ADJUSTTEXT = False

# ----------------------- CONFIG -------------------------------------------
# DATASET = "GTSRB"           # MNIST | GTSRB | GTSRB_RGB | LEGO
DATASET = "LEGO"           # MNIST | GTSRB | GTSRB_RGB | LEGO
METRIC  = "micro"
ROOT    = f"results/exports/{DATASET}/{METRIC}"
OUTPNG  = f"results/fig/perf_vs_time_scatter_{DATASET}_{METRIC}.png"
LABEL_MAX = 15

os.makedirs("results/fig", exist_ok=True)

# ----------------------- HELPERS ------------------------------------------
DATASET_PREFIXES = ("lego-", "gtsrb-", "gtsrb_rgb-", "mnist-")

_can_key_re = re.compile(
    r"(?P<family>(?:cy)?(?:resnet(?:[-_ ]?56)?|vgg(?:[-_ ]?19)))"
    r"(?:[-_ ])?"
    r"(?P<trans>(?:linearpolar|logpolar))",
    re.IGNORECASE,
)

def norm(s: str) -> str:
    return (s or "").strip()

def strip_dataset_prefix(s: str) -> str:
    x = s.lower().strip()
    for pref in DATASET_PREFIXES:
        if x.startswith(pref):
            return x[len(pref):]
    return x

def canonical_key(model: str) -> str:
    """
    Build a robust, comparable key like:
      'cyresnet56-linearpolar' or 'vgg19-logpolar'
    Works even if the original string has prefixes/suffixes.
    """
    s = strip_dataset_prefix(model)
    m = _can_key_re.search(s)
    if not m:
        return s  # fallback to the stripped string
    fam = re.sub(r"[\s_]+", "", m.group("family").lower())
    trans = m.group("trans").lower()
    # normalize variants (resnet -> resnet56 if 56 present in string)
    if fam.startswith("resnet") and "56" in fam:
        fam = "resnet56"
    if fam.startswith("cyresnet") and "56" in fam:
        fam = "cyresnet56"
    if fam in ("vgg19", "cyvgg19", "resnet56", "cyresnet56"):
        pass
    return f"{fam}-{trans}"

def infer_family_from_ckey(ckey: str) -> str:
    s = ckey.lower()
    if s.startswith("cyresnet"):  return "CyResNet56"
    if s.startswith("resnet"):    return "ResNet56"
    if s.startswith("cyvgg19"):   return "CyVGG19"
    if s.startswith("vgg19"):     return "VGG19"
    return "Other"

PALETTE = {
    "CyVGG19":    "tab:blue",
    "VGG19":      "tab:orange",
    "CyResNet56": "tab:green",
    "ResNet56":   "tab:red",
    "Other":      "tab:gray",
}

# ----------------------- LOAD DATA ----------------------------------------
# 1) ranking_quality.csv  -> avg accuracy
avg_by_model_exact = {}   # exact model -> avg
avg_by_cankey       = {}  # canonical  -> avg
with open(os.path.join(ROOT, "ranking_quality.csv"), newline="", encoding="utf-8") as f:
    r = csv.reader(f)
    header = next(r)
    hidx = {h: i for i, h in enumerate(header)}  # expects columns incl. 'model', 'avg'
    for row in r:
        model = norm(row[hidx["model"]])
        avg   = float(row[hidx["avg"]])
        avg_by_model_exact[model] = avg
        avg_by_cankey[canonical_key(model)] = avg

# 2) ranking_timeaware_avgperf.csv  -> avg_perf (join with (1))
pts = []  # (model_display, family, avg, avg_perf)
missing = []  # models present in timeaware but not matched to avg

with open(os.path.join(ROOT, "ranking_timeaware_avgperf.csv"), newline="", encoding="utf-8") as f:
    r = csv.reader(f)
    header = next(r)  # expects: model,avg_perf,robust_perf,time_s
    hidx = {h: i for i, h in enumerate(header)}
    for row in r:
        model_raw = norm(row[hidx["model"]])
        avg_perf  = float(row[hidx["avg_perf"]])

        # 1) try exact join
        if model_raw in avg_by_model_exact:
            avg = avg_by_model_exact[model_raw]
            ckey = canonical_key(model_raw)
            fam  = infer_family_from_ckey(ckey)
            pts.append((model_raw, fam, avg, avg_perf))
            continue

        # 2) try canonical join
        ckey = canonical_key(model_raw)
        if ckey in avg_by_cankey:
            avg = avg_by_cankey[ckey]
            fam = infer_family_from_ckey(ckey)
            pts.append((model_raw, fam, avg, avg_perf))
            continue

        # not matched
        missing.append(model_raw)

if not pts:
    raise SystemExit("No join between CSVs. Check file contents and paths.")

# diagnostics
from collections import Counter
fam_counts = Counter([fam for _, fam, _, _ in pts])
print("[INFO] Points per family:", dict(fam_counts))
if missing:
    # show only a few to avoid flooding
    print(f"[WARN] Unmatched models in timeaware (showing up to 10 of {len(missing)}):")
    for m in missing[:10]:
        print("   -", m, "=> canonical:", canonical_key(m))

# ----------------------- LABEL SELECTION ----------------------------------
# top by rank sum
sorted_by_avg      = sorted(pts, key=lambda x: -x[2])
sorted_by_avg_perf = sorted(pts, key=lambda x: -x[3])
rank_avg      = {m: i for i, (m, *_ ) in enumerate(sorted_by_avg)}
rank_avg_perf = {m: i for i, (m, *_ ) in enumerate(sorted_by_avg_perf)}
score = {m: rank_avg[m] + rank_avg_perf[m] for (m, *_ ) in pts}
top_by_rank = sorted(pts, key=lambda x: score[x[0]])[:10]

# Pareto frontier
def dominated(p, others):
    _, _, ax, px = p
    for _, _, ay, py in others:
        if (ay >= ax and py >= px) and (ay > ax or py > px):
            return True
    return False

pareto = [p for p in pts if not dominated(p, pts)]
labels_set = {m for m, *_ in top_by_rank}
for m, *_ in pareto:
    if len(labels_set) >= LABEL_MAX: break
    labels_set.add(m)
to_label = [(m, fam, a, p) for (m, fam, a, p) in pts if m in labels_set]

# ----------------------- PLOT ---------------------------------------------
plt.figure(figsize=(9.8, 7.4), dpi=140)
ax = plt.gca()

# scatter by family
from collections import defaultdict
by_fam = defaultdict(list)
for m, fam, a, p in pts:
    by_fam[fam].append((a, p))
for fam, arr in by_fam.items():
    xs = [a for a, _ in arr]
    ys = [p for _, p in arr]
    ax.scatter(xs, ys, s=28, alpha=0.75, label=fam, color=PALETTE.get(fam, "tab:gray"))

# labels
texts = []
for (m, fam, a, p) in to_label:
    t = ax.text(
        a, p, m, fontsize=8, ha="left", va="bottom", clip_on=False, color="black",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5)
    )
    texts.append(t)

if texts:
    if HAS_ADJUSTTEXT:
        adjust_text(
            texts, ax=ax,
            expand_points=(1.2, 1.4),
            expand_text=(1.1, 1.2),
            arrowprops=dict(arrowstyle="-", lw=0.6, alpha=0.6)
        )
    else:
        # simple repel
        for _ in range(200):
            moved = False
            for i, t1 in enumerate(texts):
                x1, y1 = t1.get_position()
                for j, t2 in enumerate(texts):
                    if j <= i: continue
                    x2, y2 = t2.get_position()
                    if abs(x2 - x1) < 0.01 and abs(y2 - y1) < 0.00002:
                        t1.set_position((x1 - 0.002, y1 + 0.00002))
                        t2.set_position((x2 + 0.002, y2 - 0.00002))
                        moved = True
            if not moved: break

ax.set_xlabel("Avg accuracy")
ax.set_ylabel("Avg perf (mean(acc) / train_time)")
ax.set_title(f"{DATASET} — per-time vs average (metric={METRIC})")
ax.grid(True, alpha=0.25)
ax.legend(title="Family", frameon=True, loc="best")
plt.tight_layout()
plt.savefig(OUTPNG)
print(f"Saved: {OUTPNG}")

"""
Responsible Outlier Detection — Python Implementation
======================================================
Covers every method from the slide deck:
  Univariate  : IQR fences, z-score, modified-z (MAD), robust-z (Qn / Sn)
  Multivariate: Classical Mahalanobis, Robust Mahalanobis (MCD),
                PCA Hotelling T², Local Outlier Factor (LOF)

Workflow follows the PDF's recommended protocol:
  Detect → Diagnose → Decide & Document
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.covariance import MinCovDet, EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# 0. Synthetic dataset (replace with your own)
# ─────────────────────────────────────────────
rng = np.random.default_rng(42)

n_inliers  = 200
n_outliers = 10

# Inlier cloud  – bivariate normal
X_in  = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], n_inliers)
# Outlier cloud – shifted and sparse
X_out = rng.multivariate_normal([4, -4], [[0.3, 0], [0, 0.3]], n_outliers)

X      = np.vstack([X_in, X_out])        # (210, 2)
labels = np.array([0]*n_inliers + [1]*n_outliers)   # 1 = true outlier

df = pd.DataFrame(X, columns=["x1", "x2"])
df["true_outlier"] = labels

print("=" * 60)
print("RESPONSIBLE OUTLIER DETECTION — Python Demo")
print("=" * 60)
print(f"\nDataset: {len(df)} observations | {n_outliers} injected outliers\n")


# ══════════════════════════════════════════════════════════════
# SECTION 1 — UNIVARIATE METHODS  (applied to x1)
# ══════════════════════════════════════════════════════════════
print("─" * 60)
print("SECTION 1 · Univariate Outlier Detection on x1")
print("─" * 60)

x = df["x1"].values


# ── 1a. IQR Fences (Tukey / NIST) ──────────────────────────
Q1, Q3 = np.percentile(x, 25), np.percentile(x, 75)
IQR    = Q3 - Q1

mild_lo,  mild_hi  = Q1 - 1.5*IQR,  Q3 + 1.5*IQR
extreme_lo, extreme_hi = Q1 - 3.0*IQR, Q3 + 3.0*IQR

iqr_mild    = (x < mild_lo)    | (x > mild_hi)
iqr_extreme = (x < extreme_lo) | (x > extreme_hi)

print(f"\n[IQR Fences]  Q1={Q1:.2f}  Q3={Q3:.2f}  IQR={IQR:.2f}")
print(f"  Mild    fence : [{mild_lo:.2f}, {mild_hi:.2f}]  → {iqr_mild.sum()} flagged")
print(f"  Extreme fence : [{extreme_lo:.2f}, {extreme_hi:.2f}]  → {iqr_extreme.sum()} flagged")


# ── 1b. Classic z-score  (|z| > 3) ──────────────────────────
z_scores = np.abs(stats.zscore(x))
z_flags  = z_scores > 3

print(f"\n[z-score |z|>3]  → {z_flags.sum()} flagged")


# ── 1c. Modified z-score via MAD  (|M| > 3.5) ───────────────
median_x = np.median(x)
MAD      = np.median(np.abs(x - median_x))
M_scores = 0.6745 * np.abs(x - median_x) / MAD   # Iglewicz & Hoaglin
mad_flags = M_scores > 3.5

print(f"\n[Modified z (MAD)]  Median={median_x:.2f}  MAD={MAD:.2f}")
print(f"  |M| > 3.5  → {mad_flags.sum()} flagged")


# ── 1d. Robust z via Qn estimator (Rousseeuw & Croux 1993) ──
def qn_scale(x):
    """
    Qn robust scale estimator (large-sample form, no finite-sample correction).
    Qn = 2.2219 * kth smallest |xi - xj| for i < j,
    where k = C(floor(n/2)+1, 2).
    """
    x   = np.asarray(x, dtype=float)
    n   = len(x)
    h   = n // 2 + 1
    k   = h * (h - 1) // 2          # k-th order statistic of pairwise diffs
    diffs = np.abs(x[:, None] - x[None, :])
    upper = diffs[np.triu_indices(n, k=1)]
    upper_sorted = np.sort(upper)
    return 2.2219 * upper_sorted[k - 1]   # 1-indexed → k-1


def sn_scale(x):
    """
    Sn robust scale estimator (large-sample form).
    Sn = 1.1926 * median_i( median_j |xi - xj| )
    """
    x   = np.asarray(x, dtype=float)
    n   = len(x)
    inner = [np.median(np.abs(x[i] - x)) for i in range(n)]
    return 1.1926 * np.median(inner)


Qn = qn_scale(x)
Sn = sn_scale(x)

robust_z_qn = np.abs(x - median_x) / Qn
robust_z_sn = np.abs(x - median_x) / Sn

qn_flags = robust_z_qn > 3.5
sn_flags = robust_z_sn > 3.5

print(f"\n[Robust z — Qn]  Qn={Qn:.3f}  |r|>3.5  → {qn_flags.sum()} flagged")
print(f"[Robust z — Sn]  Sn={Sn:.3f}  |r|>3.5  → {sn_flags.sum()} flagged")


# ══════════════════════════════════════════════════════════════
# SECTION 2 — MULTIVARIATE METHODS  (x1 + x2)
# ══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("SECTION 2 · Multivariate Outlier Detection on (x1, x2)")
print("─" * 60)

alpha = 0.975                   # 2.5 % false-positive rate
p     = X.shape[1]              # number of features
chi2_cutoff = stats.chi2.ppf(alpha, df=p)
print(f"\nχ²({p}) cutoff at α={alpha} : {chi2_cutoff:.3f}")


# ── 2a. Classical Mahalanobis Distance ──────────────────────
cov_classic  = EmpiricalCovariance().fit(X)
md2_classic  = cov_classic.mahalanobis(X)          # squared distances
md_classic   = np.sqrt(md2_classic)
classic_flags = md2_classic > chi2_cutoff

print(f"\n[Classical MD²]  flagged (>χ² cutoff): {classic_flags.sum()}")


# ── 2b. Robust Mahalanobis Distance (MCD) ───────────────────
mcd          = MinCovDet(support_fraction=0.75, random_state=42).fit(X)
md2_robust   = mcd.mahalanobis(X)
md_robust    = np.sqrt(md2_robust)
robust_flags = md2_robust > chi2_cutoff

print(f"[Robust MD² (MCD)]  flagged: {robust_flags.sum()}")


# ── 2c. PCA Hotelling T² ────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

k   = p                          # keep all components
pca = PCA(n_components=k).fit(X_scaled)
Z   = pca.transform(X_scaled)   # PC scores
lam = pca.explained_variance_   # eigenvalues

T2       = np.sum((Z ** 2) / lam[None, :], axis=1)
chi2_k   = stats.chi2.ppf(alpha, df=k)
pca_flags = T2 > chi2_k

print(f"[PCA Hotelling T²]  χ²({k}) cutoff={chi2_k:.3f}  flagged: {pca_flags.sum()}")


# ── 2d. Local Outlier Factor (LOF) ──────────────────────────
lof        = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_preds  = lof.fit_predict(X)          # -1 = outlier, 1 = inlier
lof_scores = -lof.negative_outlier_factor_  # higher = more anomalous
lof_flags  = lof_preds == -1

# Empirical quantile approach (q = 0.975 → top 2.5%)
q_lof      = np.quantile(lof_scores, 0.975)
lof_q_flags = lof_scores > q_lof

print(f"[LOF]  contamination=5 %  flagged: {lof_flags.sum()}")
print(f"[LOF empirical q=0.975]  cutoff={q_lof:.3f}  flagged: {lof_q_flags.sum()}")


# ══════════════════════════════════════════════════════════════
# SECTION 3 — SUMMARY TABLE
# ══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("SECTION 3 · Summary — flags per observation (first 20)")
print("─" * 60)

summary = pd.DataFrame({
    "true_outlier" : labels,
    "IQR_mild"     : iqr_mild.astype(int),
    "IQR_extreme"  : iqr_extreme.astype(int),
    "z_score"      : z_flags.astype(int),
    "modified_z"   : mad_flags.astype(int),
    "robust_z_Qn"  : qn_flags.astype(int),
    "robust_z_Sn"  : sn_flags.astype(int),
    "classic_MD"   : classic_flags.astype(int),
    "robust_MD_MCD": robust_flags.astype(int),
    "PCA_T2"       : pca_flags.astype(int),
    "LOF"          : lof_flags.astype(int),
})

print(summary.tail(15).to_string())   # last 15 rows contain the injected outliers

# Precision / recall vs injected labels (multivariate methods)
def pr(flags, true):
    tp = ((flags == 1) & (true == 1)).sum()
    fp = ((flags == 1) & (true == 0)).sum()
    fn = ((flags == 0) & (true == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    return prec, rec

print("\n[Detection quality vs injected outliers]")
print(f"{'Method':<22} {'Precision':>10} {'Recall':>10}")
for name, flags in [
    ("classic_MD",    classic_flags),
    ("robust_MD_MCD", robust_flags),
    ("PCA_T2",        pca_flags),
    ("LOF",           lof_flags),
]:
    prec, rec = pr(flags.astype(int), labels)
    print(f"  {name:<20} {prec:>10.2f} {rec:>10.2f}")


# ══════════════════════════════════════════════════════════════
# SECTION 4 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Responsible Outlier Detection — Python Demo", fontsize=15, fontweight="bold")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Boxplot of x1 ──
ax1 = fig.add_subplot(gs[0, 0])
ax1.boxplot(x, vert=True, patch_artist=True,
            boxprops=dict(facecolor="#d0e8f1"),
            flierprops=dict(marker="o", color="red", markersize=5))
ax1.set_title("Boxplot x1 (IQR fences)")
ax1.set_ylabel("x1")

# ── Plot 2: Modified z distribution ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(M_scores, bins=30, color="#a8d8ea", edgecolor="white")
ax2.axvline(3.5, color="red", linestyle="--", label="|M|=3.5")
ax2.set_title("Modified z-scores (MAD)")
ax2.set_xlabel("|M_i|")
ax2.legend(fontsize=8)

# ── Plot 3: Qn vs Sn robust scale ──
ax3 = fig.add_subplot(gs[0, 2])
methods  = ["MAD", "Qn", "Sn"]
scales   = [MAD,   Qn,  Sn]
colours  = ["#f6a192", "#a8d8ea", "#b5e7a0"]
bars     = ax3.bar(methods, scales, color=colours, edgecolor="grey")
ax3.set_title("Robust Scale Estimators\n(x1)")
ax3.set_ylabel("Scale value")
for bar, val in zip(bars, scales):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
             f"{val:.2f}", ha="center", va="bottom", fontsize=9)

# ── Plot 4: Classical vs Robust MD² ──
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(md2_classic, md2_robust,
            c=labels, cmap="bwr", alpha=0.6, edgecolors="k", linewidths=0.4, s=30)
ax4.axvline(chi2_cutoff, color="blue",  linestyle="--", label=f"χ² cutoff={chi2_cutoff:.1f}")
ax4.axhline(chi2_cutoff, color="green", linestyle="--")
ax4.set_xlabel("Classical MD²")
ax4.set_ylabel("Robust MD² (MCD)")
ax4.set_title("Classical vs Robust MD²\n(red = true outlier)")
ax4.legend(fontsize=7)

# ── Plot 5: PCA T² ──
ax5 = fig.add_subplot(gs[1, 1])
colors5 = ["red" if f else "#5b9bd5" for f in pca_flags]
ax5.scatter(range(len(T2)), T2, c=colors5, s=20, alpha=0.7)
ax5.axhline(chi2_k, color="orange", linestyle="--", label=f"χ²({k}) cutoff={chi2_k:.1f}")
ax5.set_title("PCA Hotelling T²")
ax5.set_xlabel("Observation index")
ax5.set_ylabel("T²")
ax5.legend(fontsize=8)

# ── Plot 6: LOF scores ──
ax6 = fig.add_subplot(gs[1, 2])
colors6 = ["red" if f else "#5b9bd5" for f in lof_flags]
ax6.scatter(range(len(lof_scores)), lof_scores, c=colors6, s=20, alpha=0.7)
ax6.axhline(q_lof, color="purple", linestyle="--", label=f"q=0.975 → {q_lof:.2f}")
ax6.set_title("LOF Scores")
ax6.set_xlabel("Observation index")
ax6.set_ylabel("LOF score")
ax6.legend(fontsize=8)

# ── Plot 7: Bivariate scatter — all four MV methods ──
titles_mv  = ["Classical MD", "Robust MD (MCD)", "PCA Hotelling T²", "LOF"]
flags_mv   = [classic_flags,  robust_flags,       pca_flags,          lof_flags]
positions  = [(2, 0), (2, 1), (2, 2)]   # only 3 remaining cells → squeeze last 2 together

for i, (title, flags) in enumerate(zip(titles_mv[:3], flags_mv[:3])):
    row, col = positions[i]
    ax = fig.add_subplot(gs[row, col])
    ax.scatter(X[~flags, 0], X[~flags, 1],
               color="#a8d8ea", edgecolors="grey", linewidths=0.3, s=20, label="inlier", alpha=0.7)
    ax.scatter(X[flags, 0],  X[flags, 1],
               color="red",   edgecolors="darkred", linewidths=0.5, s=50, label="flagged", zorder=5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(fontsize=7, loc="upper right")

plt.savefig("/mnt/user-data/outputs/outlier_detection_plots.png", dpi=150, bbox_inches="tight")
print("\n\nPlot saved → outlier_detection_plots.png")
print("\nDone. Follow the Detect → Diagnose → Decide & Document workflow before removing any points.")

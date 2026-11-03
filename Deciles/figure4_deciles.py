"""Figure 4: aggregated distribution of E&SE Asian dialogue across running time, as the mean per-film
share of each film's E&SE Asian dialogue segments falling in each tenth of its running time (19 films,
844 segments, every film weighted equally).
Input: ../Scatter Plot/DATA_Distribution.csv (the same positions as Figure 3).
Bins are tenths: [0,0.1), [0.1,0.2), ... [0.9,1.0].
Also writes deciles_per_film.csv (segment counts per film and tenth) and decile_summary.csv
(raw segments, films present and mean per-film share per tenth), the numbers quoted in Section 4.3.
Run from this folder: python3 figure4_deciles.py"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

d = pd.read_csv("../Scatter Plot/DATA_Distribution.csv", encoding="utf-8-sig")
d.columns = ["Distribution", "Film NO.", "x", "Name"]
name_map = {int(float(k)): v for k, v in zip(d["Film NO."], d["Name"]) if isinstance(v, str)}
s = d[d.Distribution.notna()].copy()
s["f"] = s["Film NO."].astype(float).astype(int)
s["Distribution"] = s.Distribution.astype(str).str.rstrip("%").astype(float) / 100
s["dec"] = np.minimum((s.Distribution * 10).astype(int), 9)
M = s.groupby(["f", "dec"]).size().unstack(fill_value=0).reindex(columns=range(10), fill_value=0)
share = M.div(M.sum(axis=1), axis=0) * 100
mean_share = share.mean().values
films_present = (M > 0).sum().values
print("n films:", len(M), "segments:", int(M.values.sum()))
print("mean share:", mean_share.round(1).tolist(), "first30 %.1f first40 %.1f" % (mean_share[:3].sum(), mean_share[:4].sum()))
print("films present:", films_present.tolist())
M.index = [name_map[i] for i in M.index]; M.columns = [f"{10*i}-{10*i+10}%" for i in range(10)]
M.to_csv("deciles_per_film.csv")
pd.DataFrame({"decile": M.columns, "segments": M.sum().values, "films_present": films_present,
              "mean_share_per_film_pct": mean_share.round(1)}).to_csv("decile_summary.csv", index=False)

cmap = LinearSegmentedColormap.from_list("white_red_soft", ["#ffffff", "#ffe6e6", "#ff9999", "#ff4d4d", "#cc0000"])

# Layout follows the original Figure 7: one strip, vertical colour bar on the right.
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
fig, ax = plt.subplots(figsize=(15, 5.5), dpi=300)
im = ax.imshow(mean_share[np.newaxis, :], cmap=cmap, vmin=0, aspect="auto", extent=(0, 100, 0, 1))
for i in range(1, 10):
    ax.axvline(i * 10, color="#e0e0e0", linewidth=0.8)
ax.set_xlim(0, 100); ax.set_ylim(0, 1)
ax.set_xticks(range(0, 101, 10)); ax.set_yticks([])
ax.tick_params(axis="x", labelsize=13)
ax.set_xlabel("Film Progress (%)", fontsize=14)
ax.set_ylabel("19 films", fontsize=14)
for sp in ax.spines.values():
    sp.set_visible(True); sp.set_linewidth(1.0); sp.set_color("black")
cb = fig.colorbar(im, ax=ax, pad=0.01, aspect=30)
cb.set_label("Share of dialogue (%)", fontsize=14)
cb.ax.tick_params(labelsize=12)
fig.tight_layout()
fig.savefig("Figure 4.png", dpi=300, facecolor="white")
print("saved")

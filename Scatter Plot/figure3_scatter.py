"""Figure 3: every E&SE Asian dialogue segment (subtitle line) as a relative position within its film's
running time, for the 19 films in which E&SE Asian actors speak (844 segments).
Input: DATA_Distribution.csv (columns: position as a percentage, film number, blank, film name on the
name rows). Run from this folder: python3 figure3_scatter.py"""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

df = pd.read_csv("DATA_Distribution.csv", encoding="utf-8-sig")
df.columns = ["Distribution", "Film NO.", "x", "Name"]
names = df[["Film NO.", "Name"]].dropna()
name_map = {int(float(k)): v for k, v in zip(names["Film NO."], names["Name"])}
s = df[df["Distribution"].notna() & df["Film NO."].notna()].copy()
s["Film NO."] = s["Film NO."].astype(float).astype(int)
s["pos"] = s["Distribution"].astype(str).str.rstrip("%").astype(float)
print("segments:", len(s), "films:", s["Film NO."].nunique())
films = sorted(s["Film NO."].unique())
colors = dict(zip(films, sns.color_palette("husl", len(films))))
plt.figure(figsize=(15, 10))
plt.scatter(s["pos"], s["Film NO."], c=s["Film NO."].map(colors), s=30, alpha=0.9)
plt.yticks(ticks=sorted(name_map), labels=[name_map[i] for i in sorted(name_map)], fontsize=12)
plt.xticks(ticks=range(0, 101, 10))
plt.xlabel("Film Progress (%)", fontsize=14); plt.ylabel("Film & Release Date", fontsize=14)
plt.grid(True, which="major", axis="both", linestyle="-", linewidth=0.5, color="#e0e0e0")
ax = plt.gca(); ax.set_facecolor("white")
for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(1.0)
plt.xlim(-2, 102); plt.ylim(s["Film NO."].min() - 1, s["Film NO."].max() + 1)
plt.tight_layout(); plt.savefig("Figure 3.png", dpi=300, bbox_inches="tight"); print("saved Figure 3.png")

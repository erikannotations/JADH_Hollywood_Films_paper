"""Figure 1: total time of spoken dialogue by E&SE Asian actors and all other actors per film,
ordered by release date, with a descriptive linear fit on each series (x = film index in release order).
Input: DATA_Total Time of Dialogues.csv (Name, release date, All Other h:mm:ss, E&SE Asian h:mm:ss).
Run from this folder: python3 figure1_dialogue_time.py"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats

df = pd.read_csv("DATA_Total Time of Dialogues.csv", encoding="utf-8-sig")
df["Time"] = pd.to_datetime(df["Time"], format="%B %d, %Y")
df = df.sort_values("Time").reset_index(drop=True)
to_s = lambda t: sum(int(v) * m for v, m in zip(str(t).split(":"), (3600, 60, 1)))
other = df["All Other"].map(to_s).astype(float); ese = df["E&SE Asian"].map(to_s).astype(float)
x = np.arange(len(df))
fo = stats.linregress(x, other); fe = stats.linregress(x, ese)
print(f"n = {len(df)} films; all other: {fo.slope:+.1f} s/film, r2 = {fo.rvalue**2:.2f} (p = {fo.pvalue:.2f}); "
      f"E&SE Asian: {fe.slope:+.1f} s/film, r2 = {fe.rvalue**2:.2f} (p = {fe.pvalue:.2f})")

hms = FuncFormatter(lambda v, p: f"{int(v)//3600:02d}:{(int(v)%3600)//60:02d}:{int(v)%60:02d}")
BLUE, RED = "#2f7fc1", "#fa7f6f"
fig, ax1 = plt.subplots(figsize=(13, 8))
ax1.plot(x, other, color=BLUE, linestyle=":", linewidth=2.5, label="All other actors")
ax1.plot(x, fo.intercept + fo.slope * x, color=BLUE, linestyle="--",
         label=f"All other actors, linear fit: {fo.slope:+.1f} s per film, $r^2$ = {fo.rvalue**2:.2f}")
ax1.set_ylabel("All other actors: dialogue time (h:mm:ss)", color=BLUE); ax1.tick_params(axis="y", labelcolor=BLUE)
ax1.yaxis.set_major_formatter(hms); ax1.set_ylim(0, other.max() * 1.2)
ax2 = ax1.twinx()
ax2.plot(x, ese, color=RED, label="E&SE Asian actors")
ax2.plot(x, fe.intercept + fe.slope * x, color=RED, linestyle="--",
         label=f"E&SE Asian actors, linear fit: {fe.slope:+.1f} s per film, $r^2$ = {fe.rvalue**2:.2f}")
ax2.set_ylabel("E&SE Asian actors: dialogue time (h:mm:ss)", color=RED); ax2.tick_params(axis="y", labelcolor=RED)
ax2.yaxis.set_major_formatter(hms); ax2.set_ylim(0, ese.max() * 1.5)
ax1.set_xticks(x); ax1.set_xticklabels(df["Name"], rotation=60, ha="right", fontsize=9)
ax1.set_xlabel("Film (ordered by release date)", labelpad=10)
ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
plt.tight_layout(); plt.savefig("Figure 1.png", dpi=300, bbox_inches="tight"); print("saved Figure 1.png")

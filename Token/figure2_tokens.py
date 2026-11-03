"""Figure 2: word tokens of direct speech by E&SE Asian actors and all other actors per film,
ordered by release date, with a descriptive linear fit on each series (x = film index in release order).
Input: DATA_Token.csv (whitespace token counts, Python str.split(), per film).
Run from this folder: python3 figure2_tokens.py"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("DATA_Token.csv", encoding="utf-8-sig")
df["Time"] = pd.to_datetime(df["Time"], format="%B %d, %Y")
df = df.sort_values("Time").reset_index(drop=True)
other = df["All Other"].astype(float); ese = df["E&SE Asian"].astype(float)
x = np.arange(len(df))
fo = stats.linregress(x, other); fe = stats.linregress(x, ese)
print(f"n = {len(df)} films; tokens E&SE {int(ese.sum()):,} / all other {int(other.sum()):,}; "
      f"mean per film {ese.mean():.0f} / {other.mean():.0f}; mean over the {int((ese>0).sum())} films with E&SE dialogue {ese[ese>0].mean():.0f}")
print(f"all other: {fo.slope:+.1f} tokens/film, r2 = {fo.rvalue**2:.2f}; E&SE Asian: {fe.slope:+.1f} tokens/film, r2 = {fe.rvalue**2:.2f} (p = {fe.pvalue:.2f})")

BLUE, RED = "#2f7fc1", "#fa7f6f"
fig, ax1 = plt.subplots(figsize=(13, 8))
ax1.plot(x, other, color=BLUE, linestyle=":", linewidth=2.5, label="All other actors")
ax1.plot(x, fo.intercept + fo.slope * x, color=BLUE, linestyle="--",
         label=f"All other actors, linear fit: {fo.slope:+.1f} tokens per film, $r^2$ = {fo.rvalue**2:.2f}")
ax1.set_ylabel("All other actors: word tokens", color=BLUE); ax1.tick_params(axis="y", labelcolor=BLUE); ax1.set_ylim(0, other.max() * 1.2)
ax2 = ax1.twinx()
ax2.plot(x, ese, color=RED, label="E&SE Asian actors")
ax2.plot(x, fe.intercept + fe.slope * x, color=RED, linestyle="--",
         label=f"E&SE Asian actors, linear fit: {fe.slope:+.1f} tokens per film, $r^2$ = {fe.rvalue**2:.2f}")
ax2.set_ylabel("E&SE Asian actors: word tokens", color=RED); ax2.tick_params(axis="y", labelcolor=RED); ax2.set_ylim(0, ese.max() * 1.5)
ax1.set_xticks(x); ax1.set_xticklabels(df["Name"], rotation=60, ha="right", fontsize=9)
ax1.set_xlabel("Film (ordered by release date)", labelpad=10)
ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
plt.tight_layout(); plt.savefig("Figure 2.png", dpi=300, bbox_inches="tight"); print("saved Figure 2.png")

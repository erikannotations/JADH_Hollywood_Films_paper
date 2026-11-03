"""Figure 5: mean sentiment score of sentences in E&SE Asian dialogue and all other dialogue, by model.
Reads the sentence-level score CSVs in this folder. Run from this folder: python3 figure5_sentiment_means.py"""
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np
d = "./"
e = pd.read_csv(d + "E&SE_Asian_sentiment_scores.csv"); o = pd.read_csv(d + "All_other_sentiment_scores.csv")
df = pd.concat([e, o], ignore_index=True)
labels = {"All_Other": "All other dialogue", "E&SE_Asian": "E&SE Asian dialogue"}
sns.set_style("whitegrid")
for variant, ci in (("", False),):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for ax, model, title in zip(axes, ("siebert_sentiment", "cardiff_sentiment"), ("Siebert model", "Cardiff model")):
        g = df.groupby("source")[model]
        means = g.mean(); sem = g.sem()
        x = np.arange(len(means))
        bars = ax.bar(x, means.values, color=["#2E86AB", "#A23B72"], alpha=0.8, edgecolor="black",
                      yerr=(1.96 * sem.values) if ci else None, capsize=6 if ci else 0)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels([labels[s] for s in means.index])
        ax.set_ylabel("Mean sentiment score", fontsize=11, fontweight="bold")
        ax.set_title("Mean sentiment score: " + title, fontsize=12, fontweight="bold")
        ax.set_ylim(-0.09, 0.01); ax.set_yticks(np.arange(-0.08, 0.011, 0.02))
        ax.grid(axis="y", alpha=0.3)
        for b, m, s in zip(bars, means.values, sem.values):
            ax.text(b.get_x() + b.get_width() / 2, m - (1.96 * s if ci else 0) - 0.003, f"{m:.3f}",
                    ha="center", va="top", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig("Figure 5.png", dpi=300, bbox_inches="tight")
    print("saved", variant or "plain")

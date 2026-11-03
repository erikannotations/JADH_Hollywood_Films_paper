"""Figure 6: proportion of positive, neutral and negative sentences in E&SE Asian dialogue and all other
dialogue, by model (neutral = Cardiff score between -0.25 and 0.25). Reads the sentence-level score CSVs
in this folder. Run from this folder: python3 figure6_sentiment_categories.py"""
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
d = "./"
df = pd.concat([pd.read_csv(d + "E&SE_Asian_sentiment_scores.csv"),
                pd.read_csv(d + "All_other_sentiment_scores.csv")], ignore_index=True)
labels = {"All_Other": "All other dialogue", "E&SE_Asian": "E&SE Asian dialogue"}
cats = ["Negative", "Neutral", "Positive"]
def classify(x): return "Positive" if x > 0.25 else ("Negative" if x < -0.25 else "Neutral")
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 4.4))
for ax, model, title in zip(axes, ("siebert_sentiment", "cardiff_sentiment"), ("Siebert model", "Cardiff model")):
    p = pd.crosstab(df["source"], df[model].apply(classify), normalize="index").reindex(columns=cats, fill_value=0) * 100
    p.index = [labels[s] for s in p.index]
    p.plot(kind="bar", stacked=True, ax=ax, color=["#E63946", "#F1FAEE", "#06D6A0"], edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Proportion (%)", fontsize=11, fontweight="bold"); ax.set_xlabel("")
    ax.set_title(f"Sentiment distribution: {title}\n(neutral: −0.25 to +0.25)", fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0); ax.set_ylim(0, 100)
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    for c in ax.containers:
        ax.bar_label(c, labels=[f"{v:.1f}%" if v > 0 else "" for v in c.datavalues], label_type="center", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig("Figure 6.png", dpi=300, bbox_inches="tight"); print("saved")

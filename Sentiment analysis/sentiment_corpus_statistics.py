"""Corpus-level sentiment numbers quoted in Section 5.2: sentence counts, mean score per collection and
model, share of sentences per category (neutral = between -0.25 and 0.25), 95% confidence interval of
each mean, and a Mann-Whitney U test between the two collections.
Run from this folder: python3 sentiment_corpus_statistics.py"""
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
e = pd.read_csv("E&SE_Asian_sentiment_scores.csv"); o = pd.read_csv("All_other_sentiment_scores.csv")
print(f"sentences: E&SE Asian {len(e):,}, all other {len(o):,}")
cat = lambda x: "negative" if x < -0.25 else ("positive" if x > 0.25 else "neutral")
for m in ["siebert_sentiment", "cardiff_sentiment"]:
    print("==", m)
    for name, df in [("E&SE Asian", e), ("All other", o)]:
        s = df[m]; ci = 1.96 * s.std(ddof=1) / np.sqrt(len(s))
        shares = s.apply(cat).value_counts(normalize=True).mul(100).round(1).to_dict()
        print(f"  {name:11s} mean {s.mean():+.3f} (95% CI +-{ci:.3f})  shares % {shares}")
    print(f"  difference {e[m].mean() - o[m].mean():+.3f}; Mann-Whitney p = {mannwhitneyu(e[m], o[m]).pvalue:.3f}")

"""Agreement between the blind manual sentiment annotation and the two classifiers (Section 5.2, Table 4).

Input:  manual_evaluation_sample.csv: 100 sentences (50 per collection) drawn from the two
        *_sentiment_scores.csv files with a fixed seed and presented to the annotator in shuffled
        order without the collection or the model scores; columns id, sentence, collection,
        annotator_score (-1..+1), siebert_score, cardiff_score.
Output: printed agreement tables (the numbers in Section 5.2 and Table 4).

Classes follow the paper's rule: < -0.25 negative, -0.25..0.25 neutral, > 0.25 positive.
Siebert is binary (sign of score).

Run:  python3 sentiment_manual_eval_analysis.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

df = pd.read_csv("manual_evaluation_sample.csv").rename(columns={"collection": "corpus", "annotator_score": "manual_label"})
df = df.dropna(subset=["manual_label"])
print(f"blind sample: {len(df)} labelled rows")


def three_class(x, lo=-0.25, hi=0.25):
    return "neg" if x < lo else ("pos" if x > hi else "neu")


def agreement(a, b):
    """Return (observed agreement, Cohen's kappa) for two equal-length label lists."""
    a, b = list(a), list(b)
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return round(po, 2), round((po - pe) / (1 - pe), 2)


df["manual"] = df.manual_label.apply(three_class)
df["cardiff"] = df.cardiff_score.apply(three_class)
df["siebert"] = np.where(df.siebert_score > 0, "pos", "neg")

print("Manual class by collection:\n", pd.crosstab(df.corpus, df.manual), "\n")
short = df.sentence.str.split().str.len() <= 3
print("Sentences <= 3 words:", short.sum())

for model in ["cardiff", "siebert"]:
    print(f"\n== {model} ==")
    for name, sub in [("all", df)] + list(df.groupby("corpus")):
        po, k = agreement(sub.manual, sub[model])
        print(f"  {name:12s} n={len(sub):3d} agreement={po:.2f} kappa={k:.2f}")
    polar = df[df.manual != "neu"]
    po, k = agreement(polar.manual, polar[model])
    print(f"  polar-only   n={len(polar):3d} agreement={po:.2f} kappa={k:.2f}")
    print(pd.crosstab(df.manual, df[model], rownames=["manual"], colnames=[model]))

for name, sub in [("<=3 words", df[short]), (">3 words", df[~short])]:
    print(f"  {name}: cardiff {agreement(sub.manual, sub.cardiff)} siebert {agreement(sub.manual, sub.siebert)} n={len(sub)}")
both = df[(df.manual != "neu") & (df.cardiff != "neu")]
print("\nCardiff polarity agreement when both commit:", (both.manual == both.cardiff).sum(), "/", len(both))
print("Spearman manual~cardiff: %.2f" % spearmanr(df.manual_label, df.cardiff_score)[0])
print("Spearman manual~siebert: %.2f" % spearmanr(df.manual_label, df.siebert_score)[0])

# Sensitivity: count mild labels (+-0.3) as neutral, i.e. polar only if |label| >= 0.5.
df["manual_lenient"] = df.manual_label.apply(lambda x: three_class(x, -0.4, 0.4))
pol = df.manual_lenient != "neu"
print("\nSensitivity (mild +-0.3 as neutral): neutral n=%d" % (~pol).sum(),
      "cardiff", agreement(df.manual_lenient, df.cardiff), "siebert", agreement(df.manual_lenient, df.siebert),
      "| polar-only siebert", agreement(df.manual_lenient[pol], df.siebert[pol]))

# Bootstrap CIs.
rng = np.random.RandomState(0)
m, c, s = df.manual.values, df.cardiff.values, df.siebert.values
acc_c, acc_s = [], []
for _ in range(5000):
    idx = rng.randint(0, len(df), len(df))
    acc_c.append(np.mean(m[idx] == c[idx]))
    pol = idx[m[idx] != "neu"]
    acc_s.append(np.mean(m[pol] == s[pol]))
print("95% CI Cardiff agreement:", np.percentile(acc_c, [2.5, 97.5]).round(2))
print("95% CI Siebert polar-only agreement:", np.percentile(acc_s, [2.5, 97.5]).round(2))

# Between-collection difference in the manual labels: reported only as a caveat
# (single annotator, n = 50 per side).
e = df[df.corpus == "E&SE_Asian"].manual_label
o = df[df.corpus == "All_Other"].manual_label
print("\nManual mean E&SE %.3f vs all other %.3f, Mann-Whitney p=%.3f" % (e.mean(), o.mean(), mannwhitneyu(e, o).pvalue))

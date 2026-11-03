"""Table 2 (v3): keyness of unigrams, E&SE Asian dialogue vs all other dialogue,
with a per-film dispersion column.

Inputs:
  ../dialogue/E&SE_Asian_dialogue.txt   E&SE Asian dialogue (4,692 whitespace tokens, matches
                                        DATA_Token.csv exactly)
  ../dialogue/All_other_dialogue.txt    all other dialogue
  ../Token/DATA_Token.csv               per-film token totals, used to split the dialogue files into
                                        films (E&SE boundaries fall exactly on line ends)
The two dialogue files are not distributed (see README); the outputs are:
  table2_keyness.csv            Table 2 of the paper (24 rows)
  keyness_all_significant.csv   every word with log-likelihood > 3.84 (471 rows), before the
                                frequency and dispersion filters
Run from this folder: python3 table2_keyness.py

Tokenizer: the paper's single tokenizer (whitespace split, punctuation stripped, lower-cased).
Log-likelihood as in the original notebook (Dunning 1993). Filter: LL > 3.84, frequency >= 5 in
both collections, occurring in the E&SE Asian dialogue of at least three of the 19 films. Ranked by the
ratio of relative frequencies (Brezina 2018, Section 3.4). Dispersion: number of the 19 films in whose E&SE dialogue the word occurs (range; as a share, Egbert &
Biber's TD, Sönning 2023).
"""
import math, os, re
from collections import Counter, defaultdict
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DIALOGUE = os.path.join(ROOT, "dialogue")

def tokenize(text):
    """Paper-wide tokenizer: whitespace split (as in Section 4), surrounding punctuation stripped, lower-cased; contractions kept whole."""
    return [t for t in (x.strip("\"'“”‘’.,!?;:()[]-…—–*").lower() for x in text.split()) if t]

def log_likelihood(a, b, c, d):
    E1 = c * (a + b) / (c + d)
    E2 = d * (a + b) / (c + d)
    a = a if a else 1e-6
    b = b if b else 1e-6
    return 2 * (a * math.log(a / E1) + b * math.log(b / E2))

tok = pd.read_csv(os.path.join(ROOT, "Token/DATA_Token.csv"), encoding="utf-8-sig")
tok = tok.rename(columns={"E&SE Asian": "Asian", "All Other": "Non-Asian"})
names = tok.Name.tolist()

raw = open(os.path.join(DIALOGUE, "E&SE_Asian_dialogue.txt"), encoding="utf-8", errors="replace").read()
lines = [l.strip() for l in re.split(r"\r\n|\r|\n", raw) if l.strip()]
films = [(n, a) for n, a in zip(names, tok.Asian) if a > 0]

# split the E&SE file into films by cumulative whitespace token count
per_film_lines = defaultdict(list)
cum, fi, target = 0, 0, films[0][1]
for l in lines:
    per_film_lines[films[fi][0]].append(l)
    cum += len(l.split())
    if cum >= target and fi < len(films) - 1:
        assert cum == target, f"boundary miss at {films[fi][0]}"
        fi += 1
        target += films[fi][1]
assert len(per_film_lines) == 19

E_film = {f: Counter(tokenize(" ".join(v))) for f, v in per_film_lines.items()}
E = Counter()
for c in E_film.values():
    E.update(c)
o_lines = [l for l in open(os.path.join(DIALOGUE, "All_other_dialogue.txt"), encoding="utf-8", errors="replace").read().splitlines() if l.strip()]
O = Counter(tokenize(" ".join(o_lines)))
# split the all-other file into films too (order: the 19 E&SE films in release order, then the 16 others; see section51_perfilm_checks.py)
other_sizes = dict(zip(names, tok["Non-Asian"]))
o_order = [f for f, _ in films] + [n for n, a in zip(names, tok.Asian) if a == 0]
O_film, cum, fi, target = defaultdict(list), 0, 0, other_sizes[o_order[0]]
for l in o_lines:
    O_film[o_order[fi]].append(l); cum += len(l.split())
    if cum >= target and fi < len(o_order) - 1:
        fi += 1; target += other_sizes[o_order[fi]]
O_film = {f: Counter(tokenize(" ".join(O_film[f]))) for f, _ in films}
def films_dir(w):
    """Over the 19 films: number in which the word's rate is higher in E&SE Asian dialogue than in the rest of the film, and number lower (films where the word does not occur at all are not counted)."""
    hi = lo = 0
    for f, _ in films:
        a, b = E_film[f][w], O_film[f][w]
        if a + b == 0: continue
        ra, rb = a / sum(E_film[f].values()), b / sum(O_film[f].values())
        hi += ra > rb; lo += ra < rb
    return hi, lo
c1, c2 = sum(E.values()), sum(O.values())
print(f"tokens (paper tokenizer): E&SE {c1:,}  all other {c2:,}")

rows = []
for w in set(E) | set(O):
    a, b = E[w], O[w]
    rows.append((w, a, b, log_likelihood(a, b, c1, c2), "E&SE" if a / c1 > b / c2 else "All other"))
df = pd.DataFrame(rows, columns=["word", "f_ese", "f_other", "LL", "key_in"]).sort_values("LL", ascending=False)
print("significant at LL > 3.84:", int((df.LL > 3.84).sum()))
sig = df[df.LL > 3.84].copy(); sig["LL"] = sig.LL.round(2)
sig.to_csv(os.path.join(HERE, "keyness_all_significant.csv"), index=False)

t2 = df[(df.LL > 3.84) & (df.f_ese >= 5) & (df.f_other >= 5)].copy()
t2["freq_ratio"] = [round((a / c1) / (b / c2), 2) for a, b in zip(t2.f_ese, t2.f_other)]  # ratio of relative frequencies (Brezina 2018, §3.4; Kilgarriff 2009), no constant needed since f >= 5 in both
t2["log_ratio"] = [round(math.log2(r), 2) for r in t2.freq_ratio]  # kept in the CSV for reference
t2["films"] = [sum(1 for f in E_film if E_film[f][w] > 0) for w in t2.word]
MIN_FILMS = 3  # keep words that occur in the E&SE Asian dialogue of at least three of the 19 films
t2 = t2[t2.films >= MIN_FILMS]
t2 = t2.sort_values("freq_ratio", ascending=False)  # LL filters, frequency ratio ranks (Brezina 2018, §3.4)
t2["top_film"] = [re.sub(r"\s*\(\d{4}\)", "", max(E_film, key=lambda f: E_film[f][w])) for w in t2.word]
t2["top_film_tokens"] = [max(E_film[f][w] for f in E_film) for w in t2.word]
t2["LL"] = t2.LL.round(1)
t2.to_csv(os.path.join(HERE, "table2_keyness.csv"), index=False)
print(f"Table 2 rows: {len(t2)}")
print(t2.to_string(index=False))

# sensitivity checks quoted in the text
mantra = re.compile(r"one with the force|force is with me|force of others", re.I)
all_lines = [l for v in per_film_lines.values() for l in v]
no_mantra = [l for l in all_lines if not mantra.search(l)]
N = Counter(tokenize(" ".join(no_mantra)))
cn = sum(N.values())
print(f"\nmantra lines: {len(all_lines) - len(no_mantra)}, tokens: {c1 - cn}")
for w in ["force", "with", "one", "others", "the", "is"]:
    print(f"  {w:7} LL full {log_likelihood(E[w], O[w], c1, c2):6.1f}   without mantra {log_likelihood(N[w], O[w], cn, c2):5.1f}")

three = ["Star Wars 8 - The Last Jedi (2017)", "Spider-Man: Far From Home (2019)", "Spider-Man: No Way Home (2021)"]
we3 = sum(E_film[f]["we"] for f in three)
rest = Counter()
for f in E_film:
    if f not in three:
        rest.update(E_film[f])
cr = sum(rest.values())
print(f"\n'we': {we3} of {E['we']} tokens in the three Rose Tico / Ned Leeds films; "
      f"rest {1000*rest['we']/cr:.1f} vs all other {1000*O['we']/c2:.1f} per 1,000, "
      f"LL {log_likelihood(rest['we'], O['we'], cr, c2):.1f}")

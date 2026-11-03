# Data and code for "The Temporality of East and Southeast Asian Actor Dialogue in 21st-Century Hollywood Films"

Every figure and table in the manuscript can be
regenerated from the files here with the script in the same folder; each script prints the numbers quoted
in the corresponding section. Requirements: Python 3 with pandas, numpy, scipy, matplotlib and seaborn
(the sentiment notebook additionally needs transformers and nltk).

## Contents

| Folder / file | Manuscript | What it holds |
|---|---|---|
| `Film list (Appendix).csv` | Appendix | The 35 films in release order with year, genre, seconds of E&SE Asian dialogue and the E&SE Asian characters (actors) |
| `Dialogue Length/` | Section 4.1, Figure 1 | `DATA_Total Time of Dialogues.csv`: per film, release date and total time (h:mm:ss) of dialogue by all other actors and by E&SE Asian actors. `figure1_dialogue_time.py` draws Figure 1 and prints the descriptive linear fits |
| `Token/` | Section 4.2, Figure 2 | `DATA_Token.csv`: per film, whitespace word-token counts (Python `str.split()`) for the two collections. `figure2_tokens.py` draws Figure 2 and prints the fits and per-film means |
| `Scatter Plot/` | Section 4.3, Figure 3 | `DATA_Distribution.csv`: the 844 E&SE Asian dialogue segments (subtitle lines) in the 19 films with such dialogue, each as a position (%) within its film's running time; film numbers follow the release order of the 35 films. `figure3_scatter.py` draws Figure 3 |
| `Deciles/` | Section 4.3, Figure 4, footnote 7 | `figure4_deciles.py` bins the positions into tenths, weights every film equally and draws Figure 4; it writes `deciles_per_film.csv` (segments per film and tenth) and `decile_summary.csv` (segments, films present and mean per-film share per tenth) |
| `Keyness/` | Section 5.1, Table 2 | `table2_keyness.py` computes log-likelihood keyness for every word type, filters (LL > 3.84, at least 5 occurrences in both collections, present in the E&SE Asian dialogue of at least 3 films) and ranks by the ratio of relative frequencies; outputs `table2_keyness.csv` (Table 2) and `keyness_all_significant.csv` (all 471 words with LL > 3.84); the script also prints the share of *we* tokens spoken by Rose Tico and Ned Leeds quoted in the text |
| `Sentiment analysis/` | Section 5.2, Figures 5–6, Tables 3–4 | `E&SE_Asian_sentiment_scores.csv` and `All_other_sentiment_scores.csv`: one row per sentence (827 and 56,297) with its Siebert and Cardiff scores and word count, produced by `Two sentiment models method.ipynb`; the sentence text itself is not included (see below). `sentiment_corpus_statistics.py` prints the means, category shares, confidence intervals and Mann–Whitney tests. `figure5_sentiment_means.py` and `figure6_sentiment_categories.py` draw the two figures. `manual_evaluation_sample.csv` is the blind manual annotation of 100 sentences (50 per collection; the annotator saw the sentences in shuffled order without collection or model scores), the only file here that quotes dialogue, and `sentiment_manual_eval_analysis.py` computes the agreement figures in Table 4 |

## What is not included

The subtitle files and the two plain-text dialogue collections (`E&SE_Asian_dialogue.txt`, 4,692 tokens, and
`All_other_dialogue.txt`) are not distributed, for copyright reasons, and the sentence-level sentiment files
carry scores and word counts only, not the sentences. The keyness script reads
the dialogue collections from a `dialogue/` folder at the root of the repository, and the sentiment notebook
from its own folder. Their outputs are included so that Table 2, the sentiment figures and the statistics quoted from them can be checked without the texts.

## Notes on the data

- **Tokenizer.** All token counts use whitespace splitting. The keyness scripts additionally strip surrounding
  punctuation and lower-case, keeping contractions whole; on the E&SE Asian file this gives exactly the 4,692
  tokens of `DATA_Token.csv`. The all-other dialogue file yields 305,029 tokens against the 305,692 of
  `DATA_Token.csv` (a slightly different export of the same corpus); the difference is immaterial for the
  relative frequencies.
- **Order of the all-other dialogue file.** It holds the 19 films with E&SE Asian dialogue first, in release
  order, then the 16 others. The E&SE Asian file is in release order. Both are split into films by the
  per-film token totals of `DATA_Token.csv`; the E&SE boundaries fall exactly on line ends.
- **Sentiment score files.** The notebook writes each sentence with its two scores; the files distributed
  here are those outputs with the sentence text removed and a sentence ID and word count added. Running the
  notebook overwrites them with versions that contain the sentences.
- **Sentiment analysis file version.** The sentiment scores and the manual evaluation were computed on an
  earlier export of the E&SE Asian dialogue that lacks 185 of the 4,692 tokens, all from *Rogue One* and
  nearly all repetitions of Chirrut Îmwe's "I'm one with the Force, the Force is with me"; the 827 sentences
  are otherwise the same text (manuscript, Section 5.2, footnote).

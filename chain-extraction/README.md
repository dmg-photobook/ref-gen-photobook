# Reference Chain Extraction
This directory contains code for the extraction of referring expressions from [PhotoBook](https://dmg-photobook.github.io/index.html) visual dialogues, using [BERTScore](https://arxiv.org/abs/1904.09675) with respect to [MSCOCO](http://cocodataset.org/#home) captions and [METEOR](http://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf) score with respect to [Visual Genome](https://visualgenome.org) scene graphs.

#### Usage

Run python scripts from this directory: `chain-extraction/`.

Creating the reference chain dataset.
- `python src/extract_segments.py` to obtain segments of scored utterances from all game logs.
- `python src/make_chains.py` to filter out irrelevant utterances and obtain reference chains.
- `python src/make_gold_chains.py` to obtain reference chains from the annotated logs.
- `python src/make_dataset.py` to obtain the dataset splits and to compute dataset statistics.

Evaluating the reference chain extraction procedure.
- Run `python src/extract_segments.py` using `--path_game_logs data/logs/test_logs.dict`.
- Run `python src/eval_chains.py` on the test segments output by the previous step.

Reproducing the whole extraction and evaluation procedure described in the paper.
- `python src/extract_segments.py out/all_segments.dict --stopwords --meteor --from_first_common --utterances_as_captions`
- `python src/make_chains.py out/all_segments.dict out/all_chains.json --score f1`
- `python src/make_gold_chains.py out/gold_chains.json --from_first_common --first_reference_only`
- `python src/make_dataset.py out/all_chains.json out/gold_chains.json out/dataset`

- `python src/extract_segments.py out/eval_segments.dict --path_game_logs data/logs/test_logs.dict --stopwords --meteor --from_first_common --utterances_as_captions`
- `python src/eval_chains.py out/eval_segments.dict`
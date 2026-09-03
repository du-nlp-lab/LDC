# LDC: Learning to Generate Research Ideas with Dynamic Control

Data collection and processing pipeline for **[LDC: Learning to Generate Research Ideas with Dynamic Control](https://github.com/du-nlp-lab/LDC)** (EMNLP 2026, Main Conference).

LDC is a two-stage framework (SFT + controllable RL) for scientific research idea generation. It trains three reward models for **novelty**, **feasibility**, and **effectiveness** on fine-grained feedback derived from real review data, and steers generation at inference time with dimensional controllers coordinated by a sentence-level decoder.

This repository currently contains the scripts used to build our dataset of 6,765 ICLR/NeurIPS (2023–2024) papers with per-dimension scores, from which the SFT / RL / evaluation splits in the paper are derived.

## Pipeline Overview

```
OpenReview ──► metadata + reviews ──► PDF section parsing ──► idea extraction
                                                                   │
                          novelty / feasibility / effectiveness scoring
                                                                   │
                                     cleaned JSON dataset (SFT / RL / eval splits)
```

| Stage | Script | What it does |
|---|---|---|
| 1. Crawl papers + reviews | `getReview.py` | Fetches accepted **and rejected** submissions with official reviews and meta-reviews from OpenReview (API v1, e.g. ICLR 2023), and optionally downloads PDFs and extracts their full text (`--download_pdfs`). |
| 1. Crawl metadata | `getData_meta.py` | Fetches accepted papers and meta-reviews from OpenReview (ICLR 2023 API v1). |
| 1. Crawl reviews | `getData2_iclr2024.py` | Fetches all replies (official reviews, meta-reviews, decisions) for ICLR 2024 submissions (API v2). |
| 2. Link records | `fillfid.py` | Fills in OpenReview `forum_id`s by fuzzy title matching (`fuzzywuzzy`). |
| 3. Parse papers | `fillsection.py` | Downloads paper PDFs and extracts *Abstract / Methodology / Experiment* sections with PyMuPDF, using a synonym list for section-title matching. |
| 4. Extract ideas | `extract_idea.py` | Prompts the LLM with the paper's title, abstract, and entities to extract the golden research idea (Method + Experiment Plan) used as the SFT target (prompt in Appendix M of the paper). |
| 4. Extract results | `extract_exp.py` | Extracts the experiment section from the paper PDF, then the main metric, baseline performance, and the proposed method's performance, which feed the effectiveness labels. |
| 4. Select supporting paper | `select_supporting.py` | Counts in-text author-year citations across the parsed sections, then prompts the LLM with the abstract/introduction and the most-cited candidates (with citation counts) to pick the single most significant supporting paper used as the SFT input; also fills `recent_works` (top-3 cited works since 2023) consumed by novelty scoring. |
| 5. Score novelty | `novelty_v3.py` | Scores novelty (1–10) with a local HuggingFace LLaMA model, conditioning on the abstract, the top-3 recently cited related works, and the review comments. (`novelty_v2.py` and `getNovelty.py` are earlier variants kept for reference.) |
| 5. Score feasibility + effectiveness | `eval.py` | Scores feasibility and effectiveness (1–10) from the method/experiment sections and review comments (rubrics in Appendix G). |
| 5. Score feasibility (legacy) | `feasibility.py` | Earlier standalone feasibility scorer, superseded by `eval.py`; kept for reference. |
| 6. Repair & clean | `reeval.py`, `reparse.py` | Re-runs examples whose generations failed JSON parsing (with retries) and extracts clean JSON from raw model outputs. |

Scoring rubrics for all three dimensions follow the definitions in Appendix G of the paper; the exact prompts are embedded in each script and listed in Appendices M–P.

## Setup

```bash
pip install -r requirements.txt
```

Before running, fill in the placeholders at the top of the scripts:

- `access_token` / `model_id` — HuggingFace token and model path for local scoring (we use `meta-llama/Meta-Llama-3-70B-Instruct`).
- `api_key` / `base_url` — API credentials in `extract_exp.py`, `extract_idea.py`, `select_supporting.py`, and `getNovelty.py` for the OpenAI-compatible endpoint.
- OpenReview credentials for the crawling scripts, read from environment variables:
  ```bash
  export OPENREVIEW_USERNAME=your_email
  export OPENREVIEW_PASSWORD=your_password
  ```

Most scripts share the same CLI:

```bash
python getReview.py    --conference ICLR --year 2023 --download_pdfs --pdf_save_dir papers/
python fillfid.py      papers.json --output_json papers_with_forum_ids.json --conference ICLR --year 2024
python fillsection.py  papers.json --output_dir output/ --save_intermediate
python extract_idea.py papers.json --output_dir output/
python extract_exp.py  papers.json --output_dir output/
python select_supporting.py papers.json --output_dir output/ --top_k 10
python novelty_v3.py   papers.json --output_dir output/
python feasibility.py  papers.json --output_dir output/
python eval.py         papers.json --output_dir output/
python reeval.py       papers.json --output_dir output/ --max_retries 3
python reparse.py      papers.json --output_json papers_reparsed.json
```

Each script reads a JSON list of paper records, adds its fields (extracted sections, ideas, or scores with justifications), and writes the updated JSON to `--output_dir`. Runs are resumable: papers already present in the output are skipped.

## Data Notice

All source texts are publicly available scientific papers and their official OpenReview reviews, retrieved via the OpenReview, Semantic Scholar, and arXiv APIs. Please respect the corresponding terms of use when re-crawling. All collected scores are normalized to [0, 1] for reward model training (see §3.3 of the paper).

## Citation

```bibtex
@inproceedings{li2026ldc,
  title     = {{LDC}: Learning to Generate Research Ideas with Dynamic Control},
  author    = {Li, Ruochen and Jing, Liqiang and Han, Chi and Zhou, Jiawei and Du, Xinya},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing},
  month     = oct,
  year      = {2026},
  address   = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics}
}
```

## License

Released under the [MIT License](LICENSE).

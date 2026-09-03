"""Select the most significant supporting paper for each target paper.

For every paper record, this script
  1. extracts in-text citations (author-year style) from the parsed sections
     and counts how often each cited work appears within the paper;
  2. prompts an LLM with the paper's abstract and introduction, together with
     the most-cited candidate works and their in-text citation counts, to pick
     the single most significant supporting paper (used as the SFT input);
  3. stores the top-3 cited works since 2023 in `recent_works`, which the
     novelty scoring script (novelty_v3.py) consumes.

Input records are expected in the format produced by fillsection.py
(`content.introduction`, `content.related_work`, ... plus `title`/`summary`).

Usage:
    python select_supporting.py papers.json --output_dir output/
"""

import os
import re
import json
import time
import logging
import argparse
from collections import Counter

from tqdm import tqdm
from openai import OpenAI
import openai

client = OpenAI(
    api_key="",  # fill in your API key
    base_url="https://api.llama-api.com"
)
MODEL_NAME = ""  # e.g. "llama3-70b"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def log_error(error_message, error_log_path):
    with open(error_log_path, 'a') as f:
        f.write(error_message + '\n')


# ---------------------------------------------------------------------------
# 1. In-text citation extraction and counting
# ---------------------------------------------------------------------------

# "Smith et al., 2023" / "Smith & Jones, 2023" / "Smith and Jones, 2023" / "Smith, 2023"
_CITE_CORE = re.compile(
    r"([A-Z][A-Za-z\-'À-ſ]+)"
    r"(\s+et\s+al\.?|\s*&\s*[A-Z][A-Za-z\-'À-ſ]+|\s+and\s+[A-Z][A-Za-z\-'À-ſ]+)?"
    r",?\s*\(?((?:19|20)\d{2})[a-z]?\)?"
)
# narrative citations: "Smith et al. (2023)"
_CITE_NARRATIVE = re.compile(
    r"([A-Z][A-Za-z\-'À-ſ]+)"
    r"(\s+et\s+al\.?|\s*&\s*[A-Z][A-Za-z\-'À-ſ]+|\s+and\s+[A-Z][A-Za-z\-'À-ſ]+)?"
    r"\s+\(((?:19|20)\d{2})[a-z]?\)"
)


def _normalize(first_author, year):
    return f"{first_author} et al., {year}"


def count_citations(text):
    """Count in-text author-year citations. Returns Counter keyed by 'Author et al., YYYY'."""
    counts = Counter()
    # citations inside parentheses, possibly "(A et al., 2020; B et al., 2021)"
    for paren in re.findall(r"\(([^()]{4,400})\)", text):
        for chunk in paren.split(';'):
            m = _CITE_CORE.search(chunk)
            if m:
                counts[_normalize(m.group(1), m.group(3))] += 1
    # narrative citations
    for m in _CITE_NARRATIVE.finditer(text):
        counts[_normalize(m.group(1), m.group(3))] += 1
    return counts


def collect_paper_citations(paper):
    """Aggregate citation counts over all parsed sections of the paper."""
    content = paper.get('content', {}) or {}
    text = "\n".join(
        content.get(k) or ''
        for k in ('introduction', 'related_work', 'method', 'experiment')
    )
    return count_citations(text)


# ---------------------------------------------------------------------------
# 2. LLM-based supporting-paper selection
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an AI assistant that analyzes scientific papers. Given a paper's "
    "abstract and introduction, and a list of works it cites (with how many "
    "times each work is cited within the paper), you identify the single most "
    "significant supporting work: the prior work that the paper most directly "
    "builds upon or is most inspired by."
)

USER_TEMPLATE = """Target paper title: {title}

Abstract: <> {abstract} </>

Introduction (excerpt): <> {introduction} </>

Cited works and their in-text citation counts within this paper:
{candidates}

Choose the ONE most significant supporting work from the list above: the prior work this paper most directly builds upon. Consider both how often a work is cited and how it is discussed in the introduction. Reply in JSON only:
{{"supporting_paper": "<entry exactly as listed above>", "reason": "<2-3 sentences>"}}
"""


def extract_json_from_string(text):
    try:
        m = re.search(r'{.*}', text, re.DOTALL)
        return json.loads(m.group(0)) if m else None
    except (json.JSONDecodeError, AttributeError):
        return None


def select_supporting_with_llama(paper, candidates, error_log_path, max_retries=3):
    title = paper.get('title', 'unknown_title')
    abstract = paper.get('summary') or paper.get('abstract') or 'N/A'
    introduction = (paper.get('content', {}) or {}).get('introduction') or 'N/A'
    candidate_lines = "\n".join(
        f"- {name} (cited {cnt} times)" for name, cnt in candidates
    )
    prompt = USER_TEMPLATE.format(
        title=title,
        abstract=abstract[:4000],
        introduction=introduction[:6000],
        candidates=candidate_lines,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt},
                ],
                temperature=0.0,
            )
            parsed = extract_json_from_string(response.choices[0].message.content)
            if parsed and parsed.get('supporting_paper'):
                return parsed
            logging.warning(f"Unparsable selection for {title}, retry {attempt + 1}")
        except openai.OpenAIError as e:
            logging.warning(f"API error for {title}: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    log_error(f"Failed to select supporting paper for {title}", error_log_path)
    return None


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def process_papers(papers, output_path, error_log_path, top_k, recent_year):
    updated = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            updated = json.load(f)
    done_titles = {p.get('title') for p in updated}

    for paper in tqdm(papers, desc="Selecting supporting papers"):
        title = paper.get('title', 'unknown_title')
        if title in done_titles:
            continue

        counts = collect_paper_citations(paper)
        if not counts:
            log_error(f"No in-text citations found for {title}", error_log_path)
            updated.append(paper)
            continue

        candidates = counts.most_common(top_k)
        paper['citation_counts'] = dict(candidates)

        # top-3 cited works since `recent_year`, used by novelty scoring
        recent = [
            (name, cnt) for name, cnt in counts.most_common()
            if int(re.search(r'(19|20)\d{2}', name).group(0)) >= recent_year
        ][:3]
        paper['recent_works'] = "; ".join(name for name, _ in recent) if recent else 'N/A'

        selection = select_supporting_with_llama(paper, candidates, error_log_path)
        if selection:
            paper['supporting_paper'] = selection

        updated.append(paper)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(updated, f, ensure_ascii=False, indent=4)

    return updated


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select the most significant supporting paper for each record.")
    parser.add_argument('input_json', type=str, help="Path to the input JSON file containing paper data")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory to save output JSON and logs")
    parser.add_argument('--top_k', type=int, default=10, help="Number of most-cited candidates given to the LLM")
    parser.add_argument('--recent_year', type=int, default=2023, help="Cutoff year for recent_works")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'papers_with_supporting.json')
    error_log_path = os.path.join(args.output_dir, 'select_supporting_errors.log')

    with open(args.input_json, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    result = process_papers(papers, output_path, error_log_path, args.top_k, args.recent_year)
    logging.info(f"Done. {len(result)} papers written to {output_path}")

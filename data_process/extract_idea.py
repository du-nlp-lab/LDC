"""Extract the golden research idea (Method + Experiment Plan) from each paper.

Prompts the LLM with the paper's title, abstract, and entities to produce the
structured research idea used as the golden output for supervised fine-tuning.

Usage:
    python extract_idea.py papers.json --output_dir output/
"""

import os
import json
import time
import logging
import argparse

from tqdm import tqdm
from openai import OpenAI
import openai

client = OpenAI(
    api_key="",
    base_url="https://api.llama-api.com"
)
MODEL_NAME = ""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def log_error(error_message, error_log_path):
    with open(error_log_path, 'a') as f:
        f.write(error_message + '\n')


SYS_PROMPT = (
    "You are an AI assistant whose primary goal is to extract specific details from "
    "the scientific literature to aid researchers in understanding and replicating "
    "the methodologies and experiment plans of the work."
)

USER_TEMPLATE = """You are tasked with extracting the Method and Experiment Plan from an academic paper. These should include:
- Method: A concise summary of the methodological approach employed in the study.
- Experiment Plan: Key details of the experiment, including dataset preparation, baseline implementation, and evaluation metrics or procedures.
Ensure that the output is clear, focused, and formatted to align with the given structure.

I am going to provide the target paper and entities as follows:
- Target paper title: {title}
- Target paper abstract: {abstract}
- Entities: {entities}

With the provided target paper and entities, extract and summarize the Method and Experiment Plan in the following format:
- Method: [Provide a concise description of the methodology used in the study.]
- Experiment Plan: [Summarize the dataset preparation, baseline implementation, and evaluation procedures.]

Example Input:
- Target paper title: "Transformer Models for Legal Text Analysis"
- Target paper abstract: "Deep learning has transformed the field of natural language processing, yet challenges remain in domain-specific applications. This paper explores the use of transformer models for legal text analysis, addressing the question: 'Can pre-trained language models be adapted effectively for legal case prediction?' The study employs fine-tuning techniques and evaluates performance on a benchmark dataset of legal cases. Results show a significant improvement in prediction accuracy compared to traditional methods."

Expected Output:
- Method: We introduce fine-tuning techniques to adapt pre-trained transformer models for legal text analysis.
- Experiment Plan:
  - Dataset Preparation: A legal benchmark dataset of case documents is used.
  - Baseline Implementation: Models are compared against traditional NLP methods.
  - Evaluation Procedure: Performance is measured in terms of prediction accuracy on unseen legal cases.
"""


def extract_idea_with_llama(title, abstract, entities, error_log_path, max_retries=5):
    prompt = USER_TEMPLATE.format(title=title, abstract=abstract, entities=entities)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {'role': 'system', 'content': SYS_PROMPT},
                    {'role': 'user', 'content': prompt},
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content
        except openai.OpenAIError as e:
            logging.warning(f"API error for {title}: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    log_error(f"Failed to extract idea for {title}", error_log_path)
    return None


def process_papers(papers, output_path, error_log_path):
    updated = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            updated = json.load(f)
    done_titles = {p.get('title') for p in updated}

    for paper in tqdm(papers, desc="Extracting research ideas"):
        title = paper.get('title', 'unknown_title')
        if title in done_titles:
            continue

        abstract = paper.get('summary') or paper.get('abstract') or 'N/A'
        entities = paper.get('entities') or paper.get('tags') or 'N/A'

        idea = extract_idea_with_llama(title, abstract, entities, error_log_path)
        if idea:
            paper['idea'] = idea
        updated.append(paper)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(updated, f, ensure_ascii=False, indent=4)

    return updated


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract the golden research idea (Method + Experiment Plan) from each paper.")
    parser.add_argument('input_json', type=str, help="Path to the input JSON file containing paper data")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory to save output JSON and logs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'papers_with_ideas.json')
    error_log_path = os.path.join(args.output_dir, 'extract_idea_errors.log')

    with open(args.input_json, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    result = process_papers(papers, output_path, error_log_path)
    logging.info(f"Done. {len(result)} papers written to {output_path}")

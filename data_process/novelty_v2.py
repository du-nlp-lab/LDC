import os
import json
import time
import logging
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

access_token = ""
# Load model with token

model_id = ""

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_error(error_message, error_log_path):
    with open(error_log_path, 'a') as f:
        f.write(error_message + '\n')

# Score novelty with the LLM
def extract_novelty_with_llama(title, abstract, related_work, recent_works, reviews, meta_review, error_log_path):
    # Related Works (extracted from Paper section): <> {related_work} </> (optional reference for novelty) \n
    prompt = f"""
    Based on the following information about a scientific paper, please evaluate its novelty:

    Title: {title} \n
    Abstract: <> {abstract} </> \n
    Related Works (top 3 from citations since 2023): <> {recent_works} </> \n
    Review Comments: <> {reviews} </> \n
    MetaReview: <> {meta_review} </> \n

    Novelty Evaluation Instructions:
    Evaluate how creative and different the idea is compared to existing works on the topic. Consider all papers that appeared online prior to July 2024 as existing work. Your evaluation should consider the degree to which the paper brings new insights and differentiates itself from prior research.

    Please assign a novelty score on a scale from 1 to 10 based on the following criteria:

    1-2 Not novel at all: There are many existing ideas that are the same.

    3-4 Mostly not novel: You can find very similar ideas.

    5 Somewhat novel: There are differences from existing ideas but not enough to turn into a new paper.

    6-7 Reasonably novel: There are notable differences from existing ideas and probably enough to turn into a new paper.

    8-9 Clearly novel: Major differences from all existing ideas.

    10 Very novel: Very different from all existing ideas in a very interesting and clever way.

    Novelty Rationale:
    After assigning a score, please provide a short justification for your rating. If the score is below 6, specify similar works that closely resemble this paper. The rationale should be at least 2-3 sentences.
    Now provide your novelty score result and reasoning based on the above given informations in this format: score (number 1-10): reason sentence \n
</>
    """
    sys_prompt = "You are a specialized assistant for scientific text evaluation. Your task is to evaluate the novelty of scientific papers. \n"
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    input_ids = input_ids.to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def process_papers(papers, output_dir, error_log_path):
    updated_papers = []

    for paper in tqdm(papers, desc="Processing Papers"):
        title = paper.get('title', 'unknown_title')
        content = paper.get('content', {})
        abstract = content.get('abstract', 'N/A')
        related_work = content.get('related_work', 'N/A')
        recent_works = paper.get('recent_works', 'N/A')
        reviews = paper.get('reviews', [])
        meta_review = paper.get('meta_review', 'N/A')

        if not reviews:
            logging.warning(f"No reviews found for paper: {title}")
            continue

        novelty_evaluation = extract_novelty_with_llama(title, abstract, related_work, recent_works, reviews, meta_review, error_log_path)

        if novelty_evaluation:
            logging.info(f"Extracted novelty score for {title}: {novelty_evaluation}")
            paper['novelty_score'] = novelty_evaluation
        else:
            logging.warning(f"Failed to extract novelty score for {title}.")
            paper['novelty_score'] = "N/A"

        updated_papers.append(paper)

    return updated_papers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process papers and extract novelty scores.")
    parser.add_argument('input_json', type=str, help="Path to the input JSON file containing paper data")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory to save PDFs, text and JSON")

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    error_log_path = os.path.join(output_dir, 'error_log.txt')

    with open(args.input_json, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    updated_papers = process_papers(papers, output_dir, error_log_path)

    output_json = os.path.join(output_dir, "updated_novelty_scores.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(updated_papers, f, ensure_ascii=False, indent=4)

    logging.info(f"Updated JSON saved to {output_json}")
    print(f"Error log saved to: {error_log_path}")

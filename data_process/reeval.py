import os
import json
import re
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

access_token = ""
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

def extract_json_from_string(text):
    """
    Extract the substring from the first '{' to the last '}' and parse it as JSON.
    """
    try:
        json_str = re.search(r'{.*}', text, re.DOTALL)
        if json_str:
            return json.loads(json_str.group())
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from text: {text}. Error: {e}")
    return None

def regenerate_scores_for_paper(title, abstract, method, experiment, reviews, meta_review, error_log_path):
    if isinstance(reviews, list):
        reviews_text = ' '.join([
            review['content'].get('summary', '') + ' ' +
            review['content'].get('weaknesses', '')
            for review in reviews
        ])
    else:
        reviews_text = reviews

    prompt = f"""
Based on the following information about a scientific paper, please evaluate its feasibility and expected effectiveness:

Title: {title} \n
Abstract: <> {abstract} </> \n
Method: <> {method} </> \n
Experiment: <> {experiment} </> \n
Review Comments: <> {reviews_text} </> \n
MetaReview: <> {meta_review} </> \n

Please provide feasibility and effectiveness scores in the following **JSON format only**:
{{
    "feasibility": {{
        "score": a number (1-10),
        "reason": "reason sentence"
    }},
    "effectiveness": {{
        "score": a number (1-10),
        "reason": "reason sentence"
    }}
}}

**Do not output anything else except this JSON**:
"""
    return generate_response(prompt, title, error_log_path)

def generate_response(prompt, title, error_log_path):
    messages = [
        {"role": "system", "content": "You are a specialized assistant for scientific text evaluation."},
        {"role": "user", "content": prompt},
    ]
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        input_ids = input_ids.to(model.device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,  # allow longer outputs
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return tokenizer.decode(response, skip_special_tokens=True)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logging.error(f"CUDA out of memory while processing paper: {title}")
            log_error(f"CUDA out of memory for paper: {title}", error_log_path)
            torch.cuda.empty_cache()  # free GPU cache
            return "CUDA out of memory"
        else:
            raise e

def fix_scores_with_model(papers, output_json_path, error_log_path, max_retries=5):
    """
    Iterate over papers and repair feasibility/effectiveness scores stored as raw strings.
    If parsing fails, regenerate the scores up to max_retries times.
    """
    updated_papers = []

    for paper in tqdm(papers, desc="Fixing Papers"):
        title = paper.get('title', 'unknown_title')
        content = paper.get('content', {})
        abstract = content.get('abstract', 'N/A')
        method = content.get('method', 'N/A')
        experiment = content.get('experiment', 'N/A')
        reviews = paper.get('reviews', [])
        meta_review = paper.get('meta_review', 'N/A')

        # Check feasibility_score and effectiveness_score
        feasibility_score = paper.get('feasibility_score')
        effectiveness_score = paper.get('effectiveness_score')

        # If stored as a string, try to extract JSON
        if isinstance(feasibility_score, str) or isinstance(effectiveness_score, str):
            retries = 0
            success = False

            while retries < max_retries and not success:
                # First try to parse the existing string
                feasibility_dict = extract_json_from_string(feasibility_score) if isinstance(feasibility_score, str) else None
                effectiveness_dict = extract_json_from_string(effectiveness_score) if isinstance(effectiveness_score, str) else None

                if feasibility_dict is not None and effectiveness_dict is not None:
                    # On success, update and stop retrying
                    paper['feasibility_score'] = feasibility_dict.get('feasibility', {})
                    paper['effectiveness_score'] = effectiveness_dict.get('effectiveness', {})
                    success = True
                else:
                    # Parsing failed; regenerate the scores
                    logging.info(f"Regenerating feasibility and effectiveness for: {title} (Retry {retries + 1})")
                    evaluation = regenerate_scores_for_paper(
                        title, abstract, method, experiment, reviews, meta_review, error_log_path
                    )
                    retries += 1

                    if evaluation:
                        try:
                            evaluation_dict = json.loads(evaluation)
                            paper['feasibility_score'] = evaluation_dict.get('feasibility', {})
                            paper['effectiveness_score'] = evaluation_dict.get('effectiveness', {})
                            success = True
                        except json.JSONDecodeError:
                            logging.error(f"Failed to parse JSON output for paper: {title} (Retry {retries})")
                            log_error(f"JSON parsing error for paper: {title} (Retry {retries})", error_log_path)

            # Log an error if all retries failed
            if not success:
                logging.error(f"Failed to generate valid JSON after {max_retries} retries for paper: {title}")
                log_error(f"Failed to generate valid JSON after {max_retries} retries for paper: {title}", error_log_path)
                paper['feasibility_score'] = f"Failed after {max_retries} retries"
                paper['effectiveness_score'] = f"Failed after {max_retries} retries"

        updated_papers.append(paper)

        # Save after each paper
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(updated_papers, f, ensure_ascii=False, indent=4)

    return updated_papers

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Fix feasibility and effectiveness scores using LLM with retry mechanism.")
    parser.add_argument('input_json', type=str, help="Path to the input JSON file containing paper data")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory to save output JSON and logs")
    parser.add_argument('--max_retries', type=int, default=3, help="Maximum number of retries if parsing fails")

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    error_log_path = os.path.join(output_dir, 'error_log.txt')
    output_json_path = os.path.join(output_dir, "fixed_papers_with_scores.json")

    with open(args.input_json, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    updated_papers = fix_scores_with_model(papers, output_json_path, error_log_path, args.max_retries)

    logging.info(f"Updated JSON saved to {output_json_path}")
    print(f"Error log saved to: {error_log_path}")

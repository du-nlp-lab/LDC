import os
import json
import time
import logging
from tqdm import tqdm
import argparse
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

# Evaluate feasibility and effectiveness in a single call
def evaluate_paper_with_llama(title, abstract, method, experiment, reviews, meta_review, error_log_path):
    # Prepare the reviews text
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

Feasibility Evaluation Instructions:
Evaluate how feasible it is to implement and execute this idea as a research project. Specifically, how feasible the idea is for a typical CS PhD student to execute within 1-2 months of time. You can assume that we have abundant OpenAI / Anthropic API access, but limited GPU compute.

Please assign a feasibility score on a scale from 1 to 10 based on the following criteria:

1 Impossible: The idea doesn’t make sense or the proposed experiments are flawed and cannot be implemented.

2

3 Very challenging: There are flaws in the proposed method or experiments, or the experiments require compute/human resources beyond any academic lab.

4

5 Moderately feasible: It can probably be executed within the given time frame but would require careful planning, efficient use of APIs, or some advanced computational strategies to overcome the limited GPU resources, and would require some modifications to the original proposal to make it work.

6 Feasible: Can be executed within the given constraints with some reasonable planning.

7

8 Highly Feasible: Straightforward to implement the idea and run all the experiments.

9

10 Easy: The whole proposed project can be quickly executed within a few days without requiring advanced technical skills.

Feasibility Rationale:
After assigning a score, please provide a short justification for your rating. If you give a low score, specify what parts are difficult to execute and why. Your rationale should be at least 2-3 sentences.

Effectiveness Evaluation Instructions:
Evaluate how likely the proposed idea is going to work well (e.g., better than existing baselines).

Please assign an effectiveness score on a scale from 1 to 10 based on the following criteria:

1 Extremely Unlikely: The idea has major flaws and definitely won’t work well

2

3 Low Effectiveness: The idea might work in some special scenarios but you don’t expect it to work in general

4

5 Somewhat ineffective: There might be some chance that the proposed idea can work better than existing baselines but the improvement will be marginal or inconsistent

6 Somewhat effective: There is a decent chance that the proposed idea can beat existing baselines by moderate margins on a few benchmarks

7

8 Probably Effective: The idea should offer some significant improvement over current methods on the relevant benchmarks

9

10 Definitely Effective: You are very confident that the proposed idea will outperform existing methods by significant margins on many benchmarks

Effectiveness Rationale:
After assigning a score, please provide a short justification for your rating. Your rationale should be at least 2-3 sentences.

Now provide your feasibility and effectiveness scores and reasoning based on the information above in this JSON format:
{{
    "feasibility": {{
        "score": a number (1-10),
        "reason": reason sentence
    }},
    "effectiveness": {{
        "score": a number (1-10),
        "reason": reason sentence
    }}
}}

Here's an example output JSON:
{{
    "feasibility": {{
        "score": 8,
        "reason": "The proposed method is straightforward to implement using existing libraries and APIs, and the experiments can be conducted with minimal computational resources. The research can be executed within the given time frame without requiring advanced technical skills."
    }},
    "effectiveness": {{
        "score": 8,
        "reason": "The proposed method is innovative and addresses key limitations in current approaches. It is likely to outperform existing baselines on several benchmarks due to its novel architecture and comprehensive evaluation plan."
    }}
}}

Now provide your feasibility and effectiveness scores and reasoning in JSON format (Note that you can only output the JSON content):
</>
    """
    sys_prompt = "You are a specialized assistant for scientific text evaluation. Your task is to evaluate the feasibility and expected effectiveness of scientific papers.\n"
    return generate_response(prompt, sys_prompt, title, error_log_path)

# Shared generation helper
def generate_response(prompt, sys_prompt, title, error_log_path):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]
    try:
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
            max_new_tokens=256,  # allow longer outputs
            eos_token_id=terminators,
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
            torch.cuda.empty_cache()
            return "CUDA out of memory"
        else:
            raise e

def process_papers(papers, output_json_path, error_log_path):
    # Load existing results (resume support)
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r', encoding='utf-8') as f:
            updated_papers = json.load(f)
    else:
        updated_papers = []

    existing_titles = {paper['title'] for paper in updated_papers}

    for paper in tqdm(papers, desc="Processing Papers"):
        title = paper.get('title', 'unknown_title')

        # Skip papers that were already processed
        if title in existing_titles:
            logging.info(f"Skipping already processed paper: {title}")
            continue

        content = paper.get('content', {})
        abstract = content.get('abstract', 'N/A')
        method = content.get('method', 'N/A')
        experiment = content.get('experiment', 'N/A')
        reviews = paper.get('reviews', [])
        meta_review = paper.get('meta_review', 'N/A')

        # Prepare the reviews text
        if isinstance(reviews, list):
            reviews_text = ' '.join([
                review['content'].get('summary', '') + ' ' +
                review['content'].get('weaknesses', '')
                for review in reviews
            ])
        else:
            reviews_text = reviews

        # Run the evaluation and collect the scores
        evaluation = evaluate_paper_with_llama(
            title, abstract, method, experiment, reviews_text, meta_review, error_log_path
        )

        if evaluation:
            logging.info(f"Extracted scores for {title}: {evaluation}")
            try:
                # Try to parse the model output as JSON
                evaluation_dict = json.loads(evaluation)
                paper['feasibility_score'] = evaluation_dict.get('feasibility', {})
                paper['effectiveness_score'] = evaluation_dict.get('effectiveness', {})
            except json.JSONDecodeError:
                logging.error(f"Failed to parse JSON output for paper: {title}")
                log_error(f"JSON parsing error for paper: {title}", error_log_path)
                paper['feasibility_score'] = f"Parsing Error: {evaluation}"
                paper['effectiveness_score'] = f"Parsing Error: {evaluation}"
        else:
            logging.warning(f"Failed to extract scores for {title}.")
            paper['feasibility_score'] = "N/A"
            paper['effectiveness_score'] = "N/A"

        updated_papers.append(paper)

        # Save after each paper
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(updated_papers, f, ensure_ascii=False, indent=4)

    return updated_papers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process papers and extract feasibility and effectiveness scores.")
    parser.add_argument('input_json', type=str, help="Path to the input JSON file containing paper data")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory to save output JSON and logs")

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    error_log_path = os.path.join(output_dir, 'error_log.txt')
    output_json_path = os.path.join(output_dir, "updated_papers_with_scores.json")

    with open(args.input_json, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    updated_papers = process_papers(papers, output_json_path, error_log_path)

    logging.info(f"Updated JSON saved to {output_json_path}")
    print(f"Error log saved to: {error_log_path}")

import os
import json
import time
import logging
from tqdm import tqdm
import argparse
from openai import OpenAI
from llamaapi import LlamaAPI
import openai

client = OpenAI(
    api_key="",
    base_url="https://api.llama-api.com"
)

class LlamaThread:
    def __init__(self, sys_message, model, name, temp = 0.01, max_token=128000):
        self.model = model
        self.temp = temp
        self.name = name
        self.max_token = max_token
        self.llama = LlamaAPI("")
        self.messages = [{'role': 'system', 'content': sys_message}]

    def respond(self, message, function=None, function_call=None, need_response=True):
        self.messages.append({'role': 'user', 'content': message})

        if not need_response:
            return None

        attempts = 0
        response_content = None

        api_request_json = {
            "model": self.model,
            "messages": self.messages,
            "max_token": self.max_token,
            "temperature": self.temp,
            "stream": False
        }

        if function:
            api_request_json["functions"] = [function]
        if function_call:
            api_request_json["function_call"] = function_call

        while attempts < 5:
            try:
                response = self.llama.run(api_request_json)
                response_content = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                break
            except Exception as e:
                if attempts < 4:
                    print(f"Error occurred: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
                    attempts += 1
                else:
                    raise Exception(f"Error after 5 attempts: {e}")

        if response_content:
            self.messages.append({'role': 'assistant', 'content': response_content})
        return response_content

    def clear_history(self):
        self.messages = [self.messages[0]]


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_error(error_message, error_log_path):
    with open(error_log_path, 'a') as f:
        f.write(error_message + '\n')

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
    sys_prompt = "You are a specialized assistant for scientific text evaluation. Your task is to evaluate the novelty of scientific papers."
    llama_thread = LlamaThread(sys_prompt, "llama3.1-70b", "novelty_evaluation")

    try:
        response = llama_thread.respond(prompt)
        return response
    except Exception as e:
        error_message = f"Error while using Llama to extract novelty score: {e}; length:{len(prompt)}"
        log_error(error_message, error_log_path)
        return None
def reviews_to_text(reviews):
    """
    Convert the reviews JSON into plain text.
    """
    reviews_texts = []
    for review in reviews:
        reviewer = review.get('reviewer', 'Unknown Reviewer')
        content = review.get('content', {})
        summary = content.get('summary', 'No summary provided.')
        soundness = content.get('soundness', 'No soundness score provided.')
        presentation = content.get('presentation', 'No presentation score provided.')
        contribution = content.get('contribution', 'No contribution score provided.')
        strengths = content.get('strengths', 'No strengths provided.')
        weaknesses = content.get('weaknesses', 'No weaknesses provided.')
        questions = content.get('questions', 'No questions provided.')
        flag_for_ethics_review = ', '.join(content.get('flag_for_ethics_review', ['No ethics review flag provided.']))
        rating = content.get('rating', 'No rating provided.')
        confidence = content.get('confidence', 'No confidence level provided.')
        code_of_conduct = content.get('code_of_conduct', 'No code of conduct mentioned.')

        review_text = f"Reviewer: {reviewer}\n" \
                      f"Summary: {summary}\n\n" \
                      f"Soundness: {soundness}\n" \
                      f"Presentation: {presentation}\n" \
                      f"Contribution: {contribution}\n\n" \
                      f"Strengths: {strengths}\n\n" \
                      f"Weaknesses: {weaknesses}\n\n" \
                      f"Questions: {questions}\n\n" \
                      f"Ethics Review: {flag_for_ethics_review}\n" \
                      f"Rating: {rating}\n" \
                      f"Confidence: {confidence}\n" \
                      f"Code of Conduct: {code_of_conduct}\n\n" \
                      f"{'-'*40}\n"
        reviews_texts.append(review_text)

    return '\n'.join(reviews_texts)

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

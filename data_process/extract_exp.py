import os
import json
import time
import logging
from tqdm import tqdm
import fitz  # PyMuPDF, for PDF parsing
import requests
import argparse
from openai import OpenAI
import openai

client = OpenAI(
    api_key="",
    base_url="https://api.llama-api.com"
)

class LlamaThread:
    def __init__(self, sys_message, model, temp, name):
        self.model = model
        self.temp = temp
        self.name = name
        self.messages = [{'role': 'system', 'content': sys_message}]

    def respond(self, message, need_response=True):
        self.messages.append({'role': 'user', 'content': message})
        if not need_response:
            return None

        attempts = 0
        response_content = None
        while attempts < 5:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=self.temp
                )
                response_content = response.choices[0].message.content
                break
            except openai.OpenAIError as e:
                if attempts < 4:
                    print(f"Error occurred: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
                    attempts += 1
                else:
                    raise Exception(f"Error after 5 attempts: {e}")

        self.messages.append({'role': 'assistant', 'content': response_content})
        return response_content

    def clear_history(self):
        self.messages = [self.messages[0]]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Error logging helper
def log_error(error_message, error_log_path):
    with open(error_log_path, 'a') as f:
        f.write(error_message + '\n')

# Extract the experiment section with the LLM
def extract_experiment_section_with_llama(full_text, error_log_path):
    prompt = f"""You need to extract the original text of the experiment section from the paper's text within the <></> tags, including tables presented in textual form: \n
    <>
    {full_text}
    </>
    Now provide the original text of the experiment section you extracted. Note that you are only allowed to output the extracted original text:
    """
    print(len(full_text))
    sys_prompt = "You are a specialized assistant for scientific text extraction. Your task is to accurately identify and extract the experiment section from an academic paper. Ensure precise extraction for further analysis."
    llama_thread = LlamaThread(sys_prompt, "llama3.1-70b", 0.01, "extract_experimentSection")

    try:
        response = llama_thread.respond(prompt)
        return response
    except Exception as e:
        error_message = f"Error while using Llama to extract experiment section: {e}"
        log_error(error_message, error_log_path)
        return None

# Extract performance metrics with the LLM
def extract_performance_metrics_with_llama(experiment_text, error_log_path):
    prompt1 = f""""
    Within the <></> tags below, I have provided the original text of the experiment section extracted from the paper. Your task is to identify and extract the main metric used in the experiment from this text: \n
    <> \n
    {experiment_text}
    </> \n
    Note that you only need to answer what the main metric is. If the metric is complex, you may also provide further explanation.
    If you believe that the provided experiment section text does not contain enough valid information, simply return the text 'Not found',
    Now provide the main metric you extracted:
    """
    sys_prompt = "You are a specialized assistant for scientific text extraction. Your task is to accurately identify and extract the information from the extracted experiment section text. Ensure precise extraction."
    llama_thread = LlamaThread(sys_prompt, "llama3.1-70b", 0.01, "further paper extract")

    try:
        response1 = llama_thread.respond(prompt1)
        prompt2 = f"""
        Having extracted the main metric in the previous step, your task now is to extract the baseline performance from the experiment section, based on the identified main metric: {response1}. \n
        If you believe that the provided experiment section text does not contain enough valid information, simply return the text 'Not found'.
        Now provide the extracted baseline performance:
        """
        try:
            response2 = llama_thread.respond(prompt2)
        except Exception as e:
            response2 = None
            error_message = f"Error while using Llama to extract performance metrics: {e}"
            log_error(error_message, error_log_path)

        prompt3 = f"""
        Having extracted the main metric in the previous step, your task now is to extract the performance corresponding to the paper's proposed idea from the experiment section, based on the identified main metric: {response1}. \n
        If you believe that the provided experiment section text does not contain enough valid information, simply return the text 'Not found'.
        Now provide the extracted idea performance:
        """
        try:
            response3 = llama_thread.respond(prompt3)
        except Exception as e:
            response3 = None
            error_message = f"Error while using Llama to extract performance metrics: {e}"
            log_error(error_message, error_log_path)
        return response1,response2,response3
    except Exception as e:
        error_message = f"Error while using Llama to extract performance metrics: {e}"
        log_error(error_message, error_log_path)
        return None, None, None

# Extract text from a PDF
def extract_text_from_pdf(pdf_path, error_log_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        error_message = f"Error while extracting text from PDF: {pdf_path}. Error: {e}"
        log_error(error_message, error_log_path)
        return None

# Download the PDF and extract its text
def download_pdf_and_extract(forum_id, title, save_dir, error_log_path):
    pdf_url = f"https://openreview.net/pdf?id={forum_id}"
    try:
        response = requests.get(pdf_url)
        if response.status_code == 200:
            file_path = os.path.join(save_dir, f"{title}.pdf")
            with open(file_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Downloaded paper: {title}")
            return extract_text_from_pdf(file_path, error_log_path)
        else:
            raise Exception(f"Failed to download PDF, status code: {response.status_code}")
    except Exception as e:
        error_message = f"Failed to download PDF for {title} (forum_id: {forum_id}): {e}"
        logging.error(error_message)
        log_error(error_message, error_log_path)
        return None

# Process each paper
def process_papers(papers, output_dir, error_log_path):
    updated_papers = []

    for paper in tqdm(papers, desc="Processing Papers"):
        title = paper.get('title', 'unknown_title')
        experiment = paper.get('content', {}).get('experiment', 'N/A')
        forum_id = paper.get('forum_id')

        # If the experiment section exists, extract the three metrics
        if paper['content'].get('experiment') and paper['content']['experiment'] != 'N/A':
            experiment_text = paper['content']['experiment']
            r1,r2,r3 = extract_performance_metrics_with_llama(experiment_text, error_log_path)

            if r1:
                logging.info(f"Extracted metrics for {title}: {r1}")
                paper['effectiveness_score'] = {
                    "main_metric": r1,
                    "baseline": r2,
                    "idea_performance": r3
                }
            else:
                logging.warning(f"Failed to extract metrics for {title}.")

        updated_papers.append(paper)

    return updated_papers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process papers, extract experiment sections and performance metrics.")
    parser.add_argument('input_json', type=str, help="Path to the input JSON file containing paper data")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory to save PDFs, text and JSON")
    parser.add_argument('--save_intermediate', action='store_true', help="Whether to save intermediate txt and section JSON files")
    args = parser.parse_args()

    output_dir = args.output_dir
    error_log_path = os.path.join(output_dir, 'error_log.txt')

    with open(args.input_json, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    updated_papers = process_papers(papers, output_dir, error_log_path)

    output_json = os.path.join(output_dir, "updated_metrics.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(updated_papers, f, ensure_ascii=False, indent=4)

    logging.info(f"Updated JSON saved to {output_json}")
    print(f"Error log saved to: {error_log_path}")

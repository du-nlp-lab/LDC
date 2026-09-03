import re
import os
import json
import argparse
import fitz  # PyMuPDF, for PDF parsing
import requests
from tqdm import tqdm
from fuzzywuzzy import fuzz  # fuzzy string matching
import logging
import traceback

# Synonym lists for section-title matching
synonyms = {
    'introduction': ['Introduction'],
    'related_work': ['Related Work', 'Previous Work', 'Literature Review', 'State of the Art', 'Background', 'Existing Work', 'Related Research'],
    'method': ['Method', 'Approach', 'Methodology', 'Implementation'],
    'experiment': ['Experiment', 'Evaluation', 'Results', 'Performance', 'Analysis']
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Error logging helper
def log_error(error_message, error_log_path):
    """
    Append an error message to the log file.
    """
    with open(error_log_path, 'a') as f:
        f.write(error_message + '\n')

# Download the PDF via OpenReview
def download_pdf_by_forum_id(forum_id, title, save_dir, error_log_path=None):
    """
    Download a paper PDF by its OpenReview forum_id and save it locally.
    """
    try:
        pdf_url = f"https://openreview.net/pdf?id={forum_id}"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        response = requests.get(pdf_url)
        if response.status_code == 200:
            file_path = os.path.join(save_dir, f"{title}.pdf")
            with open(file_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Downloaded paper: {title}")
            return file_path
        else:
            raise Exception(f"Failed to download PDF, status code: {response.status_code}")

    except Exception as e:
        error_message = f"Failed to download paper {title} (forum_id: {forum_id}): {e}"
        logging.error(error_message)
        if error_log_path:
            log_error(error_message, error_log_path)
        return None

# Extract text from a PDF
def extract_text_from_pdf(pdf_path, error_log_path=None):
    """
    Extract plain text from the PDF, logging any errors.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        error_message = f"Failed to extract text from {pdf_path}: {traceback.format_exc()}"
        logging.error(error_message)
        if error_log_path:
            log_error(error_message, error_log_path)
        return None

# Extract section titles and contents
def extract_sections_from_text(content):
    """
    Extract section titles and contents from the text and return them as a list.
    Section numbers must be increasing single digits not preceded by a '.'.
    """
    sections = []

    # Regex for section headings
    section_pattern = re.compile(r'\n(\d+)\s*\n(?!Published)([A-Za-z0-9\s:\"\-\+\*/]+)\n', re.MULTILINE)

    # Find all heading matches
    matches = list(section_pattern.finditer(content))

    # Text before the first heading is kept as the starting part
    if matches:
        first_match_start = matches[0].start()
        starting_content = content[:first_match_start].strip()
        if starting_content:
            sections.append({
                'title': 'starting',
                'content': starting_content
            })

    # Track that section numbers increase
    previous_chapter_number = 0

    # Iterate over matches and slice out each section
    for i, match in enumerate(matches):
        chapter_number = int(match.group(1))

        # Check that section numbers are increasing
        if chapter_number != previous_chapter_number + 1:
            continue  # skip non-increasing section numbers

        # Combine the number and heading as the title
        title = f"{match.group(1)} {match.group(2).strip()}"  # section number and heading
        start = match.end()  # start of this section's content

        if i + 1 < len(matches):
            end = matches[i + 1].start()  # start of the next section
        else:
            end = len(content)  # end of the last section

        section_content = content[start:end].strip()
        sections.append({
            'title': title,
            'content': section_content
        })

        # Update the previous section number
        previous_chapter_number = chapter_number

    return sections

# Return all synonym groups whose similarity exceeds the threshold
def match_section_title(section_title, target_synonyms):
    """
    Fuzzy-match the section title against each synonym list and return all matches above the threshold.
    """
    matched_sections = []
    for key, target_group in target_synonyms.items():
        for target in target_group:
            ratio = fuzz.partial_ratio(section_title.lower(), target.lower())
            if ratio > 80:  # similarity threshold
                matched_sections.append(key)
                # print(f"Matched title: '{section_title}' with synonym: '{target}' for section: {key} (Ratio: {ratio})")
    return matched_sections if matched_sections else []

# Update the JSON record with the concatenated matched sections
def update_json_with_sections(json_data, sections):
    """
    Fuzzy-match sections with the synonym lists and concatenate the matched contents into the JSON record in order.
    """
    # Initialize an empty buffer for each target section
    section_map = {
        'related_work': '',
        'method': '',
        'experiment': '',
        'introduction': ''
    }
    added_sections = {
        'related_work': set(),
        'method': set(),
        'experiment': set(),
        'introduction': set()
    }

    for section in sections:
        matched_keys = match_section_title(section['title'], synonyms)  # all matches above the threshold
        for key in matched_keys:
            if section['title'] not in added_sections[key]:  # skip titles already added
                section_content = f"Section: {section['title']}\n{section['content']}\n\n"
                section_map[key] += section_content
                added_sections[key].add(section['title'])  # mark as added
                # print(f"Appending section '{section['title']}' to {key}")

    # Write the concatenated contents back to the record
    for key, content in section_map.items():
        if content:  # only if non-empty
            json_data['content'][key] = content

    return json_data

# Save extracted sections to JSON
def save_to_json(data, json_path):
    """
    Save the extracted sections as a JSON file.
    """
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Sections saved to {json_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download PDFs, extract text, update JSON data and save.")
    parser.add_argument('input_json', type=str, help="Path to the input JSON file containing paper data")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory to save PDFs, text and JSON")
    parser.add_argument('--save_intermediate', action='store_true', help="Whether to save intermediate txt and section JSON files")

    args = parser.parse_args()

    with open(args.input_json, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    output_dir = args.output_dir
    error_log_path = os.path.join(output_dir, 'error_log.txt')

    processed_papers = []

    for paper in tqdm(papers, desc="Processing Papers"):
        title = paper.get('title', 'unknown_title')
        forum_id = paper.get('forum_id')

        if not forum_id:
            logging.warning(f"No forum_id found for paper: {title}")
            processed_papers.append(paper)
            continue

        # Download the PDF
        pdf_path = download_pdf_by_forum_id(forum_id, title, output_dir, error_log_path)
        if not pdf_path:
            processed_papers.append(paper)
            continue

        # Extract the PDF text
        extracted_text = extract_text_from_pdf(pdf_path, error_log_path)
        if not extracted_text:
            processed_papers.append(paper)
            continue

        if args.save_intermediate:
            text_save_path = os.path.join(output_dir, f"{title}.txt")
            with open(text_save_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)

        sections = extract_sections_from_text(extracted_text)

        if args.save_intermediate:
            section_save_path = os.path.join(output_dir, f"{title}_sections.json")
            save_to_json(sections, section_save_path)

        updated_paper = update_json_with_sections(paper, sections)
        processed_papers.append(updated_paper)

    output_json_path = os.path.join(output_dir, "updated_papers.json")
    save_to_json(processed_papers, output_json_path)

    print(f"Error log saved to: {error_log_path}")

import json
import os
import re
import logging
from tqdm import tqdm

def extract_json_from_string(text):
    """
    Extract the substring from the first '{' to the last '}' and parse it as JSON.
    """
    try:
        # Extract the JSON between '{' and '}'
        json_str = re.search(r'{.*}', text, re.DOTALL)
        if json_str:
            return json.loads(json_str.group())
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from text: {text}. Error: {e}")
    return None

def repair_scores_in_papers(papers, output_json_path):
    """
    Repair novelty/feasibility/effectiveness scores that were stored as raw strings.
    Report the fraction of scores that remain unparsable.
    """
    total_papers = len(papers)
    feasibility_string_count = 0
    effectiveness_string_count = 0
    novelty_string_count = 0
    updated_papers = []

    for paper in tqdm(papers, desc="Repairing Papers"):
        # Check feasibility_score
        feasibility_score = paper.get('feasibility_score')
        if isinstance(feasibility_score, str):
            feasibility_dict = extract_json_from_string(feasibility_score)
            if feasibility_dict:
                paper['feasibility_score'] = feasibility_dict.get('feasibility', feasibility_score)
            else:
                feasibility_string_count += 1  # still a raw string after repair

        # Check effectiveness_score
        effectiveness_score = paper.get('effectiveness_score')
        if isinstance(effectiveness_score, str):
            effectiveness_dict = extract_json_from_string(effectiveness_score)
            if effectiveness_dict:
                paper['effectiveness_score'] = effectiveness_dict.get('effectiveness', effectiveness_score)
            else:
                effectiveness_string_count += 1  # still a raw string after repair

        # Check novelty_score
        novelty_score = paper.get('novelty_score')
        if isinstance(novelty_score, str):
            novelty_dict = extract_json_from_string(novelty_score)
            if novelty_dict:
                paper['novelty_score'] = novelty_dict
            else:
                novelty_string_count += 1  # still a raw string after repair

        updated_papers.append(paper)

    # Save the repaired results
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(updated_papers, f, ensure_ascii=False, indent=4)

    # Compute the remaining-string ratios
    feasibility_string_ratio = feasibility_string_count / total_papers if total_papers > 0 else 0
    effectiveness_string_ratio = effectiveness_string_count / total_papers if total_papers > 0 else 0
    novelty_string_ratio = novelty_string_count / total_papers if total_papers > 0 else 0

    print(f"Total papers: {total_papers}")
    print(f"Feasibility score still as string: {feasibility_string_count} ({feasibility_string_ratio:.2%})")
    print(f"Effectiveness score still as string: {effectiveness_string_count} ({effectiveness_string_ratio:.2%})")
    print(f"Novelty score still as string: {novelty_string_count} ({novelty_string_ratio:.2%})")

    return updated_papers, feasibility_string_ratio, effectiveness_string_ratio, novelty_string_ratio

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Repair score fields stored as raw strings and report statistics.")
    parser.add_argument('input_json', type=str, help="Path to the input JSON file containing scored paper data")
    parser.add_argument('--output_json', type=str, default="papers_reparsed.json", help="Path to save the repaired JSON")
    args = parser.parse_args()

    with open(args.input_json, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    # Repair the three score fields and report how many remain unparsable
    repair_scores_in_papers(papers, args.output_json)

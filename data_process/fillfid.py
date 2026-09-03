import os
import openreview
import json
import logging
from fuzzywuzzy import fuzz
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_papers_from_openreview(client, conference_id):
    """
    Fetch accepted papers from OpenReview.
    """
    try:
        logging.info(f"Fetching papers from OpenReview for {conference_id}")
        venue_group = client.get_group(conference_id)
        submission_name = venue_group.content['submission_name']['value']
        decision_name = venue_group.content['decision_name']['value']

        logging.info(f"submission_name: {submission_name}, decision_name: {decision_name}")

        # Fetch all submissions
        submissions = client.get_all_notes(invitation=f'{conference_id}/-/{submission_name}', details='replies')
        logging.info(f"Fetched {len(submissions)} submissions")

        # Keep accepted papers only
        accepted_papers = []
        for submission in submissions:
            forum_id = submission.forum
            replies = client.get_all_notes(forum=forum_id, details='directReplies')
            for reply in replies:
                if hasattr(reply, 'invitations') and isinstance(reply.invitations, list):
                    for invitation in reply.invitations:
                        if decision_name in invitation and 'Accept' in reply.content.get('decision', {}).get('value', ''):
                            accepted_papers.append({
                                'title': submission.content.get('title', 'No Title'),
                                'forum_id': forum_id
                            })
                            break
        return accepted_papers
    except Exception as e:
        logging.error(f"Error fetching papers: {e}")
        return []

def update_json_with_forum_ids(json_data, accepted_papers):
    """
    Fuzzy-match each JSON entry by title and fill in its forum_id ('N/A' if no match).
    """
    for entry in tqdm(json_data, desc="Updating JSON with forum_ids"):
        json_title = entry.get('title', '')
        best_match = None
        best_ratio = 0
        for paper in accepted_papers:
            # partial_ratio allows partial matches; threshold set to 80
            paper_title = paper['title'].get('value', '') if isinstance(paper['title'], dict) else str(paper['title'])
            match_ratio = fuzz.partial_ratio(json_title.lower(), paper_title.lower())
            if match_ratio > best_ratio and match_ratio > 80:
                best_match = paper
                best_ratio = match_ratio

        if best_match:
            entry['forum_id'] = best_match['forum_id']
        else:
            entry['forum_id'] = 'N/A'  # no match found

    return json_data

def save_to_json(data, filename):
    """
    Save the updated JSON data.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logging.info(f"Data saved to {filename}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Fill in OpenReview forum_ids by fuzzy title matching.")
    parser.add_argument('input_json', type=str, help="Path to the input JSON file containing paper data")
    parser.add_argument('--output_json', type=str, default="papers_with_forum_ids.json", help="Path to save the updated JSON")
    parser.add_argument('--conference', type=str, default='ICLR', help='Conference name')
    parser.add_argument('--year', type=str, default='2024', help='Conference year')
    args = parser.parse_args()

    client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net', username=os.environ.get('OPENREVIEW_USERNAME'), password=os.environ.get('OPENREVIEW_PASSWORD'))

    conference_id = f'{args.conference}.cc/{args.year}/Conference'
    accepted_papers = get_papers_from_openreview(client, conference_id)

    with open(args.input_json, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    updated_json_data = update_json_with_forum_ids(json_data, accepted_papers)
    save_to_json(updated_json_data, args.output_json)

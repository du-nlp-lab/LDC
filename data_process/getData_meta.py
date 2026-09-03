import os
import openreview
import time
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_accepted_forum_ids(blind_notes):
    """
    Collect accepted papers from the submissions.
    """
    forum_ids = set()
    for note in blind_notes:
        for reply in note.details["directReplies"]:
            if reply["invitation"].endswith("Decision") and 'Accept' in reply["content"]['decision']:
                forum_ids.add(reply['forum'])
    return forum_ids

def get_paper_reviews(forum_id):
    """
    Fetch the official reviews and meta-review of a paper.
    """
    reviews = []
    meta_review = None
    replies = client.get_notes(forum=forum_id)
    for reply in replies:
        if reply.invitation.endswith('Official_Review'):  # official reviews
            reviews.append({
                'reviewer': reply.signatures[0],  # reviewer signature
                'content': reply.content  # full review content (scores, strengths/weaknesses, etc.)
            })
        elif reply.invitation.endswith('Meta_Review') or reply.invitation.endswith('Decision'):
            meta_review = {
                'meta_reviewer': reply.signatures[0],  # meta-reviewer signature
                'content': reply.content  # meta-review content (final recommendation)
            }
    return reviews, meta_review

def format_note_with_reviews(note, conference_name):
    """
    Format a paper record with its reviews and meta-review.
    """
    authors_string = ','.join(note.content['authors'])
    tags_string = ','.join(note.content['keywords']) if 'keywords' in note.content else 'N/A'
    localTime = time.localtime(note.pdate / 1000)
    strTime = time.strftime('%Y-%m-%d', localTime)

    reviews, meta_review = get_paper_reviews(note.forum)

    return {
        'title': note.content['title'],
        'url': 'https://openreview.net/forum?id=' + note.forum,
        'pub_date': strTime,
        'summary': note.content['abstract'],
        'authors': authors_string,
        'tags': tags_string,
        'conference': conference_name,
        'reviews': reviews,
        'meta_review': meta_review
    }

def get_papers_from_openreview(conference_id):
    """
    Fetch accepted papers with reviews and meta-reviews from OpenReview.
    """
    try:
        logging.info(f"Fetching papers from OpenReview for {conference_id}")
        submissions = client.get_all_notes(
            invitation=conference_id + '/Conference/-/Blind_Submission', details='directReplies')
        accepted_forum_ids = get_accepted_forum_ids(submissions)
        notes_list = [format_note_with_reviews(note, conference_name) for note in submissions if note.forum in accepted_forum_ids]
        logging.info(f"Found {len(notes_list)} accepted papers for {conference_id}")
        return notes_list
    except Exception as e:
        logging.error(f"Error fetching papers: {e}")
        return []

def save_to_json(data, filename):
    """
    Save data to a local JSON file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logging.info(f"Data saved to {filename}")

if __name__ == '__main__':
    client = openreview.Client(baseurl='https://api.openreview.net', username=os.environ.get('OPENREVIEW_USERNAME'), password=os.environ.get('OPENREVIEW_PASSWORD'))

    conference_name = 'ICLR'
    conference_year = '2023'

    paper_list = get_papers_from_openreview(conference_name + '.cc/' + conference_year)

    if paper_list:
        save_to_json(paper_list, f"{conference_name}_{conference_year}_papers_meta_reviews.json")
    else:
        print("No papers found.")

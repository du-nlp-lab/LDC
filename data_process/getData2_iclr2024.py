import os
import openreview
import time
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Collected reply objects, kept for inspection
collected_replies = []

def get_paper_replies(forum_id):
    """
    Fetch all replies for a paper (official reviews, meta-review, decision, etc.).
    """
    try:
        logging.debug(f"Fetching replies for forum {forum_id}")
        replies = client.get_all_notes(forum=forum_id, details='directReplies')
        logging.info(f"Fetched {len(replies)} replies for forum {forum_id}")
        return replies
    except Exception as e:
        logging.error(f"Error fetching replies for forum {forum_id}: {e}")
        return []

def get_accepted_forum_ids(submissions, decision_name, venue_id):
    """
    Collect forum_ids of accepted papers from the submissions.
    """
    forum_ids = set()
    for submission in submissions:
        logging.info(f"Processing submission {submission.id}")
        replies = get_paper_replies(submission.id)
        if not replies:
            logging.info(f"No replies found for submission {submission.id}")
        for reply in replies:
            # Inspect the reply structure
            # print(f"Reply object: {reply.to_json()}")

            # Check whether the invitations list contains the decision name
            if hasattr(reply, 'invitations') and isinstance(reply.invitations, list):
                for invitation in reply.invitations:
                    suffix = invitation.split('/')[-1]  # part after the last slash
                    if suffix == decision_name:
                        # print("found ==")
                        # Keep the reply for inspection
                        collected_replies.append(reply.to_json())
                        # Check whether the decision is 'Accept'
                        print(f"content: {reply.content.get('decision', {}).get('value', '')}; {'Accept' in reply.content.get('decision', {}).get('value', '')}")
                        if 'Accept' in reply.content.get('decision', {}).get('value', ''):
                            # print("ACCEPT!")
                            forum_ids.add(reply.forum)
                            logging.info(f"Accepted forum {reply.forum}")
            else:
                logging.warning(f"'invitations' not found or not a list in reply {reply.id}")
    return forum_ids

def get_paper_reviews(forum_id, review_name, meta_review_name):
    """
    Fetch the official reviews and meta-review of a paper.
    """
    reviews = []
    meta_reviews = []
    logging.debug(f"Fetching reviews for forum {forum_id}")

    # Fetch all replies including directReplies
    replies = client.get_all_notes(forum=forum_id, details='directReplies')
    logging.info(f"Fetched {len(replies)} replies for forum {forum_id}")

    for reply in replies:
        # invitations is a list; iterate over it
        if hasattr(reply, 'invitations') and isinstance(reply.invitations, list):
            for invitation in reply.invitations:
                if review_name in invitation:
                    reviews.append({
                        'reviewer': reply.signatures[0],
                        'content': reply.content
                    })
                    logging.info(f"Review found for forum {forum_id}")
                elif meta_review_name in invitation:
                    meta_reviews.append({
                        'meta_reviewer': reply.signatures[0],
                        'content': reply.content
                    })
                    logging.info(f"Meta review found for forum {forum_id}")
        else:
            logging.warning(f"'invitations' not found or not a list in reply {reply.id}")
    return reviews, meta_reviews

def format_note_with_reviews(note, conference_name, review_name, meta_review_name):
    """
    Format a paper record with its reviews and meta-review.
    """
    logging.info(f"Formatting note {note.id}")
    authors_string = ','.join(note.content.get('authors', {}).get('value', []))
    tags_string = ','.join(note.content.get('keywords', {}).get('value', 'N/A'))
    localTime = time.localtime(note.pdate / 1000)
    strTime = time.strftime('%Y-%m-%d', localTime)

    reviews, meta_reviews = get_paper_reviews(note.id, review_name, meta_review_name)

    return {
        'title': note.content.get('title', 'No Title'),
        'url': 'https://openreview.net/forum?id=' + note.forum,
        'pub_date': strTime,
        'summary': note.content.get('abstract', 'No Abstract'),
        'authors': authors_string,
        'tags': tags_string,
        'conference': conference_name,
        'reviews': reviews,
        'meta_reviews': meta_reviews
    }

def save_accepted_forum_ids_to_txt(accepted_forum_ids, filename="accepted_forum_ids.txt"):
    """
    Save accepted forum_ids to a text file.
    """
    with open(filename, 'w') as f:
        for forum_id in accepted_forum_ids:
            f.write(f"{forum_id}\n")
    logging.info(f"Accepted forum IDs have been saved to {filename}")

def get_papers_from_openreview(conference_id):
    """
    Fetch accepted papers with reviews, meta-reviews, and metadata from OpenReview.
    """
    try:
        logging.info(f"Fetching papers from OpenReview for {conference_id}")
        venue_group = client.get_group(conference_id)
        submission_name = venue_group.content['submission_name']['value']
        review_name = venue_group.content['review_name']['value']
        meta_review_name = venue_group.content['meta_review_name']['value']
        decision_name = venue_group.content['decision_name']['value']

        logging.info(f"submission_name: {submission_name}, review_name: {review_name}, meta_review_name: {meta_review_name}, decision_name: {decision_name}")

        submissions = client.get_all_notes(invitation=f'{conference_id}/-/{submission_name}', details='replies')
        logging.info(f"Fetched {len(submissions)} submissions")

        accepted_forum_ids = get_accepted_forum_ids(submissions, decision_name, conference_id)
        save_accepted_forum_ids_to_txt(accepted_forum_ids)
        logging.info(f"Accepted forum IDs: {accepted_forum_ids}")

        papers_info = [format_note_with_reviews(note, conference_id, review_name, meta_review_name) for note in submissions if note.forum in accepted_forum_ids]

        logging.info(f"Found {len(papers_info)} accepted papers for {conference_id}")
        return papers_info
    except Exception as e:
        logging.error(f"Error fetching papers: {e}", exc_info=True)
        return []

def save_to_json(data, filename):
    """
    Save data to a local JSON file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logging.info(f"Data saved to {filename}")

if __name__ == '__main__':
    client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net', username=os.environ.get('OPENREVIEW_USERNAME'), password=os.environ.get('OPENREVIEW_PASSWORD'))

    conference_name = 'ICLR'
    conference_year = '2024'

    paper_list = get_papers_from_openreview(f'{conference_name}.cc/{conference_year}/Conference')

    if paper_list:
        save_to_json(paper_list, f"{conference_name}_{conference_year}_papers_meta_reviews.json")
    else:
        print("No papers found.")

    # Save all collected reply objects
    save_to_json(collected_replies, f"{conference_name}_{conference_year}_collected_replies.json")

import openreview
import time
import logging
import json
import requests
import os
import fitz
import argparse
# pip install openreview-py PyMuPDF

def get_accepted_forum_ids(blind_notes):
    forum_ids = set()
    for note in blind_notes:
        for reply in note.details["directReplies"]:
            if reply["invitation"].endswith("Decision") and 'Accept' in reply["content"]['decision']:
                forum_ids.add(reply['forum'])
    return forum_ids

def get_paper_reviews(forum_id):
    reviews = []
    meta_review = None  
    replies = client.get_notes(forum=forum_id)
    for reply in replies:
        if reply.invitation.endswith('Official_Review'):
            reviews.append({
                'reviewer': reply.signatures[0],  
                'content': reply.content  
            })
        elif reply.invitation.endswith('Meta_Review') or reply.invitation.endswith('Decision'): 
            meta_review = {
                'meta_reviewer': reply.signatures[0],  
                'content': reply.content 
            }
    return reviews, meta_review 

def format_note_with_reviews(note, conference_name):
    authors_string = ', '.join(note.content['authors'])
    tags_string = ', '.join(note.content['keywords']) if 'keywords' in note.content else 'N/A'
    if note.pdate:
        localTime = time.localtime(note.pdate / 1000)
        strTime = time.strftime('%Y-%m-%d', localTime)
    else:
        strTime = 'N/A' 

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
    try:
        print(f'Fetching papers from OpenReview for {conference_id}')
        submissions = client.get_all_notes(
            invitation=conference_id + '/Conference/-/Blind_Submission', details='directReplies')
        accepted_forum_ids = get_accepted_forum_ids(submissions)
        accepted_notes = [note for note in submissions if note.forum in accepted_forum_ids]
        rejected_notes = [note for note in submissions if note.forum not in accepted_forum_ids]
        print(f"Found {len(accepted_notes)} accepted papers and {len(rejected_notes)} rejected papers for {conference_id}")
        return accepted_notes, rejected_notes
    except Exception as e:
        logging.error(f"Error fetching papers: {e}")
        return [], []

def download_pdf_from_openreview(forum_id, title, save_dir):
    pdf_url = f"https://openreview.net/pdf?id={forum_id}"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    response = requests.get(pdf_url)
    if response.status_code == 200:
        safe_title = ''.join(c for c in title if c.isalnum() or c in (' ', '_', '-')).rstrip()
        file_path = os.path.join(save_dir, f"{safe_title}.pdf")
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded paper: {title}")
        return file_path
    else:
        logging.warning(f"Failed to download paper: {title}")
        return None

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def save_text_to_file(text, title, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    safe_title = ''.join(c for c in title if c.isalnum() or c in (' ', '_', '-')).rstrip()
    file_path = os.path.join(save_dir, f"{safe_title}.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved extracted text: {title}")

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch papers from OpenReview.')
    parser.add_argument('--download_pdfs', action='store_true', help='Download PDFs')
    parser.add_argument('--pdf_save_dir', type=str, default='./papers', help='Directory to save PDFs')
    parser.add_argument('--conference', type=str, default='ICLR', help='Conference name (e.g., ICLR)')
    parser.add_argument('--year', type=str, default='2023', help='Conference year')
    args = parser.parse_args()

    # Credentials are read from environment variables; never hardcode them.
    #   export OPENREVIEW_USERNAME=your_email
    #   export OPENREVIEW_PASSWORD=your_password
    client = openreview.Client(
        baseurl='https://api.openreview.net',
        username=os.environ['OPENREVIEW_USERNAME'],
        password=os.environ['OPENREVIEW_PASSWORD'],
    )

    conference_name = args.conference
    conference_year = args.year

    accepted_notes, rejected_notes = get_papers_from_openreview(conference_name + '.cc/' + conference_year)

    if accepted_notes:
        accepted_paper_list = [format_note_with_reviews(note, conference_name) for note in accepted_notes]

        output_filename_accepted = f"{conference_name}_{conference_year}_accepted_papers_reviews_meta.json"
        save_to_json(accepted_paper_list, output_filename_accepted)

        if args.download_pdfs:
            save_dir_accepted = os.path.join(args.pdf_save_dir, 'accepted')
            for paper in accepted_paper_list:
                pdf_path = download_pdf_from_openreview(paper['url'].split('=')[-1], paper['title'], save_dir_accepted)
                if pdf_path:
                    extracted_text = extract_text_from_pdf(pdf_path)
                    save_text_to_file(extracted_text, paper['title'], save_dir_accepted)
    else:
        print("No accepted papers found.")

    if rejected_notes:
        rejected_paper_list = [format_note_with_reviews(note, conference_name) for note in rejected_notes]

        output_filename_rejected = f"{conference_name}_{conference_year}_rejected_papers_reviews_meta.json"
        save_to_json(rejected_paper_list, output_filename_rejected)

        if args.download_pdfs:
            save_dir_rejected = os.path.join(args.pdf_save_dir, 'rejected')
            for paper in rejected_paper_list:
                pdf_path = download_pdf_from_openreview(paper['url'].split('=')[-1], paper['title'], save_dir_rejected)
                if pdf_path:
                    extracted_text = extract_text_from_pdf(pdf_path)
                    save_text_to_file(extracted_text, paper['title'], save_dir_rejected)
    else:
        print("No rejected papers found.")
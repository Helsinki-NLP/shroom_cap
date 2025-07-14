import os
import sys
import random
import requests
import webbrowser
import pandas as pd

ROOT = '..'

assert len(sys.argv) > 1, "Missing language input argument.\nCall this script using: \n python3 capture_questions.py <your_language>"
YOUR_LANG = sys.argv[1].lower()
OUTFILE = f'{ROOT}/data/{YOUR_LANG}/questions.jsonl'

os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
db = pd.read_json(f'{ROOT}/data/papers-with-awards.jsonl', lines=True)
db['question'] = None

if os.path.isfile(OUTFILE):
    print(f'Loading {OUTFILE}')
    dbout = pd.read_json(OUTFILE, lines=True)
else:
    dbout = pd.DataFrame(columns=db.columns)

while len(dbout) <= 100:
    print(f'Recording Q {len(dbout)+1}/100')
    idx = random.randint(0, (~db.abstract.isna()).sum() - len(dbout))
    row = db[~db.title.isin(set(dbout.title)) & ~db.abstract.isna()].iloc[idx].copy()
    _ = requests.get(row.url)
    if row.url and _.status_code == 200 and not ('github' in row.url):
        rowurl = row.url
    else:
        # fallback using the paper's doi
        _ = requests.get(f'https://doi.org/{row.doi}')
        rowurl = _.history[-1].url + '.pdf'
    print(f'\nOpening article {rowurl} in your browser (rownum: {row.name})')
    webbrowser.open(rowurl)
    user_input = input(f"Enter a question IN {YOUR_LANG.upper()} about the paper ({row.title[:15]}...) that just opened in your browser: ")
    if not user_input.lower() in ['exit', 'skip', 'next']:
        row.question = user_input
        row.url = rowurl
        row_df = pd.DataFrame([row])
        dbout = pd.concat([dbout, row_df], ignore_index=True)
        dbout.to_json(OUTFILE, orient="records", lines=True)

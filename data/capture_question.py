import os
import random
import requests
import webbrowser
import pandas as pd

ROOT = '..'
LANG = 'spanish'  # change this!
OUTFILE = f'{ROOT}/data/{LANG}/questions.jsonl'

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
    idx = random.randint(0, len(db))
    row = db.iloc[idx].copy()
    _ = requests.get(row.url)
    if row.url and _.status_code == 200 and not ('github' in row.url):
        rowurl = row.url
    else:
        # fallback using the paper's doi
        _ = requests.get(f'https://doi.org/{row.doi}')
        rowurl = _.history[-1].url + '.pdf'
    print(f'Opening article {rowurl} in your browser')
    webbrowser.open(rowurl)
    user_input = input(f"Enter a question in {LANG} about the paper that just opened in your browser: ")
    if not user_input.lower() in ['exit', 'skip', 'next']:
        row.question = user_input
        row.url = rowurl
        dbout = dbout.append(row)
        dbout.to_json(OUTFILE, orient="records", lines=True)

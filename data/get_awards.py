import pathlib

from bs4 import BeautifulSoup
import pandas as pd
import tqdm


def try_get(obj, child):
	return obj.find(child).text if obj.find(child) else None

def maybe_pdf_url(obj):
	maybe_url = try_get(obj, 'url')
	if maybe_url and not 'http' in maybe_url:
		return f'https://aclanthology.org/{maybe_url}.pdf', True
	elif 'http' in maybe_url:
		return maybe_url, False
	else:
		return None, False

records = []
files = list(pathlib.Path('acl-anthology/data/xml').glob('*.xml'))
for file in tqdm.tqdm(files):
	with open(file, 'r') as istr:
		soup = BeautifulSoup(istr.read(), 'xml')
	for paper in soup.find_all('paper'):
		if paper.find('award'):
			url, extracted = maybe_pdf_url(paper)
			records.append({
				'title': try_get(paper, 'title'),
				'abstract': try_get(paper, 'abstract'),
				'doi': try_get(paper, 'doi'),
				'url': url,
				'extracted': extracted,
				'datafile': file.name,
				'authors': [
					{
						'first': try_get(author, 'first'),
						'last': try_get(author, 'last'),
					}
					for author in paper.find_all('author')
				],
			})

df = pd.DataFrame.from_records(records)
df.to_json('papers-with-awards.jsonl', lines=True, orient='records')

import pathlib

from bs4 import BeautifulSoup
import pandas as pd
import tqdm


def try_get(obj, child):
	return obj.find(child).text if obj.find(child) else None

records = []
files = list(pathlib.Path('acl-anthology/data/xml').glob('*.xml'))
for file in tqdm.tqdm(files):
	with open(file, 'r') as istr:
		soup = BeautifulSoup(istr.read(), 'xml')
	for paper in soup.find_all('paper'):
		if paper.find('award'):
			records.append({
				'title': try_get(paper, 'title'),
				'abstract': try_get(paper, 'abstract'),
				'doi': try_get(paper, 'doi'),
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

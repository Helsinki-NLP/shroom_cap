import sys
import random
import pathlib
import argparse
import pandas as pd


def askquestion(row, printfull=True):
    if printfull:
        print("QUESTION:   " + row.prompt)
        print("url:   " + row['url'])
        print("LLM-OUTPUT: \n\n" + row.output_text + "<END_OF_LLM_OUTPUT> \n\n")

    userans = input(""" Does this output contains fluency mistakes? (options: y/n/minor)
                    Use 'minor' if the output contains up to 3 minor mistakes like: orthography mistake, missing or misusing punctuation.\n>""")
    userans = userans.strip().lower()

    while userans not in ['y', 'n', 'yes', 'no', 'none', 'm', 'minor', 'end', 'save']:
        print('''ERROR only accepted answers are: y, n, m, yes, no, minor, Y, N, M, YES, NO, MINOR, END, end, End''')
        userans = askquestion(row, printfull=False)

    userans2 = input(""" Does this output contains hallucinations? (options: y/n).\n>""")
    userans2 = userans2.strip().lower()

    while userans2 not in ['y', 'n', 'yes', 'no', 'none', 'end', 'save']:
        print('''ERROR only accepted answers are: y, n, yes, no, Y, N, M, YES, NO, END, end, End''')
        userans2 = askquestion(row, printfull=False)

    return userans, userans2


def saveprogress(finaldb, outfile):
    finaldb.to_json(outfile, orient="records", lines=True)


ROOT = '../'


def check_condition(db, i):
    # CHECK IF THE OUTPUT HAS BEEN ANNOTATED ALREADY
    C1 = db.has_fluency_mistakes.isna().loc[i]
    if db.model_id.unique().size == 2:
        # MAKE SURE THAT WE SAMPLE TWO OUTPUTS PER PROMPT, MODEL AND QUESTION (2^3=8  annotations per question):
        current_db = db[(db.question == db.iloc[i].question) & (db.prompt == db.iloc[i].prompt) & (db.model_id == db.iloc[i].model_id)]  # 3 rows: 3 configs
        C2 = current_db.has_fluency_mistakes.notna().sum() < 2
        C3 = current_db.has_factual_mistakes.notna().sum() < 2
    elif db.model_id.unique().size == 3:
        # MAKE SURE THAT WE SAMPLE TWO OUTPUTS PER PROMPT ADN QUESTION, THE MODEL BECOMES RANDOM
        # to have 8 annotatations per question we need to annotate 4 entries from current_db
        current_db = db[(db.question == db.iloc[i].question) & (db.prompt == db.iloc[i].prompt)]  # 9 rows <- 3models*3configs
        C2 = current_db.has_fluency_mistakes.notna().sum() < 4
        C3 = current_db.has_factual_mistakes.notna().sum() < 4
    else:
        raise ValueError('You needed to prompt 2 or 3 models')

    # annotate if all three conditions are True
    return C1 and C2 and C3


def main(args):
    language = args.language.lower()
    R_FILE = f'{ROOT}/data/{language}/{args.split}_generated_answers.jsonl'
    records = pd.read_json(R_FILE, orient='records', lines=True)

    outfile = f'{ROOT}/data/{language}/{args.split}_annotated_data.jsonl'
    if not pathlib.Path(outfile).is_file():
        finaldb = records.reset_index()
        finaldb['has_fluency_mistakes'] = None
        finaldb['has_factual_mistakes'] = None
    else:
        finaldb = pd.read_json(outfile, orient='records', lines=True)
        saveprogress(finaldb, outfile + '.backup')
        print(f'INFO: Found started file: {outfile}.')

    qnum = finaldb.has_fluency_mistakes.notna().sum() + 1
    print(f'INFO: Starting form question number {qnum} of {len(finaldb)}\n')

    # group by article and shuffle within each group
    grouped = finaldb.groupby(['question'])
    shuffled_indices = []

    for _, group in grouped:
        indices = list(group.index)
        random.shuffle(indices)  # shuffle within the article
        shuffled_indices.extend(indices)

    # random_ids = random.sample(range(0, len(finaldb.index)), len(finaldb.index))
    for i in shuffled_indices:
        if check_condition(finaldb, i):
            userans = askquestion(finaldb.loc[i])
            finaldb.loc[i, 'has_fluency_mistakes'] = user2ans[userans[0]]
            finaldb.loc[i, 'has_factual_mistakes'] = user2ans[userans[1]]
            saveprogress(finaldb, outfile)
    print(f'You are done! By now you have annotated 8 outputs per question. If the number of annotations is different than 8 per quesiton, please report it to the organizers (there might be a bug needing FIXES... sorry if this is the case :O) ')
    saveprogress(finaldb, outfile)


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str,
                        help="Lanuage you will be checking for fluency",
                        default='English',
                        )
    parser.add_argument('--split', type=str,
                        help="Split you are annotating. Options: valid, train, test",
                        default='train',
                        )
    args = parser.parse_args()
    return args


global user2ans
user2ans = {'yes': 'y', 'no': 'n', 'none': 'n', 'minor': 'm', 'y': 'y', 'n': 'n', 'm': 'm'}


if __name__ == '__main__':
    args = parse_options()
    main(args)


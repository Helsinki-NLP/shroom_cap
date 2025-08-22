# SHROOM-CAP - data folder
Most probably you are here because you are in charge of creating and annotating the datapoints for a language, to extend the SHROOM-CAP shared-task dataset.

With SHROOM-CAP, we want to promote resereach on Scientific Cross-lingual Hallucination Detection. For this, we use award-winning ACL papers that were published before the cut off of (most) SOTA LLMs. 

# What you'll find here:
- `capture_question.py`: script to capture questions, intended to make your life easier. 

    _NOTE_: call it giving in the name of the language you are working with (we refer to it as `<your_language>` from now on) as the first argument (e.g., `capture_questions.py spanish`). This will create a folder with the languge and start filling in the questions. 

    _NOTE 2_: using the script is optional. If you don't use it, make sure that the `./<your_language>/questions.jsonl` file that you create, has the same structure (check the une in the `./english/` ).
- `papers-with-awards.jsonl`: contains the list of ACL anthology papers to serve as reference data
- `get_awards.py`: scripts used to generate the reference data list (i.e., `papers-with-awards.jsonl`)

# How to Proceed:
You are the expert of the language you are in charge of.
1. Create 100 questions about the papers in `papers-with-awards.jsonl`, following the guidelines below:

    - **Write the questions without the prompt:** The questions you write, are to be added to (one or more) prompt tamplates in the following steps. Hence, the questions do not have to contain the name of the work, or the authors. Just ask the questions as if the context was given.

    - **IMPORTANT:** Consider that the prompt templates will be of the form: _"In the study/article < title > by < authors >, < question >"_, so your questions should fit this template.

2. Use the script `prompt_models.py` to prompt 2 different LLMs with your the questions you just made (stored in `./<your_language>/questions.jsonl`). Use the following steps:

    - Modify the script `prompt_models.py` to use the LLM of your choice: The LLM should would deliver _good_ results in <your_language>.
        - Modify the dictionary `MODELS` to include two new models associated to your language: `{'<your_language>': ["<huggingface_model_1>", "<huggingfacE_model_2>"}`
        - Modify the dictionary `PROMPT_TEMPLATES` to include a new template written in your language. `{'<your_language>': {'prefix': "In the article titled \"{title}\" by {last},{first} {aux}, ", 'abstract': "Here is the article abstract for your reference: {abstract}"    },'}` <- basically, just translate these to `<your_language>`.

    - If the models you use do NOT follow the template: `message = [{"role": "user", "content": prompt}]`, modify the script accordingly.
    - The script will produce outputs using several sampling configurations, and store them in `./<your_language>/generated_answers.jsonl`

3. We will annotate all the 12xnum_questions (12 comes from 2 models, 2 prompts, 3 sampling configurations) outputs with two binary annotations, using the script `annotate_fluency_hallucs.py`. For each output, the script will show you the prompt, the response text, and will open the article in your browser. Use that to answer the two questions: 
    - are the outputs fluent / understandable? (code-switching is okay, as long as it's between <your_language> and English) 
    - does the output contain a hallucination?

4. The scripts produce a file containing the finalized datasets, language experts double check the quality.

5. Post-processing: use the script `update_index.py` to update the index format in validation data (it has to be e.g. `en-val-0`).


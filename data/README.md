# SHROOM-CAP - data folder
Most probably you are here because you are in charge of creating and annotating the datapoints for a language, to extend the SHROOM-CAP shared-task dataset.

With SHROOM-CAP, we want to promote resereach on Scientific Cross-lingual Hallucination Detection. For this, we use award-winning ACL papers that were published before the cut off of (most) SOTA LLMs. 

# What you'll find here:
- `capture_question.py`: script to capture questions, intended to make your life easier. 

    _NOTE_: call it giving in the name of the language you are working with (we refoer to it as <your_language> from now on) as the first argument (e.g., `capture_questions.py spanish`). This will create a folder with the languge and start filling in the questions. 

    _NOTE 2_: using the script is optional. If you don't use it, make sure that the `./<your_language>/questions.jsonl` file that you create, has the same structure (check the une in the `./english/` ).
- `papers-with-awards.jsonl`: contains the list of ACL anthology papers to serve as reference data
- `get_awards.py`: scripts used to generate the reference data list (i.e., `papers-with-awards.jsonl`)

# How to Proceed:
You are the expert of the language you are in charge of.
1. Create 100 questions about the papers in `papers-with-awards.jsonl`, following the guidelines below:

    - **Write the questions without the prompt:** The questions you write, are to be added to (one or more) prompt tamplates in the following steps. Hence, the questions do not have to contain the name of the work, or the authors. Just ask the questions as if the context was given.

    - **IMPORTANT:** Consider that the prompt templates will be of the form: _"In the study/article < title > by < authors >, < question >"_, so your questions should fit this template.

2. Use the script <PLACEHOLDER> to prompt 2-3 LLMs with your the questions you just made (stored in `./<your_language>/questions.jsonl`). Use the following steps:

    - Modify the script to use the LLM of your choice: The LLM should would deliver _good_ results in <your_language>.
    
    - The script will produce outputs using several sampling configurations, and store them in `./<your_language>/generated_answers.jsonl`

3. Out of all the sampled outputs per question,
select 1 answer that have a hallucination
select 1 answer that doesn’t have a hallucination

4. The scripts produce a file containing the finalized datasets, language experts double check the quality


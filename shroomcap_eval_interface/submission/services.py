import datetime
import json
import pandas as pd

from django.conf import settings
from django.db.models import Max, F

from . import models
from . import scorer

def can_submit():
    return datetime.datetime.now() < settings.TEST_PHASE_END_DATE
    
def can_display():
    return datetime.datetime.now() < settings.LEADERBOARD_DISPLAY_DATE

def get_profile(user):
    profile = (
        models
        .Profile
        .objects
        .get(participant__username=user.username)
    )
    return profile

def get_rankings(_force_val=False):
    if can_display():
        return None
        
    split = 'VAL' if _force_val else 'TST'
    submissions = models.Submission.objects.filter(split=split).values(
        'submitter__team_name', 
        'language',
        'flue_score',
        'fact_score',
    )
    df_rankings = pd.DataFrame.from_records(submissions)
    groupings = ['submitter__team_name', 'language']
    best_subs = df_rankings.groupby(groupings)['fact_score'].max().reset_index()
    df_rankings = df_rankings.merge(best_subs)
    df_rankings = df_rankings.sort_values(by=['fact_score', 'flue_score'], ascending=False)
    df_rankings = df_rankings.drop_duplicates(groupings)
    return df_rankings

def get_ref_data(split, lang):
    refs = (
        models
        .DataPoint
        .objects
        .filter(language=lang, split=split, is_ref=True)
        .order_by('datapoint_id')
    )
    return refs

def handle_valid_file(pred_dicts, form_dict, profile):
    lang, split, _ = pred_dicts[0]['datapoint_id'].split('-')
    split, lang = split.upper(), lang.upper()
    ref_data = get_ref_data(split, lang)
    true_factual = list(ref_data.values_list('factual', flat=True))
    pred_factual = [pdict['factual'] for pdict in pred_dicts]
    true_fluency = list(ref_data.values_list('fluent', flat=True))
    pred_fluency = [pdict['fluent'] for pdict in pred_dicts]
    scores = scorer.main(true_factual, true_fluency, pred_factual, pred_fluency)
    fact_score = scores['f1_factual_macro']
    flue_score = scores['f1_fluency_macro']
    
    submission_inst = models.Submission.objects.create(
        identifier=form_dict['identifier'],
        language=lang,
        split=split,
        submitter=profile,
        is_prompt=form_dict['is_prompt'],
        is_rag=form_dict['is_rag'],
        system_description=form_dict['system_description'],
        dataset_description=form_dict['dataset_description'],
        plms_description=form_dict['plms_description'],
        extra_description=form_dict['extra_description'],
        fact_score=fact_score,
        flue_score=flue_score,
    )
    models.DataPoint.objects.bulk_create([
        models.DataPoint(
            submission=submission_inst,
            split=pred_dict["split"],
            language=pred_dict["lang"],
            datapoint_id=pred_dict['datapoint_id'],
            factual=pred_dict['factual'],
            fluent=pred_dict['fluent'],
            is_ref=False,
        )
        for pred_dict in pred_dicts
    ])

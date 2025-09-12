import json
import os
from sklearn.metrics import f1_score


def main(true_factual, true_fluency, pred_factual, pred_fluency):
    # Compute macro F1 scores
    f1_factual = f1_score(true_factual, pred_factual, average='macro')
    f1_fluency = f1_score(true_fluency, pred_fluency, average='macro')

    # Load metadata
    # with open(os.path.join(prediction_dir, 'metadata.json')) as f:
    #    duration = json.load(f).get('duration', -1)

    # Final scores
    scores = {
        'f1_factual_macro': f1_factual,
        'f1_fluency_macro': f1_fluency,
        # 'duration': duration
    }
    return scores

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-reference_dir', type=str, default=os.path.join('/app/input/', 'ref'))
    p.add_argument('-prediction_dir', type=str, default=os.path.join('/app/input/', 'res'))
    p.add_argument('-score_dir', type=str, default='/app/output/')
    args = p.parse_args()
    reference_dir = args.reference_dir
    prediction_dir = args.prediction_dir
    score_dir = args.score_dir

    # Load gold labels
    print('Reading gold labels...')
    with open(os.path.join(reference_dir, 'label.json')) as f:
        truth_lines = [json.loads(line) for line in f]

    # Load predictions
    print('Reading predictions...')
    with open(os.path.join(prediction_dir, 'prediction.json')) as f:
        pred_lines = [json.loads(line) for line in f]

    # Sort both lists by 'index' to ensure matching order
    truth_lines.sort(key=lambda x: x["index"])
    pred_lines.sort(key=lambda x: x["index"])

    # Convert "y"/"n" to binary labels
    def yn_to_binary(label):
        return 1 if label == "y" else 0

    # Extract fields
    true_factual = [yn_to_binary(x["has_factual_mistakes"]) for x in truth_lines]
    pred_factual = [yn_to_binary(x["has_factual_mistakes"]) for x in pred_lines]

    true_fluency = [yn_to_binary(x["has_fluency_mistakes"]) for x in truth_lines]
    pred_fluency = [yn_to_binary(x["has_fluency_mistakes"]) for x in pred_lines]

    scores = main(true_factual, true_fluency, pred_factual, pred_fluency)

    print('Scores:', scores)

    # Save to output
    with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
        json.dump(scores, score_file)

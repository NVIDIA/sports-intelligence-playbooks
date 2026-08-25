import argparse
import os

from avlm.inference.common.utils.json_io import load_json, write_json

def get_median(scores):
    if not scores:
        return 0

    scores = sorted(scores)
    mid = len(scores) // 2
    if len(scores) % 2 == 1:
        return scores[mid]

    return (scores[mid - 1] + scores[mid]) / 2

def get_score_distribution(scores):
    return {
        "excellent_9_10": len([score for score in scores if score >= 9]),
        "good_8": len([score for score in scores if score == 8]),
        "fair_7": len([score for score in scores if score == 7]),
        "poor_6": len([score for score in scores if score == 6]),
        "very_poor_1_5": len([score for score in scores if 1 <= score <= 5]),
    }

def parse_results(judge_predictions):
    scores = []
    num_failed = 0

    for pred in judge_predictions:
        score = int(pred.get("score", 0))
        if pred.get("error") or score == 0:
            num_failed += 1
        else:
            scores.append(score)

    results_dict = {
        "total_judged": len(judge_predictions),
        "num_success": len(scores),
        "num_failed": num_failed,
        "mean_score": sum(scores) / len(scores) if scores else 0,
        "median_score": get_median(scores),
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "score_distribution": get_score_distribution(scores),
    }

    return results_dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-dir", required=True)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    inference_dir = args.inference_dir
    llm_judge_dir = os.path.join(inference_dir, 'llm_judge')
    llm_judge_pred_path = os.path.join(llm_judge_dir, 'llm_judge_predictions.json')

    if not os.path.exists(llm_judge_pred_path):
        raise FileNotFoundError(f'Pred path does not exist: {llm_judge_pred_path}')

    judge_predictions = load_json(llm_judge_pred_path)["results"]

    results_dict = parse_results(judge_predictions)
    results_dict["per_class"] = {}

    question_class_predictions = {}
    for pred in judge_predictions:
        question_class = pred["class"]
        if not question_class in question_class_predictions:
            question_class_predictions[question_class] = []
        question_class_predictions[question_class].append(pred)

    for question_class, question_class_preds in question_class_predictions.items():
        results_dict["per_class"][question_class] = parse_results(question_class_preds)

    results_path = os.path.join(llm_judge_dir, 'llm_judge_results.json')
    write_json(results_path, results_dict)

if __name__ == "__main__":
    main()

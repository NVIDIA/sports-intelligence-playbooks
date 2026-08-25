import argparse
import os
import re
import warnings

from avlm.inference.common.utils.json_io import load_jsonl, write_json

def parse_mcq_answer(text):
    match = re.match(r"^\s*(?:\*\*)?\(([a-zA-Z])\)(?:\*\*)?(?=\s|$)", text)
    return (match.group(1)).lower() if match else None

def parse_results(num_infer_failed,
                  num_parse_failed,
                  num_correct,
                  num_incorrect,
                  ):
    total_questions = (num_infer_failed + num_parse_failed + 
                       num_correct + num_incorrect)

    results_dict = {
        'total_mcq_questions': total_questions,
        'num_infer_failed': num_infer_failed,
        'num_parse_failed': num_parse_failed,
        'num_correct': num_correct,
        'num_incorrect': num_incorrect,
        'pct_correct': num_correct / total_questions if total_questions else 0.0
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
    pred_path = os.path.join(inference_dir, 'predictions.jsonl')

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f'Pred path does not exist: {pred_path}')

    mcq_dir = os.path.join(inference_dir, 'mcq')
    os.makedirs(mcq_dir, exist_ok=True)

    predictions = load_jsonl(pred_path)

    num_infer_failed = 0
    num_parse_failed = 0
    num_correct = 0
    num_incorrect = 0
    question_class_stats = {}
    for pred in predictions:
        evaluation_type = pred.get('evaluation_type')
        if evaluation_type is None:
            warnings.warn(
                f"Missing evaluation_type for prediction: {pred.get('id')}"
            )

        if evaluation_type != 'mcq':
            continue

        question_class = pred['class']
        if not question_class in question_class_stats:
            question_class_stats[question_class] = {
                'num_infer_failed': 0,
                'num_parse_failed': 0,
                'num_correct': 0,
                'num_incorrect': 0,
            }

        inference_status = pred['inference_status']
        # not currently possible with automodel inference 
        # but including in case we expand
        if inference_status != 'success':
            num_infer_failed += 1
            question_class_stats[question_class]['num_infer_failed'] += 1
            continue

        gt_answer = parse_mcq_answer(pred['ground_truth'])
        pred_answer = parse_mcq_answer(pred['prediction'])
        
        if gt_answer is None:
            warnings.warn(f"Invalid gt answer. Skipping. Index: {pred['index']}")
            continue

        if pred_answer is None:
            num_parse_failed += 1
            question_class_stats[question_class]['num_parse_failed'] += 1
            continue

        if gt_answer == pred_answer:
            num_correct += 1
            question_class_stats[question_class]['num_correct'] += 1
        else:
            num_incorrect += 1
            question_class_stats[question_class]['num_incorrect'] += 1
    
    results_dict = parse_results(num_infer_failed, num_parse_failed,
                                 num_correct, num_incorrect)

    results_dict['per_class'] = {}
    for question_class, question_res in question_class_stats.items():
        results_dict['per_class'][question_class] = parse_results(
                                                        question_res['num_infer_failed'],
                                                        question_res['num_parse_failed'],
                                                        question_res['num_correct'],
                                                        question_res['num_incorrect'],
                                                    )

    results_path = os.path.join(mcq_dir, 'mcq_results.json')
    write_json(results_path, results_dict)

if __name__ == "__main__":
    main()

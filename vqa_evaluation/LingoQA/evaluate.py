import click
import torch
import pandas as pd

from datasets import Dataset
from functools import partial 
from constants import LINGOQA_TEST, Keys
from judge import LingoJudge


@click.command()
@click.option('--predictions_path', help='Path to predictions file.')
@click.option('--batch_size', help='Batch size for evaluation.', default=1)
def evaluate(predictions_path: str, batch_size: int) -> float:
    """
    Simple script for running evaluation on the LingoQA benchmark.

    Args:
        predictions_path: path to a .csv file containing the model predictions.
        batch_size: batch size for evaluation.
    Out:
        benchmark_score: evaluation score obtained from running the textual classifier on the benchmark.
    """
    references = pd.read_parquet(LINGOQA_TEST)
    references = references[[Keys.question_id.value, Keys.segment_id.value, Keys.question.value, Keys.answer.value]]
    references = references.groupby([Keys.question_id.value, Keys.segment_id.value, Keys.question.value]).agg(list)
    references = references.rename({Keys.answer.value: Keys.references.value}, axis=1)
    print(f"Loaded {len(references)} references.")

    predictions = pd.read_csv(predictions_path)
    predictions = predictions.rename({Keys.answer.value: Keys.prediction.value}, axis=1)
    print(f"Loaded {len(predictions)} predictions.")

    merged = pd.merge(predictions, references, on=[Keys.question_id.value, Keys.segment_id.value])
    print(f"Matched {len(merged)} predictions with references.")
    if len(merged) != 500:
        print("WARNING! You are evaluating on a subset of the LingoQA benchmark. Please check your input file for missing or mis-matched examples.")

    # Debug: verify expected columns exist after merge (helps diagnose KeyError in HF datasets map)
    print("[Debug] merged columns:", list(merged.columns))
    if Keys.references.value not in merged.columns:
        ref_like = [c for c in merged.columns if "reference" in str(c).lower()]
        print(f"[Debug] missing expected column {Keys.references.value!r}. reference-like columns: {ref_like}")
    if Keys.prediction.value not in merged.columns:
        pred_like = [c for c in merged.columns if "pred" in str(c).lower()]
        print(f"[Debug] missing expected column {Keys.prediction.value!r}. pred-like columns: {pred_like}")

    dataset = Dataset.from_pandas(merged)
    print("[Debug] dataset features:", dataset.features)

    judge = LingoJudge().eval().to("cuda:0")
    dataset_evaluated = dataset.map(partial(evaluate_question, judge), batched=True, batch_size=batch_size)
    dataset_filtered = dataset_evaluated.filter(select_correct)

    benchmark_score = dataset_filtered.num_rows/dataset_evaluated.num_rows
    print(f"The overall benchmark score is {benchmark_score*100}%")
    return benchmark_score


def evaluate_question(metric: LingoJudge, data_dict: dict) -> dict:
    """
    Run evaluation for a batch of questions.

    Args:
        metric: the evaluation metric for computing the scores.
        data_dict: the data dictionary containing questions, references, and predictions.

    Out:
        data_dict: updated data dictionary containing information such as
        the maximum score, the probability of correctness, and a boolean
        indicating whether the prediction is correct or not.
    """
    if (
        Keys.references.value not in data_dict
        or Keys.prediction.value not in data_dict
        or Keys.question.value not in data_dict
    ):
        print("[Debug] batch keys:", list(data_dict.keys()))

    questions = data_dict[Keys.question.value]
    references = data_dict[Keys.references.value]
    prediction = data_dict[Keys.prediction.value]

    scores = metric.compute(questions, references, prediction)

    data_dict[Keys.score.value] = scores
    data_dict[Keys.probability.value] = torch.sigmoid(scores)
    data_dict[Keys.correct.value] = scores > 0.0
    return data_dict


def select_correct(data_dict: dict) -> bool:
    """
    Filtering function for selecting the predictions classified as correct.
    """
    return data_dict[Keys.correct.value]


if __name__=="__main__":
    _ = evaluate()
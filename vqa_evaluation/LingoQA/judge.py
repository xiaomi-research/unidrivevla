import torch
from torch import nn

from tqdm import trange
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from constants import LINGO_JUDGE, Keys


class LingoJudge(nn.Module):
    """
    LingoJudge is a textual classifier that evaluates the truthfulness of an answer on the LingoQA benchmark.
    """
    def __init__(self, pretrained_model=LINGO_JUDGE):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model).eval()

    @torch.inference_mode()
    def forward(self, question: str, references: List[str], prediction: str):
        """
        Inference function for textual classifier with multiple reference answers.
        Args:
            question: Input question.
            references: List of references.
            prediction: Model prediction.
        Output:
            scores: Score indicating truthfulness.
        """
        device = next(self.parameters()).device
        texts = [
            f"{self.tokenizer.cls_token}\nQuestion: {question}\nAnswer: {a_gt}\nStudent: {prediction}"
            for a_gt in references
        ]

        encoded_input = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        output = self.model(**encoded_input)
        scores = output.logits.squeeze(-1)
        return scores

    def compute(self, questions: List[str], references: List[List[str]], predictions: List[str]):
        """
        Compute maximum classifier metric. For multiple reference answers, selects the highest one.
        Args:
            questions: List of input questions.
            references: List of lists, with multiple references per question supported.
            predictions: List of model predictions.
        Output:
            scores: Score indicating truthfulness. 
        """
        max_scores = []

        for index, question in enumerate(questions):
            references_preprocessed = [self.preprocess(reference) for reference in references[index]]
            prediction_preprocessed = self.preprocess(predictions[index])
            scores = self.forward(question, references_preprocessed, prediction_preprocessed)
            max_score = [max(scores)]
            max_scores.extend(max_score)
        return torch.Tensor(max_scores)
        
    def preprocess(self, string: str):
        """
        Preprocessing function for consistency. 
        Args:
            string: input string to be processed.
        Output:
            output: processed string with lower cases and trailing lines removed.
        """
        output = str(string).lower().lstrip().rstrip() 
        return output
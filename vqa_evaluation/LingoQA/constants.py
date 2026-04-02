"""
LingoQA datasets are stored in Google Cloud.
This file provides the download link for the datasets, as well as reference keys for the data.
"""
from enum import Enum

#LINGOQA_TEST = "https://drive.usercontent.google.com/u/1/uc?id=1I8u6uYysQUstoVYZapyRQkXmOwr-AG3d&export=download"
LINGOQA_TEST = "val.parquet"

#LINGO_JUDGE = "wayveai/Lingo-Judge"
LINGO_JUDGE = "wayveai/Lingo-Judge"

class Keys(str, Enum):
    question_id = "question_id"
    segment_id = "segment_id"
    question = "question"
    answer = "answer"
    references = "references"
    prediction = "prediction"
    max_score = "max_score"
    score = "score"
    probability = "probability"
    correct = "correct"
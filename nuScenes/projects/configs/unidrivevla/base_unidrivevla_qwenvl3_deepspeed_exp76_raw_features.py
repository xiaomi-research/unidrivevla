_base_ = ['./base_unidrivevla_qwenvl3_deepspeed.py']

model = dict(
    planning_head=dict(
        feature_source="raw"
    )
)

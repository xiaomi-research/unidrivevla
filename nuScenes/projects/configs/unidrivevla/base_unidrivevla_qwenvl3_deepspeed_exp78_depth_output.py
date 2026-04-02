_base_ = ['./base_unidrivevla_qwenvl3_deepspeed.py']

model = dict(
    planning_head=dict(
        with_depth_supervision=True,
        depth_supervision_source="output"
    )
)

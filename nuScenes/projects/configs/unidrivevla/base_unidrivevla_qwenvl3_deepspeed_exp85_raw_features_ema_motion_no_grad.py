_base_ = ['./base_unidrivevla_qwenvl3_deepspeed.py']

model = dict(
    planning_head=dict(
        feature_source="raw"
    )
)
custom_hooks = [
    dict(type='EMAHook', momentum=0.0002, interval=1, warm_up=2000, priority='VERY_HIGH')
]
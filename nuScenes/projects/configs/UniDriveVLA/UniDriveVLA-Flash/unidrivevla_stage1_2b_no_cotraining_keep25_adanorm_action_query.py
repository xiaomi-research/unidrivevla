_base_ = ["./unidrivevla_stage1_2b_no_cotraining_flowmatching_keep25.py"]

model = dict(
    planning_head=dict(
        action_training_mode="query_supervised",
        action_query_cfg=dict(
            num_queries=8,
            l1_loss_weight=0.0,
            l2_loss_weight=1.0,
        ),
        flow_distill_cfg=dict(enabled=False),
        loss_planning=None,
    )
)

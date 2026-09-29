from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class WASPAccumulationHook(Hook):
    """Compute batch-independent WASP regularization once per optimizer step."""

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self._planning_head = None

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, "module") else model

    def before_run(self, runner):
        model = self._unwrap_model(runner.model)
        planning_head = getattr(model, "planning_head", None)
        if planning_head is None:
            raise RuntimeError(
                "WASPAccumulationHook requires model.planning_head."
            )
        if not hasattr(planning_head, "set_wasp_accumulation_state"):
            raise RuntimeError(
                "planning_head does not support WASP accumulation control."
            )
        self._planning_head = planning_head
        self._planning_head.set_wasp_accumulation_state(True, 1)

    def before_train_iter(self, runner):
        engine = runner.model
        compute = True
        scale = 1
        if (
            self.enabled
            and hasattr(engine, "is_gradient_accumulation_boundary")
            and hasattr(engine, "gradient_accumulation_steps")
        ):
            compute = bool(engine.is_gradient_accumulation_boundary())
            scale = (
                int(engine.gradient_accumulation_steps()) if compute else 0
            )
        self._planning_head.set_wasp_accumulation_state(
            compute,
            scale,
        )

# ============================================================================
#  This module is part of the CaRL autonomous driving agent.
#  Original source: https://github.com/autonomousvision/carl
#  All credit for the original implementation goes to the CaRL authors.
# ============================================================================
"""
Base class for observation managers.
Code adapted from https://github.com/zhejz/carla-roach
"""


class ObsManagerBase:
    def __init__(self):
        pass

    def attach_ego_vehicle(self, parent_actor):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def clean(self):
        raise NotImplementedError

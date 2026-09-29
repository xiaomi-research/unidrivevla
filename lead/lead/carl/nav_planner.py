# ============================================================================
#  This module is part of the CaRL autonomous driving agent.
#  Original source: https://github.com/autonomousvision/carl
#  All credit for the original implementation goes to the CaRL authors.
# ============================================================================
"""
Route planner that tracks progress along a predefined route.
"""

import carla


class RoutePlanner:
    def __init__(self):
        self.route = []
        self.windows_size = 2

    def set_route(self, global_plan):
        self.route = global_plan
        self.index = 0
        self.route_length = len(self.route)

    def run_step(self, gps):
        location = carla.Location(x=gps[0], y=gps[1])

        for index in range(
            self.index,
            min(self.index + self.windows_size + 1, self.route_length),
        ):
            route_transform = self.route[index]
            route_location = route_transform[0].location
            wp_dir = route_transform[0].get_forward_vector()
            wp_veh = location - route_location

            if wp_veh.dot(wp_dir) > 0:
                self.index = index

        output = self.route[self.index : self.route_length]
        return output

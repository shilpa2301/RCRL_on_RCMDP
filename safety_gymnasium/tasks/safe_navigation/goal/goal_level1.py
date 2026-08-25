# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Goal level 1."""

from safety_gymnasium.assets.free_geoms import Vases
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.assets.geoms import Sigwalls
from safety_gymnasium.assets.geoms import Pillars
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0


class GoalLevel1(GoalLevel0):
    """An agent must navigate to a goal while avoiding hazards.

    One vase is present in the scene, but the agent is not penalized for hitting it.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-1.125, -1.125, 1.125, 1.125]

        # self._add_geoms(Pillars(num=5, keepout=0.1, locations=[(0, 0), (-1, -1), (-0.8, 0.6), (0.9, -1.1), (0.8, 0.9)]))  # pylint: disable=no-member
        self._add_geoms(Pillars(num=4, keepout=0.1, locations=[(-1, -1), (-0.8, 0.6), (0.9, -1.1), (0.8, 0.9)]))  # pylint: disable=no-member
        # self._add_free_geoms(Vases(num=1, is_constrained=True))

        self._add_geoms(Sigwalls(num=4, locate_factor=1.5, is_constrained=True, is_lidar_observed=True))  # pylint: disable=no-member

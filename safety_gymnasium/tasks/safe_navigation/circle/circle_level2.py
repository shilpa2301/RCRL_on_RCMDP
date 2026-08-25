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
"""Circle level2."""

from safety_gymnasium.tasks.safe_navigation.circle.circle_level1 import CircleLevel1
# from safety_gymnasium.assets.geoms import Pillars

class CircleLevel2(CircleLevel1):
    """An agent want to loop around the boundary of circle, while avoid going outside the stricter boundaries."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        # self.agent.placements = [(-0.6, -0.6, 0.6, 0.6)]
        # self.agent.keepout = 0.1

        self.sigwalls.num = 4  # pylint: disable=no-member
        # placements: list of [xmin, ymin, xmax, ymax] rectangles
        # self._add_geoms(Pillars(num=1, is_constrained=True, placements=[[-1.25, -1.25, 1.25, 1.25]]))
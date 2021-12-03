# https://bark-simulator.github.io/tutorials/
# BARK Python imports
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.scenario.scenario_generation.config_with_ease import \
  LaneCorridorConfig, ConfigWithEase
from bark.runtime.runtime import Runtime
from bark.examples.paths import Data

# BARK c++ imports
from bark.core.world.opendrive import *
from bark.core.world.goal_definition import *
from bark.core.models.behavior import *

# parameters can also be set using JSON files
# param_server = ParameterServer(filename="config.json")
param_server = ParameterServer()
# this sets a default value
param_server["BehaviorIDMClassic"]["DesiredVelocity", "Default velocity", 8.]
# overwrites the default value
param_server["BehaviorIDMClassic"]["DesiredVelocity"] = 10.

class CustomLaneCorridorConfig(LaneCorridorConfig):
  def __init__(self,
               params=None,
               **kwargs):
    super(CustomLaneCorridorConfig, self).__init__(params, **kwargs)

  def goal(self, world):
    road_corr = world.map.GetRoadCorridor(
      self._road_ids, XodrDrivingDirection.forward)
    lane_corr = self._road_corridor.lane_corridors[0]
    # defines goal polygon on the left lane
    return GoalDefinitionPolygon(lane_corr.polygon)

# configure both lanes
left_lane = CustomLaneCorridorConfig(params=param_server,
                                     lane_corridor_id=0,
                                     road_ids=[0, 1],
                                     behavior_model=BehaviorMobilRuleBased(param_server),
                                     s_min=0.,
                                     s_max=50.)

# this lane has controlled_ids; ego vehicle is placed on this lane
right_lane = CustomLaneCorridorConfig(params=param_server,
                                      lane_corridor_id=1,
                                      road_ids=[0, 1],
                                      controlled_ids=True,
                                      behavior_model=BehaviorMobilRuleBased(param_server),
                                      s_min=0.,
                                      s_max=20.)


# finally, generate 3 scenarios on the merging map
scenarios = \
  ConfigWithEase(
    num_scenarios=3,
    map_file_name=Data.xodr_data("DR_DEU_Merging_MT_v01_shifted"),
    random_seed=0,
    params=param_server,
    lane_corridor_configs=[left_lane, right_lane])

viewer = MPViewer(params=param_server,
                  x_range=[-35, 35],
                  y_range=[-35, 35],
                  follow_agent_id=True)
env = Runtime(step_time=0.2,
              viewer=viewer,
              scenario_generator=scenarios,
              render=True)

# run 3 scenarios
for _ in range(0, 3):
  env.reset()
  # step scenario 90 time-steps
  for step in range(0, 90):
    env.step()
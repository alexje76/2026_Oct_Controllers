# Copyright 2022 Open Source Robotics Foundation, Inc. and Monterey Bay Aquarium Research Institute
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

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

package_name = 'mbari_wec_oct_controllers_py'

# Convert launch argument strings to the ROS parameter types expected by the controller.
PARAM_TYPES = {
    'use_sim_time': bool,
    'active_policy': str,
    'piston_soft_stop_enabled': bool,
    'stepwise_integrated_u_range': float,
    'linear_damper_scale_factor': float,
    'linear_damper_retract_factor': float,
    'nextwave_log_interval_sec': float,
    'nextwave_stale_sec': float,
}


def _as_bool(value):
    value = value.lower()
    if value in ('true', '1', 'yes', 'on'):
        return True
    if value in ('false', '0', 'no', 'off'):
        return False
    raise ValueError(f'Expected a boolean launch argument, got: {value!r}')


def _launch_node(context, config):
    overrides = {}
    for name, value_type in PARAM_TYPES.items():
        value = LaunchConfiguration(name).perform(context)
        # Blank means: leave the YAML value in effect.
        if value == '':
            continue
        overrides[name] = _as_bool(value) if value_type is bool else value_type(value)

    return [Node(
        package=package_name,
        name='oct_controllers',
        executable='oct_controllers',
        parameters=[config, overrides],  # YAML first; supplied launch args override it.
        output='screen',
    )]


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'controller.yaml',
    )

    args = [
        DeclareLaunchArgument(name, default_value='', description=f'Override {name} from YAML')
        for name in PARAM_TYPES
    ]

    return LaunchDescription(args + [OpaqueFunction(function=lambda context: _launch_node(context, config))])

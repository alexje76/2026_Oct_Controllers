#!/usr/bin/python3

# Copyright 2022 Open Source Robotics Foundation, Inc. and Monterey Bay 
# Aquarium Research Institute
# Copyright 2026 Alex Eagan
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
import threading
import random
import csv
import os
import time

from datetime import timezone, datetime
from pathlib import Path

import numpy as np

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.duration import Duration #For timing
from rclpy.parameter import Parameter

from buoy_api import Interface

## Logging
OUTPUT_DIR = os.path.expanduser("~/BuoyLogging")

class DailyCsvLogger:
    HEADER = (
        "timestamp_utc",
        "wall_epoch_seconds",
        "ros_epoch_seconds",
        "event",
        "controller",
        "previous_controller",
    )

    def __init__(self):
        self._lock = threading.Lock()
        self._directory = Path(OUTPUT_DIR)
        self._directory.mkdir(parents=True, exist_ok=True)

        self._file = None
        self._writer = None
        self._day = None
        self._last_flush = time.monotonic()
        self._open_file(datetime.now(timezone.utc))

    def _open_file(self, now):
        if self._file:
            self._file.flush()
            self._file.close()

        # Unique filename on every process restart, including restarts on
        # the same UTC day.
        stamp = now.strftime("%Y-%m-%d_%H%M%S_%fZ")
        path = self._directory / f"controller_{stamp}.csv"

        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)
        self._day = now.date()
        self._last_flush = time.monotonic()

    def _seconds_string_from_ns(self, ns):
        seconds, nanoseconds = divmod(int(ns), 1_000_000_000)
        return f"{seconds}.{nanoseconds:09d}"

    def log(
        self,
        event,
        controller,
        previous_controller="",
        ros_time_ns=None,
        flush=False,
    ):
        wall_time_ns = time.time_ns()
        wall_seconds = self._seconds_string_from_ns(wall_time_ns)
        now = datetime.fromtimestamp(
            wall_time_ns / 1_000_000_000,
            timezone.utc,
        )

        ros_seconds = (
            ""
            if ros_time_ns is None
            else self._seconds_string_from_ns(ros_time_ns)
        )

        with self._lock:
            if now.date() != self._day:
                self._open_file(now)

            self._writer.writerow((
                now.isoformat(timespec="microseconds").replace("+00:00", "Z"),
                wall_seconds,
                ros_seconds,
                event,
                controller,
                previous_controller,
            ))

            if flush or time.monotonic() - self._last_flush >= 600:
                self._file.flush()
                self._last_flush = time.monotonic()

    def close(self):
        with self._lock:
            if self._file:
                self._file.flush()
                self._file.close()
                self._file = None

class ControlPolicy(object):
    """Common for all control policies"""
    def __init__(self, logger):
        self.state = {
            "range": 0.0,
        }
        self.lock = threading.Lock()
        self._logger = logger
        #Default States for bounding
        self._piston_bounded = False

    ## Updates for Spring Callback ##
    def update_range(self, value):
        with self.lock:
            self.state["range"] = float(value) * 39.3701 #Converts from m to in

    ## Bounding for target ##
    def piston_bounding(self, target):
        with self.lock:
            bounded_target = dict(target)
            spring_bound_upper = 65 #NEED TO CHECK THAT THERE IS NOT INCHES CONVERSION NEEDED
            spring_bound_lower = 15
            if (
                self.state["range"] < spring_bound_lower or 
                self.state["range"] > spring_bound_upper
            ):
                self._logger.info(f"srping range inside piston bounding is:{self.state['range']}")
                bounded_target["Value"] = 0.0
                self._logger.info(f'Target piston bounded, Overwritten '
                                       'with 0.0 Wind Curr')
                #TODO add logging
            else:
                pass
            return bounded_target

    def update_params(self, now):
        """Placeholder update_params"""
        pass

    def reset(self):
        """Reset state when entering/leaving control mode"""
        pass

################## Control Policies ###################################
class StepwiseRandomBoundedPolicy(ControlPolicy):
    """SystemID:
       Control that provides stepwise random control inputs 
        on a 2s interval
       Is bounded by piston stroke 
    """
    def __init__(self, logger):
        super().__init__(logger)
        self._piston_bounded = True

        self._u_on = False # Control State
        self._u_range = 4.0 #Winding current amps
        self.u = 0.0 #Control Input
        self._swap_time = None
        self._swap_duration = Duration(seconds = 2.0)

        self._target = {
            "Control Knob": 'Winding Current',
            "Value": 0.0,
        }

        self.update_params(now = None)

    def update_params(self, now):
        #Setup Call
        if now is None:
            return
        
        with self.lock:
            #First calls
            if self._swap_time is None:
                self._swap_time = now
            #End of First call 

            elapsed = now - self._swap_time

            if elapsed >= self._swap_duration:
                self._swap_time = now
                if self._u_on:
                    self.u = random.uniform(-self._u_range, self._u_range)
                else:
                    self.u = 0
                self._u_on = not self._u_on

    def target(self, state, now):
        self._target["Value"] = self.u
        return self._target

    def reset(self):
        with self.lock:
           self._u_on = False
           self.u = 0.0
           self._swap_time = None


        
class StepwiseIntegratedBoundedPolicy(ControlPolicy):
    """SystemID:
       Control that provides stepwise random control inputs 
        on an integrated interval
       Is bounded by piston stroke 
    """
    def __init__(self, logger):
        super().__init__(logger)
        self._piston_bounded = True

        self._u_on = False # Control State
        self._u_range = 35
        self.u = 0.0
        self._currentseconds = 0.0
        self._currentseconds_max = 8.0
        self._swap_time = None
        self._swap_duration = Duration(seconds = 2.0)

        self._target = {
            "Control Knob": 'Winding Current',
            "Value": 0.0,
        }
        self.update_params(None)

    def update_params(self, now):
        #Setup Call
        if now is None:
            return

        with self.lock:
            #First calls
            if self._swap_time is None:
                self._swap_time = now
            #End of First call 

            if self._u_on:
                elapsed = (now - self._swap_time).nanoseconds / 1e9
                self._currentseconds = elapsed*self.u
                if abs(self._currentseconds) >= self._currentseconds_max:
                    self.u = 0.0
                    self._currentseconds = 0.0
                    self._u_on = False
                else:
                    pass
            else: #Branch for if control is currently off
                elapsed = now - self._swap_time
                if elapsed >= self._swap_duration:
                    self.u = random.uniform(-self._u_range, self._u_range)
                    self._u_on = True 
                    self._swap_time = now
                else:
                    pass

    def target(self, state, now):
        self._target["Value"] = self.u
        return self._target

    def reset(self):
        with self.lock:
            self._u_on = False
            self.u = 0.0
            self._currentseconds = 0.0
            self._swap_time = None

class FreeResponsePolicy(ControlPolicy):
    """SystemID:
       Truly free response control policy 
    """
    def __init__(self, logger):
        super().__init__(logger)
        self._piston_bounded = True

        self._target = {
            "Control Knob": 'None',
            "Value": 0.0,
        }
        self.update_params(None)

    def update_params(self, now):
        pass

    def target(self, state, now):
        return self._target

    def reset(self):
        pass


class Controller(Interface):
    """Shared buoy interface with swappable control policies"""

    def __init__(self):
        super().__init__('controller')

        self.set_parameters([
            Parameter(
                'use_sim_time',
                Parameter.Type.BOOL,
                True,
            ),
        ])

        self._policy_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.state = {}
        self._csv_logger = DailyCsvLogger()

        self.policies = {
            "stepwise_random_bounded": StepwiseRandomBoundedPolicy(self.get_logger()),
            "free_response": FreeResponsePolicy(self.get_logger()),
            "stepwise_integrated_bounded": StepwiseIntegratedBoundedPolicy(self.get_logger()),
        }
        self.active_policy_name = "free_response"

        self.set_params()

        self._log_csv(
            event="controller_started",
            controller=self.active_policy_name,
            flush=True,
        )

        self.add_on_set_parameters_callback(self.parameter_callback)
        self._minute_log_timer = self.create_timer(
            60.0,
            self._log_controller_minute,
        )

    def _log_csv(
        self,
        event,
        controller,
        previous_controller="",
        flush=False,
    ):
        # Node.get_clock() is ROS_TIME. It follows /clock when
        # use_sim_time is enabled.
        ros_time_ns = self.get_clock().now().nanoseconds

        self._csv_logger.log(
            event=event,
            controller=controller,
            previous_controller=previous_controller,
            ros_time_ns=ros_time_ns,
            flush=flush,
        )

    @property
    def active_policy(self):
            with self._policy_lock:
                return self.policies[self.active_policy_name]

    def set_params(self):
        self.declare_parameter("active_policy", "free_response")

        requested_policy = self.get_parameter("active_policy").value

        if requested_policy not in self.policies:
            raise ValueError(f"Unknown policy: {requested_policy}")

        self.active_policy_name = requested_policy

    def parameter_callback(self, params):
        for param in params:
            if param.name != "active_policy":
                continue

            if param.value not in self.policies:
                return SetParametersResult(
                    successful=False,
                    reason=f"Unknown policy: {param.value}",
                )

            swapped = None

            with self._policy_lock:
                old_name = self.active_policy_name

                if old_name != param.value:
                    self.policies[old_name].reset()
                    self.policies[param.value].reset()
                    self.active_policy_name = param.value
                    swapped = (old_name, param.value)

            if swapped:
                old_name, new_name = swapped

                self.get_logger().info(
                    f"Switched policy from {old_name} -> {new_name}"
                )

                self._log_csv(
                    event="controller_swapped",
                    controller=new_name,
                    previous_controller=old_name,
                    flush=True,
                )



        return SetParametersResult(successful=True)

    def _log_controller_minute(self):
        with self._policy_lock:
            policy_name = self.active_policy_name

        self._log_csv(
            event="controller_minute",
            controller=policy_name,
        )


    def close(self):
        self._minute_log_timer.cancel()
        self._csv_logger.close()

        # set packet rates from controllers here
        # controller defaults to publishing @ 10Hz
        # call these to set rate to 50Hz or provide argument for specific rate
        # self.set_pc_pack_rate(blocking=False)  # set PC publish rate to 50Hz
        # self.set_sc_pack_rate(blocking=False)  # set SC publish rate to 50Hz

        # Use this to set node clock to use sim time from /clock (from gazebo sim time)
        # Access node clock via self.get_clock() or other various
        # time-related functions of rclpy.Node
        # self.use_sim_time()

    # To subscribe to any topic, simply define the specific callback, e.g. power_callback
    # def power_callback(self, data):
    #     """Enables feedback of '/power_data' topic from Power Controller"""
    #     # get target value from control policy
    #     target_value = self.policy.target(data.rpm, data.scale, data.retract)

    #     # send a command, e.g. winding current
    #     self.send_pc_wind_curr_command(target_value, blocking=False)

    # Available commands to send within any callback:
    # self.send_pump_command(duration_mins, blocking=False)
    # self.send_valve_command(duration_sec, blocking=False)
    # self.send_pc_wind_curr_command(wind_curr_amps, blocking=False)
    # self.send_pc_bias_curr_command(bias_curr_amps, blocking=False)
    # self.send_pc_scale_command(scale_factor, blocking=False)
    # self.send_pc_retract_command(retract_factor, blocking=False)

    def ahrs_callback(self, data):
        """Provide feedback of '/ahrs_data' topic from XBowAHRS."""
        # Update class variables, get control policy target, send commands, etc.
        pass

    def battery_callback(self, data):
        """Provide feedback of '/battery_data' topic from Battery Controller."""
        # Update class variables, get control policy target, send commands, etc.
        pass

    def spring_callback(self, data):
        """Provide feedback of '/spring_data' topic from Spring Controller."""
        ## Updates for state variables ##
        self.active_policy.update_range(data.range_finder)

        ## Update the policy as needed
        self.active_policy.update_params(self.get_clock().now())

        ## Send out command
        self.send_command()

    def power_callback(self, data):
        """Provide feedback of '/power_data' topic from Power Controller."""
        # Update class variables, get control policy target, send commands, etc.
        pass

    def trefoil_callback(self, data):
        """Provide feedback of '/trefoil_data' topic from Trefoil Controller."""
        # Update class variables, get control policy target, send commands, etc.
        pass

    def powerbuoy_callback(self, data):
        """Provide feedback of '/powerbuoy_data' topic -- Aggregated data from all topics."""
        # Update class variables, get control policy target, send commands, etc.
        pass 

    def send_command(self):
        with self._state_lock: #Take state "screenshot"
            state = dict(self.state)
    
        with self._policy_lock:
            policy = self.policies[self.active_policy_name]
        now = self.get_clock().now()

        target  = policy.target(state, now) #Get target with screenshot

        if policy._piston_bounded:
            target = policy.piston_bounding(target)

        self.get_logger().info(
            f"{self.active_policy_name} sending {target['Control Knob']} value {target['Value']}"
        )
        match target["Control Knob"]:
            case 'Pump':
                self.send_pump_command(target["Value"], blocking=False)
            case 'Valve':
                self.send_valve_command(target["Value"], blocking=False)
            case 'Winding Current':
                self.send_pc_wind_curr_command(target["Value"], blocking=False)
            case 'Bias Current':
                self.send_pc_bias_curr_command(target["Value"], blocking=False)
            case 'Scale':
                self.send_pc_scale_command(target["Value"], blocking=False)
            case 'Retract':
                self.send_pc_retract_command(target["Value"], blocking=False)
            case 'None':
                    self.send_pump_command(0.0, blocking=False)
                    self.send_valve_command(0.0, blocking=False)
                    self.send_pc_wind_curr_command(0.0, blocking=False)
                    self.send_pc_bias_curr_command(0.0, blocking=False)
            case _:
                self.get_logger().info(
                    f"Send Command function called without matching control"
                    )
                pass

def main():
    rclpy.init()
    controller = Controller()
    try:
        controller.spin()
    finally:
        controller.close()
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

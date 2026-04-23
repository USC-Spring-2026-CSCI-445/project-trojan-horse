#!/usr/bin/env python3
from typing import Optional, Dict, List
from argparse import ArgumentParser
from math import sqrt, atan2, pi, inf
import math
import json
import numpy as np
import copy
import rospy
from geometry_msgs.msg import Twist
from random import uniform
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

# Import your existing implementations
from lab8_9_starter import Map, ParticleFilter,Particle,  PIDController, angle_to_neg_pi_to_pi # :contentReference[oaicite:2]{index=2}
from lab10_starter import RrtPlanner, PIDController as WaypointPID, GOAL_THRESHOLD  # :contentReference[oaicite:3]{index=3}


class PFRRTController:
    """
    Combined controller that:
      1) Localizes using a particle filter (by exploring).
      2) Plans with RRT from PF estimate to goal.
      3) Follows that plan with a waypoint PID controller while
         continuing to run the particle filter.
    """

    def __init__(self, pf: ParticleFilter, planner: RrtPlanner, goal_position: Dict[str, float]):
        self._pf = pf
        self._planner = planner
        self.goal_position = goal_position

        # Robot state from odom / laser
        self.current_position: Optional[Dict[str, float]] = None
        #self.last_odom: Optional[Dict[str, float]] = None
        self.laserscan: Optional[LaserScan] = None

        # Command publisher
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        
        # Subscribers
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.laserscan_callback)
        
        # PID controllers for tracking waypoints (copied from your ObstacleFreeWaypointController)
        #self.linear_pid = WaypointPID(0.3, 0.0, 0.1, 10, -0.22, 0.22)
        #self.angular_pid = WaypointPID(0.5, 0.0, 0.2, 10, -2.84, 2.84)
        
        # Waypoint tracking state
        self.plan: Optional[List[Dict[str, float]]] = None
        #self.current_wp_idx: int = 0

        #self.rate = rospy.Rate(10)

        while (self.current_position is None or self.laserscan is None) and (not rospy.is_shutdown()):
            rospy.loginfo("Waiting for /odom and /scan...")
            rospy.sleep(0.1)

    # ----------------------------------------------------------------------
    # Basic callbacks
    # ----------------------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )
        self.current_position = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

        new_pose = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

        # Use odom delta to propagate PF motion model
        # if self.last_odom is not None:
        #     dx_world = new_pose["x"] - self.last_odom["x"]
        #     dy_world = new_pose["y"] - self.last_odom["y"]
        #     dtheta = angle_to_neg_pi_to_pi(new_pose["theta"] - self.last_odom["theta"])

        #     # convert world delta to robot frame of previous pose
        #     ct = math.cos(self.last_odom["theta"])
        #     st = math.sin(self.last_odom["theta"])
        #     dx_robot = ct * dx_world + st * dy_world
        #     dy_robot = -st * dx_world + ct * dy_world

        #     # propagate all particles
        #     self._pf.move_by(dx_robot, dy_robot, dtheta)

        # self.last_odom = new_pose
        # self.current_position = new_pose
    def laserscan_callback(self, msg: LaserScan):
        self.laserscan = msg

    # ----------------------------------------------------------------------
    # Low-level motion primitives
    # ----------------------------------------------------------------------
    def move_forward(self, distance: float):
        """
        Move the robot straight by a commanded distance (meters)
        using a constant velocity profile.
        """
        #made some changes as needed for our code
        dist_pid = PIDController(kp=1.23, ki=0.1, kd=1.4, kS=0.5, u_min=-0.2, u_max=0.2)
        ang_pid = PIDController(kp=1.23, ki=0.19, kd=0.9, kS=0.4, u_min=-2.026, u_max=2.026)
        origin = copy.deepcopy(self.current_position)
        base_theta = origin["theta"]
        loop_rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            curr = self.current_position
            offset_x = curr["x"] - origin["x"]
            offset_y = curr["y"] - origin["y"]
            comp_x = offset_x * math.cos(base_theta)
            comp_y = offset_y * math.sin(base_theta)
            traveled = comp_x + comp_y
            err_dist = goal_dist - traveled
            theta_now = curr["theta"]
            err_theta = angle_to_neg_pi_to_pi(base_theta - theta_now)
            if abs(err_dist) < 0.02:
                break
            v_cmd = dist_pid.control(err_dist, rospy.get_time())
            w_cmd = ang_pid.control(err_theta, rospy.get_time())
            msg = Twist()
            msg.linear.x = v_cmd
            msg.angular.z = w_cmd
            self.cmd_pub.publish(msg)
            loop_rate.sleep()
        self.cmd_pub.publish(Twist())
        end = self.current_position
        dx_total = end["x"] - origin["x"]
        dy_total = end["y"] - origin["y"]
        dtheta_total = angle_to_neg_pi_to_pi(end["theta"] - base_theta)
        self._pf.move_by(dx_total, dy_total, dtheta_total)
        self._pf.visualize_particles()

    def rotate_in_place(self, angle: float):
        """
        Rotate robot by a relative angle (radians).
        """
        #made changes as needed
        rotate = PIDController(kp=1.23, ki=0.2, kd=1.1, kS=0.5, u_min=-2.026, u_max=2.026)
        begin = self.current_position["theta"]
        end = angle_to_neg_pi_to_pi(begin + angle)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            error = angle_to_neg_pi_to_pi(end - self.current_position["theta"])
            if abs(error) < 0.03:
                break
            angularvelo = rotate.control(error, rospy.get_time())
            twist = Twist()
            twist.angular.z = angularvelo
            self.cmd_pub.publish(twist)
            rate.sleep()
        self.cmd_pub.publish(Twist())
        realchange = angle_to_neg_pi_to_pi(self.current_position["theta"] - begin)
        self._pf.move_by(0, 0, realchange)
        self._pf.visualize_particles()

    # ----------------------------------------------------------------------
    # Measurement update
    # ----------------------------------------------------------------------
    def take_measurements(self):
        """
        Use 3 beams (-15°, 0°, +15° in the robot frame) from /scan
        to update the particle filter via its measurement model.
        """
        if self.laserscan is None:
            return
        selected_angles = [-135, -90, -45, 0, 45, 90, 135]
        for angle_deg in selected_angles:
            angle_rad = math.radians(angle_deg)
            idx = int((angle_rad - self.laserscan.angle_min) / self.laserscan.angle_increment)
            if idx < 0 or idx >= len(self.laserscan.ranges):
                continue
            z = self.laserscan.ranges[idx]
            if z == float("inf") or math.isnan(z):
                continue
            self._pf.measure(z, angle_rad)
        self._pf.resample()
        self._pf.visualize_particles()
        self._pf.visualize_estimate()

    # ----------------------------------------------------------------------
    # Phase 1: Localization with PF (explore a bit)
    # ----------------------------------------------------------------------
    def localize_with_pf(self, max_steps: int = 400):
        """
        Simple autonomous exploration policy:
          - If front is free, go forward.
          - If obstacle close in front, back up and rotate.
        After each motion, apply PF measurement updates and check convergence.
        """
        ######### Your code starts here #########
        rate = rospy.Rate(10)
        rando_prob = 0.1 #helpful to explore rando
        spread_strict = 0.08 #how strict i want the cluster to be
        num_converge = 1 #i only wanna converge once
        con_cnt = 0 #converge count
        while not rospy.is_shutdown():
            standevix = np.std([p.x for p in self._pf._particles])#get standard deviation
            standeviy = np.std([p.y for p in self._pf._particles])#smaller the better
            rospy.loginfo(f"standard dev of x: {standevix:.3f}, of y: {standeviy:.3f} ||| num of converges: {con_cnt}/{num_converge}")
            if standevix < spread_strict and standeviy < spread_strict: #if spread is small than we are good
                con_cnt += 1
                rospy.loginfo(f"spread is good...converges: ({con_cnt}/{num_converge})")
                if con_cnt >= num_converge: #if the converges are done
                    rospy.loginfo("particle process done")
                    break
                rospy.loginfo("restart for next set")#this is in case we have multiple converges
                x_minimum, x_maximum = self._pf._map.map_aabb[0], self._pf._map.map_aabb[1]
                y_minimum, y_maximum = self._pf._map.map_aabb[2], self._pf._map.map_aabb[3]
                newtparts = []
                resetparts = 0
                while resetparts < self._pf._n_particles:
                    x = uniform(x_minimum, x_maximum)
                    y = uniform(y_minimum, y_maximum)
                    if not (x_minimum <= x <= x_maximum and y_minimum <= y <= y_maximum):
                        continue
                    in_obstacle = False
                    for mapobs in self._pf._map.obstacles:
                        if mapobs[0] <= x <= mapobs[1] and mapobs[2] <= y <= mapobs[3]:
                            in_obstacle = True
                            break
                    if in_obstacle:
                        continue
                    theta = uniform(-pi, pi)
                    newtparts.append(Particle(x, y, theta, 0.0))
                    resetparts += 1
                self._pf._particles = newtparts
                self._pf.visualize_particles()
            beam_index = int((0.0 - self.laserscan.angle_min) / self.laserscan.angle_increment) #we are scanning, finding index of beam
            beam_index = max(0, min(len(self.laserscan.ranges) - 1, beam_index))#clamping index
            wallfrontdist = self.laserscan.ranges[beam_index]#front beam dist
            if math.isnan(wallfrontdist) or (wallfrontdist != float("inf") and wallfrontdist < 0.55): #wall in front
                rospy.loginfo("wall in front")
                turn = np.random.choice([pi /2, -pi/ 2])#truning
                self.rotate_in_place(turn)
            else:
                if np.random.rand() < rando_prob: #random turn for exploration
                    rospy.loginfo("random turn")
                    turn = np.random.choice([pi/ 2, -pi/ 2])
                    self.rotate_in_place(turn)
                else:
                    self.move_forward(0.4)
            self.take_measurements()
            rate.sleep()
        self.cmd_pub.publish(Twist())
        ######### Your code ends here #########

        

    # ----------------------------------------------------------------------
    # Phase 2: Planning with RRT
    # ----------------------------------------------------------------------
    def plan_with_rrt(self):
        """
        Generate a path using RRT from PF-estimated start to known goal.
        """
        ######### Your code starts here #########
        estimate_x, estimate_y, _ = self._pf.get_estimate() #estimate of robot position
        start = {"x": estimate_x, "y": estimate_y} #use estimate to start plan
        rospy.loginfo(f"start position x={estimate_x:.3f}  y={estimate_y:.3f}")
        rospy.loginfo(f"goal position x={self.goal_position['x']:.3f}  y={self.goal_position['y']:.3f}")
        self.plan, graph = self._planner.generate_plan(start, self.goal_position) #generate rrt plan
        if not self.plan:#contingency plans
            rospy.logwarn("retrying")
            odom_start = {
                "x": self.current_position["x"],
                "y": self.current_position["y"],
            }
            self.plan, graph = self._planner.generate_plan(odom_start, self.goal_position)
        if not self.plan:
            rospy.logerr("quitting")
            return
        rospy.loginfo(f"plan ready there are {len(self.plan)} waypoints")
        self._planner.visualize_plan(self.plan)
        self._planner.visualize_graph(graph)
        ######### Your code ends here #########

    # ----------------------------------------------------------------------
    # Phase 3: Following the RRT path
    # ----------------------------------------------------------------------
    def follow_plan(self):
        """
        Follow the RRT waypoints using PID on (distance, heading) error.
        Keep updating PF along the way.
        """
        ######### Your code starts here #########
        if not self.plan:
            rospy.logwarn("no plan is present")
            return

        rospy.loginfo(f"{len(self.plan)} waypoints are there")
        lincontrol = PIDController(1.21, 0.0, 1.4, kS=0.5, u_min=0.0, u_max=0.226)# this is linear movement
        angcontrol = PIDController(1.21, 0.17, 0.9, kS=0.5, u_min=-2.06, u_max=2.06)#this is angular movement
        rate = rospy.Rate(20)
        ctrl_msg = Twist()
        wp_index = 0 #initialize the waypoint
        while not rospy.is_shutdown():
            if self.current_position is None:
                rate.sleep()#waiting
                continue
            if wp_index >= len(self.plan):
                ctrl_msg.linear.x = 0.0 #stop robo if all waypoints reached
                ctrl_msg.angular.z = 0.0
                self.cmd_pub.publish(ctrl_msg)
                break
            target = self.plan[wp_index]#get a target waypoint
            dx = target["x"] - self.current_position["x"]#get dist to target waypoint
            dy = target["y"] - self.current_position["y"]
            dist = sqrt(dx**2 + dy**2)
            desired_heading = atan2(dy, dx) #get heading pointing to target wp
            heading_err = desired_heading - self.current_position["theta"] # get heading error
            heading_err = atan2(math.sin(heading_err), math.cos(heading_err))#wrapping in case
            t = rospy.get_time()#getting control sigs
            velocitylin = lincontrol.control(dist, t)
            velocityang = angcontrol.control(heading_err, t)
            if abs(heading_err) > 0.5:
                velocitylin = 0.0#dont move if we face the wrong way
            ctrl_msg.linear.x = velocitylin
            ctrl_msg.angular.z = velocityang
            self.cmd_pub.publish(ctrl_msg)
            if dist < GOAL_THRESHOLD: #check if we reached
                wp_index += 1

            rate.sleep()

        ######### Your code ends here #########

    # ----------------------------------------------------------------------
    # Top-level
    # ----------------------------------------------------------------------
    def run(self):
        self.localize_with_pf()
        self.plan_with_rrt()
        self.follow_plan()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--map_filepath", type=str, required=True)
    args = parser.parse_args()

    with open(args.map_filepath, "r") as f:
        map_data = json.load(f)
        obstacles = map_data["obstacles"]
        map_aabb = map_data["map_aabb"]
        if "goal_position" not in map_data:
            raise RuntimeError("Map JSON must contain a 'goal_position' field.")
        goal_position = map_data["goal_position"]

    # Initialize ROS node
    rospy.init_node("pf_rrt_combined", anonymous=True)

    # Build map + PF + RRT
    map_obj = Map(obstacles, map_aabb)
    num_particles = 200
    translation_variance = 0.003
    rotation_variance = 0.03
    measurement_variance = 0.1

    pf = ParticleFilter(
        map_obj,
        num_particles,
        translation_variance,
        rotation_variance,
        measurement_variance,
    )
    planner = RrtPlanner(obstacles, map_aabb)

    controller = PFRRTController(pf, planner, goal_position)

    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass

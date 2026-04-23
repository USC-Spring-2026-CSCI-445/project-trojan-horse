#!/usr/bin/env python3
from typing import Optional, Tuple, List, Dict
from argparse import ArgumentParser
from math import inf, sqrt, atan2, pi
from time import sleep, time
import queue
import json
import math
from random import uniform
import copy

import scipy
import numpy as np
import rospy
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Twist, Point32, PoseStamped, Pose, Vector3, Quaternion, Point, PoseArray
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, PointCloud, ChannelFloat32
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import scipy.stats
from numpy.random import choice

np.set_printoptions(linewidth=200)

# AABB format: (x_min, x_max, y_min, y_max)
OBS_TYPE = Tuple[float, float, float, float]
# Position format: {"x": x, "y": y, "theta": theta}
POSITION_TYPE = Dict[str, float]

# don't change this
GOAL_THRESHOLD = 0.1


def angle_to_0_to_2pi(angle: float) -> float:
    while angle < 0:
        angle += 2 * pi
    while angle > 2 * pi:
        angle -= 2 * pi
    return angle


def angle_to_neg_pi_to_pi(angle: float) -> float:
    while angle < -pi:
        angle += 2 * pi
    while angle > pi:
        angle -= 2 * pi
    return angle


# see https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
def ray_line_intersection(ray_origin, ray_direction_rad, point1, point2):
    # Convert to numpy arrays
    ray_origin = np.array(ray_origin, dtype=np.float32)
    ray_direction = np.array([math.cos(ray_direction_rad), math.sin(ray_direction_rad)])
    point1 = np.array(point1, dtype=np.float32)
    point2 = np.array(point2, dtype=np.float32)

    # Ray-Line Segment Intersection Test in 2D
    v1 = ray_origin - point1
    v2 = point2 - point1
    v3 = np.array([-ray_direction[1], ray_direction[0]])
    denominator = np.dot(v2, v3)
    if denominator == 0:
        return None
    t1 = np.cross(v2, v1) / denominator
    t2 = np.dot(v1, v3) / denominator
    if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
        return [ray_origin + t1 * ray_direction]
    return None


class Map:
    def __init__(self, obstacles: List[OBS_TYPE], map_aabb: Tuple):
        self.obstacles = obstacles
        self.map_aabb = map_aabb

    @property
    def top_right(self) -> Tuple[float, float]:
        return self.map_aabb[1], self.map_aabb[3]

    @property
    def bottom_left(self) -> Tuple[float, float]:
        return self.map_aabb[0], self.map_aabb[2]

    def draw_distances(self, origins: List[Tuple[float, float]]):
        """Example usage:
        map_ = Map(obstacles, map_aabb)
        map_.draw_distances([(0.0, 0.0), (3, 3), (1.5, 1.5)])
        """

        # Draw scene
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(10, 10))
        fig.tight_layout()
        x_min_global, x_max_global, y_min_global, y_max_global = self.map_aabb
        for aabb in self.obstacles:
            width = aabb[1] - aabb[0]
            height = aabb[3] - aabb[2]
            rect = patches.Rectangle(
                (aabb[0], aabb[2]), width, height, linewidth=2, edgecolor="r", facecolor="r", alpha=0.4
            )
            ax.add_patch(rect)
        ax.set_xlim(x_min_global, x_max_global)
        ax.set_ylim(y_min_global, y_max_global)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("2D Plot of Obstacles")
        ax.set_aspect("equal", "box")
        plt.grid(True)

        # Draw rays
        angles = np.linspace(0, 2 * math.pi, 10, endpoint=False)
        for origin in origins:
            for angle in angles:
                closest_distance = self.closest_distance(origin, angle)
                if closest_distance is not None:
                    x = origin[0] + closest_distance * math.cos(angle)
                    y = origin[1] + closest_distance * math.sin(angle)
                    plt.plot([origin[0], x], [origin[1], y], "b-")
        plt.show()

    def closest_distance(self, origin: Tuple[float, float], angle: float) -> Optional[float]:
        """Returns the closest distance to an obstacle from the given origin in the given direction `angle`. If no
        intersection is found, returns `None`.
        """

        def lines_from_obstacle(obstacle: OBS_TYPE):
            """Returns the four lines of the given AABB format obstacle.
            Example usage: `point0, point1 = lines_from_obstacle(self.obstacles[0])`
            """
            x_min, x_max, y_min, y_max = obstacle
            return [
                [(x_min, y_min), (x_max, y_min)],
                [(x_max, y_min), (x_max, y_max)],
                [(x_max, y_max), (x_min, y_max)],
                [(x_min, y_max), (x_min, y_min)],
            ]

        # Iterate over the obstacles in the map to find the closest distance (if there is one). Remember that the
        # obstacles are represented as a list of AABBs (Axis-Aligned Bounding Boxes) with the format
        # (x_min, x_max, y_min, y_max).
        result = None
        origin = np.array(origin)

        for obstacle in self.obstacles:
            for line in lines_from_obstacle(obstacle):
                p = ray_line_intersection(origin, angle, line[0], line[1])
                if p is None:
                    continue

                dist = np.linalg.norm(np.array(p) - origin)
                if result is None:
                    result = dist
                else:
                    result = min(result, dist)
        return result

# PID controller class
######### Your code starts here #########
class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, kS=0.5, u_min=-1.0, u_max=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.kS = kS
        self.u_min = u_min
        self.u_max = u_max
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = None

    def control(self, error: float, current_time: float) -> float:
        if self.prev_time is None:
            self.prev_time = current_time
            return 0.0
        dt = current_time - self.prev_time
        if dt <= 1e-6:
            return 0.0
        derivative = (error - self.prev_error) / dt
        self.integral += error * dt
        self.integral = max(-self.kS, min(self.kS, self.integral))
        control_signal = ( self.kp * error + self.ki * self.integral + self.kd * derivative)
        control_signal = max(self.u_min, min(self.u_max, control_signal))
        self.prev_error = error
        self.prev_time = current_time

        return control_signal
######### Your code ends here #########

class Particle:
    def __init__(self, x: float, y: float, theta: float, log_p: float):
        self.x = x
        self.y = y
        self.theta = theta
        self.log_p = log_p

    def __str__(self) -> str:
        return f"Particle<pose: {self.x, self.y, self.theta}, log_p: {self.log_p}>"


class ParticleFilter:

    def __init__(
        self,
        map_: Map,
        n_particles: int,
        translation_variance: float,
        rotation_variance: float,
        measurement_variance: float,
    ):
        self.particles_visualization_pub = rospy.Publisher("/pf_particles", PoseArray, queue_size=10)
        self.estimate_visualization_pub = rospy.Publisher("/pf_estimate", PoseStamped, queue_size=10)

        # Initialize uniformly-distributed particles
        ######### Your code starts here #########
        self._particles = []
        self._map = map_
        self._translation_variance = translation_variance
        self._rotation_variance = rotation_variance
        self._measurement_variance = measurement_variance
        self._n_particles = n_particles

        x_minimum, x_maximum, y_minimum, y_maximum = self._map.map_aabb

        def valid(x_value, y_value):
            if not (x_minimum <= x_value <= x_maximum and y_minimum <= y_value <= y_maximum):
                return False
            for obstacle in self._map.obstacles:
                if obstacle[0] <= x_value <= obstacle[1] and obstacle[2] <= y_value <= obstacle[3]:
                    return False
            return True

        count = 0
        while count < self._n_particles:
            x_value = uniform(x_minimum, x_maximum)
            y_value = uniform(y_minimum, y_maximum)

            if not valid(x_value, y_value):
                continue

            theta_value = uniform(-math.pi, math.pi)
            self._particles.append(Particle(x_value, y_value, theta_value, 0.0))
            count += 1
        ######### Your code ends here #########

    def visualize_particles(self):
        pa = PoseArray()
        pa.header.frame_id = "odom"
        pa.header.stamp = rospy.Time.now()
        for particle in self._particles:
            pose = Pose()
            pose.position = Point(particle.x, particle.y, 0.01)
            q_np = quaternion_from_euler(0, 0, float(particle.theta))
            pose.orientation = Quaternion(*q_np.tolist())
            pa.poses.append(pose)
        self.particles_visualization_pub.publish(pa)

    def visualize_estimate(self):
        ps = PoseStamped()
        ps.header.frame_id = "odom"
        ps.header.stamp = rospy.Time.now()
        x, y, theta = self.get_estimate()
        pose = Pose()
        pose.position = Point(x, y, 0.01)
        q_np = quaternion_from_euler(0, 0, float(theta))
        pose.orientation = Quaternion(*q_np.tolist())
        ps.pose = pose
        self.estimate_visualization_pub.publish(ps)

    def move_by(self, delta_x, delta_y, delta_theta):
        delta_theta = angle_to_neg_pi_to_pi(delta_theta)

        # Propagate motion of each particle
        ######### Your code starts here #########
        x_minimum, x_maximum, y_minimum, y_maximum = self._map.map_aabb

        def valid(x_value, y_value):
            if not (x_minimum <= x_value <= x_maximum and y_minimum <= y_value <= y_maximum):
                return False
            for obstacle in self._map.obstacles:
                if obstacle[0] <= x_value <= obstacle[1] and obstacle[2] <= y_value <= obstacle[3]:
                    return False
            return True

        def crosses_wall(x1, y1, x2, y2):
            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx**2 + dy**2)
            if dist < 1e-6:
                return False
            angle = math.atan2(dy, dx)
            for obs in self._map.obstacles:
                x_min, x_max, y_min, y_max = obs
                walls = [
                    [(x_min, y_min), (x_max, y_min)],
                    [(x_max, y_min), (x_max, y_max)],
                    [(x_max, y_max), (x_min, y_max)],
                    [(x_min, y_max), (x_min, y_min)],
                ]
                for wall in walls:
                    result = ray_line_intersection((x1, y1), angle, wall[0], wall[1])
                    if result is not None:
                        hit_dist = np.linalg.norm(np.array(result[0]) - np.array([x1, y1]))
                        if hit_dist <= dist:
                            return True
            return False

        for p in self._particles:
            new_theta = angle_to_neg_pi_to_pi(
                p.theta + delta_theta + np.random.normal(0, self._rotation_variance)
            )

            dist = math.sqrt(delta_x**2 + delta_y**2)

            if dist > 1e-6:
                noisy_dist = dist + np.random.normal(0, self._translation_variance)
                move_angle = p.theta + np.random.normal(0, self._rotation_variance / 2)

                next_x = p.x + noisy_dist * math.cos(move_angle)
                next_y = p.y + noisy_dist * math.sin(move_angle)

                if crosses_wall(p.x, p.y, next_x, next_y) or not valid(next_x, next_y):
                    p.log_p = -1e12
                    p.theta = new_theta
                    continue

                p.x = next_x
                p.y = next_y

            p.theta = new_theta
        ######### Your code ends here #########

    def measure(self, z: float, scan_angle_in_rad: float):
        """Update the particles based on the measurement `z` at the given `scan_angle_in_rad`.

        Args:
            z: distance to an obstacle
            scan_angle_in_rad: Angle in the robots frame where the scan was taken
        """

        # Calculate posterior probabilities and resample
        ######### Your code starts here #########
        x_minimum, x_maximum, y_minimum, y_maximum = self._map.map_aabb

        def valid(x_value, y_value):
            if not (x_minimum <= x_value <= x_maximum and y_minimum <= y_value <= y_maximum):
                return False
            for obstacle in self._map.obstacles:
                if obstacle[0] <= x_value <= obstacle[1] and obstacle[2] <= y_value <= obstacle[3]:
                    return False
            return True
        sigma = math.sqrt(self._measurement_variance)
        max_range = 10.0

        for p in self._particles:
            if not valid(p.x, p.y):
                p.log_p = -1e12
                continue

            expected = self._map.closest_distance(
                (p.x, p.y),
                p.theta + scan_angle_in_rad
            )

            if expected is None:
                p.log_p = -1e12
                continue

            gauss = scipy.stats.norm(loc=expected, scale=sigma).pdf(z)
            uniform_prob = 1.0 / max_range

            likelihood = 0.9 * gauss + 0.1 * uniform_prob

            p.log_p += math.log(likelihood + 1e-12)
    def resample(self):
        def valid(x_value, y_value):
            if not (x_minimum <= x_value <= x_maximum and y_minimum <= y_value <= y_maximum):
                return False
            for obstacle in self._map.obstacles:
                if obstacle[0] <= x_value <= obstacle[1] and obstacle[2] <= y_value <= obstacle[3]:
                    return False
            return True
        x_minimum, x_maximum, y_minimum, y_maximum = self._map.map_aabb
        estimate_x, estimate_y, estimate_theta = self.get_estimate()
        replaced = 0
        for p in self._particles:
            if p.log_p < -1e9:
                for _ in range(50):
                    if uniform(0, 1) < 0.7:
                        rx = estimate_x + np.random.normal(0, 0.15)
                        ry = estimate_y + np.random.normal(0, 0.15)
                        rtheta = angle_to_neg_pi_to_pi(estimate_theta + np.random.normal(0, 0.2))
                    else:
                        rx = uniform(x_minimum, x_maximum)
                        ry = uniform(y_minimum, y_maximum)
                        rtheta = uniform(-math.pi, math.pi)
                    if valid(rx, ry):
                        p.x, p.y, p.theta = rx, ry, rtheta
                        p.log_p = 0.0
                        replaced += 1
                        break
        log_vals = np.array([p.log_p for p in self._particles])
        log_vals -= np.max(log_vals)
        weights = np.exp(log_vals)
        weights /= np.sum(weights)
        n_eff = 1.0 / np.sum(weights**2)
        n_threshold = 0.5 * self._n_particles

        if n_eff >= n_threshold:
            for p, w in zip(self._particles, weights):
                p.log_p = math.log(w + 1e-12)
            return  
        updated = []
        r = uniform(0, 1.0 / self._n_particles)
        c = weights[0]
        i = 0
        for m in range(self._n_particles):
            u = r + m / self._n_particles
            while u > c:
                i += 1
                if i >= len(weights):
                    i = len(weights) - 1
                    break
                c += weights[i]
            new_p = self._particles[i]
            updated.append(Particle(
                new_p.x + np.random.normal(0, 0.02),
                new_p.y + np.random.normal(0, 0.02),
                angle_to_neg_pi_to_pi(new_p.theta + np.random.normal(0, 0.01)),
                0.0
            ))

        self._particles = updated
        ######### Your code ends here #########

    def get_estimate(self) -> Tuple[float, float, float]:
        # Estimate robot's location using particle weights
        ######### Your code starts here #########
        particle_logs = np.array([p.log_p for p in self._particles], dtype=np.float64) #get log probs of each particle
        maximum_logp = np.max(particle_logs)
        if np.isneginf(maximum_logp): #if all are neg inf reset to uniform dist and return avg of particles
            weights = np.ones(len(self._particles), dtype=np.float64) / len(self._particles)
        else:
            weights = np.exp(particle_logs - maximum_logp) #subtract max logp for numerical stability then get wieghts by exponentiating
            weights_sum = np.sum(weights)
            if weights_sum == 0 or np.isnan(weights_sum):#if all weights are zero reset to unifrom dist
                weights = np.ones(len(self._particles), dtype=np.float64) / len(self._particles)
            else:
                weights /= weights_sum #normalize
        
        all_x = np.array([p.x for p in self._particles], dtype=np.float64)#get x,y,theta of each particle
        all_y = np.array([p.y for p in self._particles], dtype=np.float64)
        all_theta = np.array([p.theta for p in self._particles], dtype=np.float64)
        x_estimate = np.sum(all_x * weights)#get estimates by weighted avg of all particlaes
        y_estimate = np.sum(all_y * weights)
        theta_estimate = atan2(np.sum(np.sin(all_theta) * weights), np.sum(np.cos(all_theta) * weights))
        return x_estimate, y_estimate, theta_estimate
        ######### Your code ends here #########


class Controller:
    def __init__(self, particle_filter: ParticleFilter):
        rospy.init_node("particle_filter_controller", anonymous=True)
        self._particle_filter = particle_filter
        self._particle_filter.visualize_particles()

        #
        self.current_position = None
        self.laserscan = None
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.laserscan_sub = rospy.Subscriber("/scan", LaserScan, self.robot_laserscan_callback)
        self.robot_ctrl_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.pointcloud_pub = rospy.Publisher("/scan_pointcloud", PointCloud, queue_size=10)
        self.target_position_pub = rospy.Publisher("/waypoints", MarkerArray, queue_size=10)

        while ((self.current_position is None) or (self.laserscan is None)) and (not rospy.is_shutdown()):
            rospy.loginfo("waiting for odom and laserscan")
            rospy.sleep(0.1)

    def odom_callback(self, msg):
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.current_position = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

    def robot_laserscan_callback(self, msg: LaserScan):
        self.laserscan = msg

    def visualize_laserscan_ranges(self, idx_groups: List[Tuple[int, int]]):
        """Helper function to visualize ranges of sensor readings from the laserscan lidar.

        Example usage for visualizing the first 10 and last 10 degrees of the laserscan:
            `self.visualize_laserscan_ranges([(0, 10), (350, 360)])`
        """
        pcd = PointCloud()
        pcd.header.frame_id = "odom"
        pcd.header.stamp = rospy.Time.now()
        for idx_low, idx_high in idx_groups:
            for idx, d in enumerate(self.laserscan.ranges[idx_low:idx_high]):
                if d == inf:
                    continue
                angle = math.radians(idx) + self.current_position["theta"]
                x = d * math.cos(angle) + self.current_position["x"]
                y = d * math.sin(angle) + self.current_position["y"]
                z = 0.1
                pcd.points.append(Point32(x=x, y=y, z=z))
                pcd.channels.append(ChannelFloat32(name="rgb", values=(0.0, 1.0, 0.0)))
        self.pointcloud_pub.publish(pcd)

    def visualize_position(self, x: float, y: float):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position = Point(x, y, 0.0)
        marker.pose.orientation = Quaternion(0, 0, 0, 1)
        marker.scale = Vector3(0.075, 0.075, 0.1)
        marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.5)
        marker_array.markers.append(marker)
        self.target_position_pub.publish(marker_array)

    def take_measurements(self):
        # Take measurement using LIDAR
        ######### Your code starts here #########
        # NOTE: with more than 2 angles the particle filter will converge too quickly, so with high likelihood the
        # correct neighborhood won't be found.
        if self.laserscan is None:
            return

        selected_angles = [-45, 0, 45]

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
        ######### Your code ends here #########

    def autonomous_exploration(self):
        """Randomly explore the environment here, while making sure to call `take_measurements()` and
        `_particle_filter.move_by()`. The particle filter should converge on the robots position eventually.

        Note that the following visualizations functions are available:
            visualize_position(...)
            visualize_laserscan_ranges(...)
        """
        # Robot autonomously explores environment while itzes itself
        ######### Your code starts here #########
        stable =0
        step= 0
        rate = rospy.Rate(10)
        while not rospy.is_shutdown(): #exploreuntil the convergence is met
            frontindex = int(round((0 - self.laserscan.angle_min) / self.laserscan.angle_increment))
            frontindex = max(0,min(frontindex, len(self.laserscan.ranges)-1))
            frontdistance = self.laserscan.ranges[frontindex]
            if np.isinf(frontdistance) or np.isnan(frontdistance):
                frontdistance = 10.0
            if frontdistance < 0.5:# if obs is close turn
                turn_dir = 1 if step % 2 == 0 else -1
                goal_theta = angle_to_neg_pi_to_pi(self.current_position["theta"] + turn_dir * math.pi / 2)
                self.rotate_action(goal_theta)
            else:
                self.forward_action(0.25)
            self.take_measurements()
            step += 1
            estimate_x, estimate_y, estimate_theta = self._particle_filter.get_estimate()#get estimates
            all_x = np.array([p.x for p in self._particle_filter._particles], dtype=np.float64)
            all_y = np.array([p.y for p in self._particle_filter._particles], dtype=np.float64)
            all_theta = np.array([p.theta for p in self._particle_filter._particles], dtype=np.float64)
            dispersement = np.sqrt(np.var(all_x) + np.var(all_y))#varience of particles in x and y direction
            heading =np.sqrt(np.mean(np.sin(all_theta))**2 + np.mean(np.cos(all_theta))**2)#concet of parts in same direction
            xydistance = np.sqrt((all_x - estimate_x)**2 + (all_y - estimate_y)**2) #dists of particles to estimate
            cluster= np.mean(xydistance < 0.16) #perecntage of particles close to estimate
            print("auto step is taken")
            if dispersement < 0.05 and heading > 0.5 and cluster > 0.9:
                stable += 1
            else:
                stable = 0
            
            if step>=7 or stable >=3:
                print("auto exploration is done")
                break
            rate.sleep()
        ######### Your code ends here #########

    def forward_action(self, distance: float):
        # Robot moves forward by a set amount during manual control
        ######### Your code starts here #########
        pid_dist = PIDController(1.2, 0.0, 1.5, u_min=-0.22, u_max=0.22)
        pid_angle = PIDController(1.2, 0.2, 1.0, u_min=-2.0, u_max=2.0)

        start = copy.deepcopy(self.current_position)
        start_theta = start["theta"]

        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            dx = self.current_position["x"] - start["x"]
            dy = self.current_position["y"] - start["y"]

            forward_progress = dx * math.cos(start_theta) + dy * math.sin(start_theta)
            error_dist = distance - forward_progress

            error_angle = angle_to_neg_pi_to_pi(start_theta - self.current_position["theta"])

            if abs(error_dist) < 0.02:
                break

            v = pid_dist.control(error_dist, rospy.get_time())
            w = pid_angle.control(error_angle, rospy.get_time())

            twist = Twist()
            twist.linear.x = v
            twist.angular.z = w
            self.robot_ctrl_pub.publish(twist)

            rate.sleep()

        self.robot_ctrl_pub.publish(Twist())

        dx = self.current_position["x"] - start["x"]
        dy = self.current_position["y"] - start["y"]
        dtheta = angle_to_neg_pi_to_pi(self.current_position["theta"] - start_theta)

        self._particle_filter.move_by(dx, dy, dtheta)
        ######### Your code ends here #########

    def rotate_action(self, goal_theta: float):
        # Robot turns by a set amount during manual control
        ######### Your code starts here #########
        pid = PIDController(1.2, 0.2, 1.0, u_min=-2.0, u_max=2.0)

        start_theta = self.current_position["theta"]
        target_theta = angle_to_neg_pi_to_pi(start_theta + goal_theta)

        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            error = angle_to_neg_pi_to_pi(target_theta - self.current_position["theta"])

            if abs(error) < 0.03:
                break

            ang_vel = pid.control(error, rospy.get_time())

            twist = Twist()
            twist.angular.z = ang_vel
            self.robot_ctrl_pub.publish(twist)

            rate.sleep()

        self.robot_ctrl_pub.publish(Twist())

        actual_delta = angle_to_neg_pi_to_pi(self.current_position["theta"] - start_theta)
        self._particle_filter.move_by(0, 0, actual_delta)
        ######### Your code ends here ######### 


""" Example usage

rosrun development lab8_9.py --map_filepath src/csci455l/scripts/lab8_9_map.json
"""


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--map_filepath", type=str, required=True)
    args = parser.parse_args()
    with open(args.map_filepath, "r") as f:
        map_ = json.load(f)
        obstacles = map_["obstacles"]
        map_aabb = map_["map_aabb"]

    map_ = Map(obstacles, map_aabb)
    num_particles = 200
    translation_variance = 0.003
    rotation_variance = 0.03
    measurement_variance = 0.1
    particle_filter = ParticleFilter(map_, num_particles, translation_variance, rotation_variance, measurement_variance)
    controller = Controller(particle_filter)

    try:
        # Manual control
        goal_theta = 0
        controller.take_measurements()
        while not rospy.is_shutdown():
            print("\nEnter 'a', 'w', 's', 'd' to move the robot:")
            uinput = input("")
            if uinput == "w": # forward
                ######### Your code starts here #########
                controller.forward_action(0.28)
                ######### Your code ends here #########
            elif uinput == "a": # left
                ######### Your code starts here #########
                goal_theta = angle_to_neg_pi_to_pi(controller.current_position["theta"] + math.pi / 2)# add 90 degs to current theta
                controller.rotate_action(goal_theta)
                ######### Your code ends here #########
            elif uinput == "d": #right
                ######### Your code starts here #########
                goal_theta = angle_to_neg_pi_to_pi(controller.current_position["theta"] - math.pi / 2)# subtract 90 degs
                controller.rotate_action(goal_theta)
                ######### Your code ends here #########
            elif uinput == "s": # backwards
                ######### Your code starts here #########
                controller.forward_action(-0.28)
                ######### Your code ends here #########
            elif uinput == "auto":
                print("Switching to auto explore")
                break
            else:
                print("Invalid input")
            ######### Your code starts here #########
            controller.take_measurements()
            controller._particle_filter.visualize_estimate()
            controller._particle_filter.visualize_particles()
            ######### Your code ends here #########

        # Autonomous exploration
        ######### Your code starts here #########
        controller.autonomous_exploration()
        ######### Your code ends here #########

    except rospy.ROSInterruptException:
        print("Shutting down...")

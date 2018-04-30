from grid import *
from particle import Particle
from utils import *
from setting import *
import math
import copy
import random

def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments: 
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*

        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    motion_particles = []
    for particle in particles:
        rand_odom = add_odometry_noise(odom, ODOM_HEAD_SIGMA, ODOM_TRANS_SIGMA)
        dx, dy = rotate_point(rand_odom[0], rand_odom[1], particle.h)
        motion_particles.append(Particle(
            particle.x + dx,
            particle.y + dy,
            heading=(particle.h + rand_odom[2]) % 360
        ))
    return motion_particles

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments: 
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before measurement update (but after motion update)

        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree

                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
				* Note that the robot can see mutliple markers at once, and may not see any one

        grid -- grid world map, which contains the marker information, 
                see grid.py and CozGrid for definition
                Can be used to evaluate particles

        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """
    tot_particles = len(particles)

    # Weigh particles
    weights = [0]
    tot_weight = 0
    for particle in particles:
        cur_score = score(particle.read_markers(grid), measured_marker_list)
        tot_weight += cur_score
        weights.append(cur_score)
    tot_weight = max(1, tot_weight)
    for i in range(1, tot_particles + 1):
        # Convert to normalized cdf
        weights[i] = weights[i] / tot_weight + weights[i-1]
    weights[-1] = 1

    # Resample particles
    measured_particles = []
    num_rand_particles = min(10, int(tot_particles / 10))
    for i in range(tot_particles - num_rand_particles):
        pick = random.uniform(0, 1)
        index = binary_search(pick, weights)
        particle_to_add = copy.deepcopy(particles[index])
        particle_to_add.x = add_gaussian_noise(particle_to_add.x, ODOM_TRANS_SIGMA)
        particle_to_add.y = add_gaussian_noise(particle_to_add.y, ODOM_TRANS_SIGMA)
        particle_to_add.h = add_gaussian_noise(particle_to_add.h, ODOM_HEAD_SIGMA)
        measured_particles.append(particle_to_add)

    # Generate random particles
    measured_particles += Particle.create_random(num_rand_particles, grid)

    return measured_particles

def score(particle_marker_list, measured_marker_list):
    len_p = len(particle_marker_list)
    len_m = len(measured_marker_list)
    if len_p != len_m:
        return 0
    elif len_m == 0:
        return 1
    else:
        score = 0
        for i in range(len_p):
            score += math.e ** -get_dist(particle_marker_list[i], measured_marker_list[i])
        return score

def get_dist(p, m):
    # Law of cosines
    a = grid_distance(p[0], p[1], 0, 0)
    b = grid_distance(m[0], m[1], 0, 0)
    angle = min((p[2] - m[2]) % 360, (m[2] - p[2]) % 360)
    return math.sqrt(
        a ** 2
        + b ** 2
        - 2 * a * b * math.cos(math.radians(angle))
    )

def binary_search(target, cdf):
    min = 0
    max = len(cdf) - 2
    while True:
        if min >= max:
            return min
        cur = int((min + max) / 2)
        if cdf[cur] <= target and target <= cdf[cur+1]:
            return cur
        elif target < cdf[cur]:
            max = cur - 1
        else:
            min = cur + 1

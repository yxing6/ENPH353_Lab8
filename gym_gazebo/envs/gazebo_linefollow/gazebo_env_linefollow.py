import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected
        self.mid_x = 0.0  # center of the frame initialized to be 0, updated at each find_middle function call

    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # cv2.imshow("raw", cv_image)

        NUM_BINS = 3
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        done = False

        # image processing:
        # change the frame to grey scale
        gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        # Get the total height and width information of the frame
        dim_y, dim_x, _ = gray.shape

        # blur it so the "road" far away become less easy to detect
        # kernel_size: Gaussian kernel size
        # sigma_x: Gaussian kernel standard deviation in X direction
        # sigma_y: Gaussian kernel standard deviation in Y direction
        kernel_size = 13
        sigma_x = 5
        sigma_y = 5
        blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma_x, sigma_y)  # gray scale the image

        # binary it
        ret, binary = cv2.threshold(blur_gray, 70, 255, cv2.THRESH_BINARY)
        last_row = binary[-1, :]

        first_index = 0
        last_index = 0

        if np.any(last_row == 0):
            last_list = last_row.tolist()
            first_index = last_list.index(0)
            last_index = len(last_list) - 1 - last_list[::-1].index(0)
        elif last_row[0] == 0:
            first_index = 0
        elif last_row[-1] == 0:
            last_index = dim_x - 1

        mid = (first_index + last_index) / 2

        sub_x = dim_x / len(state)

        # Set up state array
        state_idx = int(mid / sub_x)
        state[state_idx] = 1

        # Keep track of if car is out of the road
        if mid <= 1 or mid >= dim_x - 1:
            self.timeout += 1
        else:
            self.timeout = 0

        if self.timeout >= 30:
            done = True
            self.timeout = 0
        #
        # The state array is a list of 10 elements indicating where in the
        # image the line is:
        # i.e.
        #    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] indicates line is on the left
        #    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] indicates line is in the center
        #
        # The episode termination condition should be triggered when the line
        # is not detected for more than 30 frames. In this case set the done
        # variable to True.
        #
        # You can use the self.timeout variable to keep track of which frames
        # have no line detected.

        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        # Set the rewards for your action
        if not done:
            if action == 0:  # FORWARD
                reward = 4
            elif action == 1:  # LEFT
                reward = 2
            else:
                reward = 2  # RIGHT
        else:
            reward = -200

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state

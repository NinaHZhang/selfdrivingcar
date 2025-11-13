import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import time

class CatRacingEnv(gym.Env):
    """
    Custom environment for a cat racing on a track between two lines. Folloes gnymasium interface"""

    def __init__(self, render=False):
        super(CatRacingEnv, self).__init__()

        if render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT) #headless (faster training)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        # set physics timestep for stability
        p.setTimeStep(1.0/240.0)  # 240 hz physics simulation

        #define the action space [steering, throttle]
        #steering: -1 (left) to 1 (right)
        #throttle: 0(stopped) to 1 (full speed)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        #define observation space: 6 values
        # [left_dist, right_dist, center_offset, speed, steering, heading]
        #because continous space, spaces.box says observations are within these ranges
        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (6,), #1d array with 6 values
            dtype=np.float32
        )
        #track parameters
        self.track_width = 4.0 #width between boundaries 
        self.track_length = 200.0 #length of straight track (extended)

        #load environment
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(
            self.plane_id,
            -1,
            lateralFriction=1.0,
            restitution=0.0  # no bouncing on ground
        )
        self._create_track()
        self._create_cat()
        
        # set camera view to see full track
        if render:
            # camera position: further back and higher up to see full track
            # cameraDistance: how far camera is from target
            # cameraYaw: horizontal rotation
            # cameraPitch: vertical angle
            # cameraTargetPosition: what the camera is looking at
            p.resetDebugVisualizerCamera(
                cameraDistance=10.0,   # closer to see cat but still see track
                cameraYaw=90,         # look from the side
                cameraPitch=-30,       # angle down to see track
                cameraTargetPosition=[0, 0, 0]  # look at start of track where cat begins
            )
        #todo: define action and observation spaces
        #todo: connect to pybullet
        #todo: load track and the cat
    
    def _create_track(self):
        '''
        create track boundaries (just two parallel walls because im lazy)'''

        #left wall
        left_wall_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents = [self.track_length/2, 0.1, 0.5]
        )
        self.left_wall = p.createMultiBody(
            baseMass = 0, 
            baseCollisionShapeIndex = left_wall_shape,
            basePosition = [0, -self.track_width/2, 0.5]
        )

        # right wall
        right_wall_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.track_length/2, 0.1, 0.5]
        )
        self.right_wall = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=right_wall_shape,
            basePosition=[0, self.track_width/2, 0.5]
        )

    def _create_cat(self):
        """
        create the cat (simple box for now).
        """
        cat_visual = p.createVisualShape(
            p.GEOM_MESH,
            fileName="car.obj",
            meshScale=[0.05, 0.05, 0.05],
            
            rgbaColor=[1, 1, 1, 1]

        )

        cat_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents = [0.25, 0.15, 0.1] 
        )
        ''' # cat body (box shape)
        cat_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.5, 0.3, 0.2]  # length, width, height
        )'''
            
        # calculate proper z position: halfExtents z=0.1, so center should be at 0.1 (resting on ground at z=0)
        cat_z = 0.1
        self.cat_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=cat_collision,
            baseVisualShapeIndex=cat_visual,
            basePosition=[0, 0, cat_z],  # halfExtents z=0.1, so center at 0.1, bottom at 0.0
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])  # start facing forward (x direction)

        )
        p.changeDynamics(
            self.cat_id,
            -1,  # -1 means the base link
            lateralFriction=1.0,     # friction with ground
            spinningFriction=0.1,
            rollingFriction=0.01,
            linearDamping=0.1,       # air resistance (reduced to allow movement)
            angularDamping=0.1,
            restitution=0.0,         # no bouncing (0 = no bounce)
            contactStiffness=10000,  # stiffer contacts
            contactDamping=100       # damping for contacts
        )
        cat_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.25, 0.15, 0.1]  # smaller: length, width, height
        )

        '''baseMass=1.0,  # 1 kg
            baseCollisionShapeIndex=cat_collision,
            basePosition=[0, 0, 0.5],  # start at center of track
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])'''
    
        # make it blue so we can see it
        #p.changeVisualShape(self.cat_id, -1, rgbaColor=[0, 0, 1, 1])
    
        # store initial position for reset
        self.initial_pos = [0, 0, 0.1]  # halfExtents z=0.1, so center at 0.1
        self.initial_orn = p.getQuaternionFromEuler([0, 0, 0])
        
        # optional: visual markers
        p.changeVisualShape(self.left_wall, -1, rgbaColor=[1, 0, 0, 1])  # red
        p.changeVisualShape(self.right_wall, -1, rgbaColor=[1, 0, 0, 1])  # red

    def _get_observation(self):
        """
        get the current observation (state) of the cat.
        
        returns:
            observation - numpy array of 6 values:
            [left_dist, right_dist, center_offset, speed, steering, heading]
        """
        # get cat's position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.cat_id)
        
        # get cat's velocity
        linear_vel, angular_vel = p.getBaseVelocity(self.cat_id)
        
        # extract useful values
        x, y, z = pos
        
        # distance from center of track (y-axis)
        center_offset = y  # 0 = centered, + = right, - = left
        
        # distance to left and right walls
        left_dist = (self.track_width / 2) + y  # distance to left wall
        right_dist = (self.track_width / 2) - y  # distance to right wall
        
        # forward speed (velocity in x direction)
        speed = linear_vel[0]
        
        # steering/angular velocity (how fast turning)
        steering = angular_vel[2]  # rotation around z-axis
        
        # heading angle (orientation)
        euler = p.getEulerFromQuaternion(orn)
        heading = euler[2]  # yaw angle (rotation around z-axis)
        
        # return as numpy array
        observation = np.array([
            left_dist,
            right_dist,
            center_offset,
            speed,
            steering,
            heading
        ], dtype=np.float32)
        
        return observation

    def reset(self, seed=None):
        """
        reset the environment to initial state.
        
        returns:
            observation - initial observation
            info - empty dict (required by gymnasium)
        """
        # reset cat position and orientation
        p.resetBasePositionAndOrientation(
            self.cat_id,
            self.initial_pos,
            self.initial_orn
        )
        
        # reset cat velocity to zero
        p.resetBaseVelocity(
            self.cat_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0]
        )
        
        # let the cat settle on the ground (step simulation a few times)
        for _ in range(10):
            p.stepSimulation()
        
        # get initial observation
        observation = self._get_observation()
        
        # info dict (can add debugging info here later)
        info = {}
        
        return observation, info

    def step(self, action):
        """
        execute one step in the environment.
        
        parameters:
            action - [steering, throttle] from the agent
        
        returns:
            observation - new state after action
            reward - reward for this step
            terminated - whether episode ended (crashed)
            truncated - whether episode was cut off (max steps)
            info - extra diagnostic info
        """
        # extract actions
        steering = action[0]  # -1 to 1
        throttle = action[1]  # 0 to 1
        
        # apply forces to the cat
        # get current position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.cat_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]  # current heading
        
        # apply forward force based on throttle
        # apply force in local frame: forward is +x in local frame
        force_magnitude = throttle * 150.0  # scale throttle to force (increased for faster movement)
        
        # get current velocity to apply damping to vertical movement
        linear_vel, _ = p.getBaseVelocity(self.cat_id)
        
        # apply force at center of mass (local frame [0,0,0]) in forward direction (local +x)
        p.applyExternalForce(
            self.cat_id,
            -1,  # apply to base
            [force_magnitude, 0, 0],  # force in local x direction (forward)
            [0, 0, 0],  # apply at center of mass in local frame
            p.LINK_FRAME  # use local frame
        )
        
        # apply damping force to vertical velocity to prevent bouncing (only if moving significantly)
        if abs(linear_vel[2]) > 0.1:  # if moving vertically significantly
            damping_force_z = -linear_vel[2] * 30.0  # damping for vertical movement (reduced to not interfere with normal physics)
            p.applyExternalForce(
                self.cat_id,
                -1,
                [0, 0, damping_force_z],
                [0, 0, 0],
                p.WORLD_FRAME
            )
        
        # apply steering (torque around z-axis)
        torque_magnitude = steering * 20.0  # scale steering to torque (increased for better turning)
        p.applyExternalTorque(
            self.cat_id,
            -1,
            [0, 0, torque_magnitude],
            p.WORLD_FRAME
        )
        
        # step the simulation multiple times for smoother physics
        for _ in range(4):  # step 4 times per action for smoother movement
            p.stepSimulation()
        
        # get new observation
        observation = self._get_observation()
        
        # calculate reward
        reward = self._calculate_reward(observation)
        
        # check if episode is done
        terminated = self._check_terminated(observation)
        truncated = False  # we'll add step limit later if needed
        
        # extra info
        info = {}
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, observation):
        """
        calculate reward based on current observation.
        
        parameters:
            observation - current state [left_dist, right_dist, center_offset, speed, steering, heading]
        
        returns:
            reward - scalar reward value
        """
        left_dist, right_dist, center_offset, speed, steering, heading = observation
        
        reward = 0.0
        
        # reward for moving forward
        reward += speed * 0.1  # encourage speed
        
        # penalty for being off-center
        reward -= abs(center_offset) * 0.5  # encourage staying centered
        
        # big penalty for getting too close to walls
        if left_dist < 0.5 or right_dist < 0.5:
            reward -= 10.0
        
        # small penalty for excessive steering (encourage smooth driving)
        reward -= abs(steering) * 0.1
        
        return reward
    
    def _check_terminated(self, observation):
        """
        check if episode is done (cat went off track or crashed).
        
        parameters:
            observation - current state
        
        returns:
            done - boolean
        """
        left_dist, right_dist, center_offset, speed, steering, heading = observation
        
        # episode ends if cat goes off track (negative distance means past the wall)
        if left_dist < 0 or right_dist < 0:
            return True
        
        # episode ends if cat hits the wall (very close to wall, within collision distance)
        # cat halfExtents y = 0.15, so if distance < 0.15, it's colliding
        if left_dist < 0.15 or right_dist < 0.15:
            return True
        
        # episode ends if cat falls off (z position too low)
        pos, _ = p.getBasePositionAndOrientation(self.cat_id)
        if pos[2] < 0.0:  # fell through ground (cat center should be at ~0.1)
            return True
        
        # check if cat is in contact with walls using pybullet contact detection
        contacts = p.getContactPoints(bodyA=self.cat_id, bodyB=self.left_wall)
        if len(contacts) > 0:
            return True
        contacts = p.getContactPoints(bodyA=self.cat_id, bodyB=self.right_wall)
        if len(contacts) > 0:
            return True
        
        return False

    def close(self):
        '''clean up pybullet connection'''
        p.disconnect()

if __name__ == "__main__":
    env = CatRacingEnv(render=True)
    print("Environment created!")
    
    obs, info = env.reset()
    print("Initial observation:", obs)
    
    # take a few random steps with forward movement
    for i in range(1000):
        # sample random action but ensure throttle is high enough for forward movement
        # steering: random between -1 and 1
        # throttle: random between 0.3 and 1.0 (ensures forward movement)
        steering = np.random.uniform(-1.0, 1.0)
        throttle = np.random.uniform(0.3, 1.0)  # minimum 0.3 throttle to ensure movement
        action = np.array([steering, throttle], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # print every 10 steps to reduce output
        if i % 10 == 0:
            # get position from observation (center_offset is y, speed gives x direction)
            left_dist, right_dist, center_offset, speed, steering_vel, heading = obs
            print(f"Step {i}: Y={center_offset:.2f}, Speed={speed:.2f}, "
                  f"Reward={reward:.2f}, Action=[{steering:.2f}, {throttle:.2f}], Terminated={terminated}")

        time.sleep(0.01)  # reduced sleep for faster testing
        
        if terminated:
            print(f"Episode ended at step {i}!")
            obs, info = env.reset()
    
    input("Press Enter to close...")
    env.close()
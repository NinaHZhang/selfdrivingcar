import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

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
        self.track_length = 50.0 #length of straight track

        #load environment
        self.plane_id = p.loadURDF("plane.urdf")
        self._create_track()
        self._create_cat()
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

        # Right wall
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
        Create the cat (simple box for now).
        """
         # Cat body (box shape)
        cat_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.5, 0.3, 0.2]  # Length, width, height
        )
            
        self.cat_id = p.createMultiBody(
            baseMass=1.0,  # 1 kg
            baseCollisionShapeIndex=cat_shape,
            basePosition=[0, 0, 0.5],  # Start at center of track
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
    
        # Make it blue so we can see it
        p.changeVisualShape(self.cat_id, -1, rgbaColor=[0, 0, 1, 1])
    
        # Store initial position for reset
        self.initial_pos = [0, 0, 0.5]
        self.initial_orn = p.getQuaternionFromEuler([0, 0, 0])
        
        # Optional: Visual markers
        p.changeVisualShape(self.left_wall, -1, rgbaColor=[1, 0, 0, 1])  # Red
        p.changeVisualShape(self.right_wall, -1, rgbaColor=[1, 0, 0, 1])  # Red

    def rest(self, seed = None):
        '''resets the environment to initial state
        returns the initial observation
        '''
        
        #todo: reset cat position, velocity
        #todo: return initial observation
        pass

    def step(self, action):
        '''execute one step in the environment
        parameters: action - [steering, throttle]
        returns:
            observation - current state
            reward - reward for this step
            done - whether episode is finished
            truncated - whether episode was cut off
            info - extra diagnostic info
        '''
        #TODO: apply action to cat
        #TODO: step simulation
        #TODO: get new observation
        #TODO: calculate reward
        #TODO: check if done
        pass

    def close(self):
        '''clean up pybullet connection'''
        p.disconnect()

if __name__ == "__main__":
    env = CatRacingEnv(render=True)
    print("Environment created successfully!")
    input("Press Enter to close...")
    env.close()
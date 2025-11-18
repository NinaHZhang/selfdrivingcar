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
        self.track_width = 7.0 #width between boundaries
        self.track_length = 50.0  # total track length (long path)
        
        #finish line and start line positions
        self.start_line_x = 0.0  # start at x=0
        self.finish_line_x = self.track_length  # finish at end of track
        self.current_x = 0.0  # track current x position
        self.max_x_reached = 0.0  # track furthest x position reached

        #load environment
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(
            self.plane_id,
            -1,
            lateralFriction=1.0,
            restitution=0.0  # no bouncing on ground
        )
        self._create_track()
        self._create_start_finish_lines()
        self._create_cat()
        
        # set camera view to see full track
        if render:
            # camera position: further back and higher up to see full track
            # cameraDistance: how far camera is from target
            # cameraYaw: horizontal rotation
            # cameraPitch: vertical angle
            # cameraTargetPosition: what the camera is looking at
            # Camera positioned to see entire track
            track_center_x = self.track_length / 2  # center of track
            p.resetDebugVisualizerCamera(
                cameraDistance=30.0,   # further back to see whole track
                cameraYaw=90,         # look from the side
                cameraPitch=-40,       # angle down to see track better
                cameraTargetPosition=[track_center_x, 0, 0]  # look at center of track
            )
       
    
    def _create_track(self):
        '''
        Create a simple straight track from start to finish.
        '''
        
        # Store wall IDs for reference
        self.left_walls = []
        self.right_walls = []
        
        # Create simple straight walls
        wall_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.track_length / 2, 0.1, 0.5]
        )
        
        # Left wall (parallel to x-axis, offset to the left)
        left_wall = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_shape,
            basePosition=[self.track_length / 2, self.track_width / 2, 0.5]
        )
        self.left_walls.append(left_wall)
        
        # Right wall (parallel to x-axis, offset to the right)
        right_wall = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_shape,
            basePosition=[self.track_length / 2, -self.track_width / 2, 0.5]
        )
        self.right_walls.append(right_wall)
    
    def _create_start_finish_lines(self):
        """
        Create visual start and finish lines on the track.
        Make them clearly visible and distinct from walls.
        Also add a physical barrier at start line to prevent going backward.
        """
        # Start line (green) - at beginning of track (x=0, y=0)
        start_line_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.2, self.track_width/2 + 0.2, 0.5],  # wider, taller, more visible
            rgbaColor=[0, 1, 0, 1.0]  # bright green, fully opaque
        )
        self.start_line = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=start_line_shape,
            basePosition=[self.start_line_x, 0, 0.5]  # at start (0, 0)
        )
        
        # Add a COLLISION barrier at start line to prevent going backward
        start_barrier_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.1, self.track_width/2 + 0.5, 0.5]  # thin but tall barrier
        )
        self.start_barrier = p.createMultiBody(
            baseMass=0,  # static barrier
            baseCollisionShapeIndex=start_barrier_shape,
            basePosition=[self.start_line_x - 0.5, 0, 0.5]  # slightly behind start line
        )
        # Make barrier invisible (or semi-transparent for debugging)
        p.changeVisualShape(self.start_barrier, -1, rgbaColor=[0, 1, 0, 0.3])  # semi-transparent green
        
        # Finish line (bright red/yellow) - at end of track
        # Track ends at x=track_length, y=0 (back to center after turns)
        finish_line_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.3, self.track_width/2 + 0.3, 0.8],  # even wider and taller
            rgbaColor=[1, 0.8, 0, 1.0]  # bright yellow/orange, fully opaque - very distinct
        )
        self.finish_line = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=finish_line_shape,
            basePosition=[self.finish_line_x, 0, 0.8]  # at finish (track_length, 0)
        )
        
        # Add finish line markers (small boxes on sides) for extra visibility
        marker_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.2, 0.2, 1.0],
            rgbaColor=[1, 0, 0, 1.0]  # bright red markers
        )
        # Left marker
        self.finish_marker_left = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=marker_shape,
            basePosition=[self.finish_line_x, -self.track_width/2 - 0.5, 1.0]
        )
        # Right marker
        self.finish_marker_right = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=marker_shape,
            basePosition=[self.finish_line_x, self.track_width/2 + 0.5, 1.0]
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
            basePosition=[self.start_line_x, 0, cat_z],  # spawn at start line (0, 0)
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])  # start facing forward

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
    
        # store initial position for reset - spawn at start line
        self.initial_pos = [self.start_line_x, 0, 0.1]  # spawn at start line (0, 0)
        self.initial_orn = p.getQuaternionFromEuler([0, 0, 0])  # facing forward (+x direction)
        
        # optional: visual markers - color all wall segments red
        for wall in self.left_walls:
            p.changeVisualShape(wall, -1, rgbaColor=[1, 0, 0, 1])  # red
        for wall in self.right_walls:
            p.changeVisualShape(wall, -1, rgbaColor=[1, 0, 0, 1])  # red

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
        
        # For straight track, centerline is at y=0
        centerline_y = 0.0
        
        # Distance from centerline (0 = on centerline, + = right, - = left)
        center_offset = y - centerline_y
        
        # Distance to left and right walls (simplified - assumes walls follow curve)
        # For curved track, we use the track width and center offset
        left_dist = (self.track_width / 2) + center_offset  # distance to left wall
        right_dist = (self.track_width / 2) - center_offset  # distance to right wall
        
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
        
        # reset position tracking - start at start line
        self.current_x = self.start_line_x
        self.max_x_reached = self.start_line_x
        self.prev_max_x = self.start_line_x
        
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
        
        # store previous max before updating (for reward calculation)
        self.prev_max_x = self.max_x_reached
        
        # apply forward force based on throttle
        # IMPORTANT: Apply force in WORLD frame toward finish line (+x direction)
        # This ensures force always pushes toward finish, regardless of car rotation
        # Reduced force to make movement more visible and controllable
        force_magnitude = throttle * 70.0  # reduced from 100.0 to slow down acceleration
        
        # get current velocity to apply damping to vertical movement
        linear_vel, _ = p.getBaseVelocity(self.cat_id)
        
        # Apply force in WORLD frame toward finish line (+x direction)
        # This ensures the car always moves toward finish, even if rotated
        p.applyExternalForce(
            self.cat_id,
            -1,  # apply to base
            [force_magnitude, 0, 0],  # force in WORLD +x direction (toward finish line)
            [0, 0, 0],  # apply at center of mass in world frame
            p.WORLD_FRAME  # use WORLD frame - always push toward finish!
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
        # Reduced torque to prevent excessive spinning
        torque_magnitude = steering * 10.0  # reduced from 20.0 to prevent spinning in place
        p.applyExternalTorque(
            self.cat_id,
            -1,
            [0, 0, torque_magnitude],
            p.WORLD_FRAME
        )
        
        # step the simulation multiple times for smoother physics
        for _ in range(4):  # step 4 times per action for smoother movement
            p.stepSimulation()
        
        # IMPORTANT: Update position tracking AFTER simulation steps
        # Get the NEW position after physics simulation
        pos, _ = p.getBasePositionAndOrientation(self.cat_id)
        self.current_x = pos[0]  # update current x position
        if self.current_x > self.max_x_reached:
            self.max_x_reached = self.current_x
        
        # get new observation
        observation = self._get_observation()
        
        # calculate reward (includes progress and finish line rewards)
        reward = self._calculate_reward(observation)
        
        # check if episode is done (crash or finish line reached)
        terminated = self._check_terminated(observation)
        truncated = False  # we'll add step limit later if needed
        
        # extra info
        info = {
            'x_position': self.current_x,
            'max_x_reached': self.max_x_reached,
            'progress': self.max_x_reached / self.finish_line_x,
            'finished': terminated and self.current_x >= self.finish_line_x
        }
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, observation):
        """
        calculate reward based on current observation.
        SIMPLIFIED: Focus on positive rewards, minimal penalties.
        
        parameters:
            observation - current state [left_dist, right_dist, center_offset, speed, steering, heading]
        
        returns:
            reward - scalar reward value
        """
        left_dist, right_dist, center_offset, speed, steering, heading = observation
        
        reward = 0.0
        
        # BIG REWARD for reaching finish line
        if self.current_x >= self.finish_line_x:
            reward += 100.0  # large completion bonus
            return reward  # return immediately with completion reward
        
        # ===== POSITIVE REWARDS (encourage good behavior) =====
        
        # 1. Reward for forward speed - PRIMARY DRIVER
        forward_speed = max(0, speed)  # only reward positive forward speed
        reward += forward_speed * 1.5  # strong reward for moving forward
        
        # 2. Reward for distance from start (being further = better)
        distance_from_start = self.current_x - self.start_line_x
        reward += distance_from_start * 0.5  # linear reward for distance
        
        # 3. BONUS for making NEW progress (reaching further than ever before)
        if self.current_x > self.prev_max_x:
            new_progress = self.current_x - self.prev_max_x
            reward += new_progress * 3.0  # VERY strong bonus for exploration
        
        # 4. Distance-based reward (stronger as you get closer to finish)
        distance_to_finish = self.finish_line_x - self.current_x
        max_distance = self.finish_line_x - self.start_line_x
        if max_distance > 0:
            progress_ratio = 1.0 - (distance_to_finish / max_distance)  # 0 at start, 1 at finish
            distance_reward = progress_ratio * 5.0  # linear, max 5.0 at finish
            reward += distance_reward
        
        # 5. Small bonus for staying roughly centered (don't make this too important)
        if abs(center_offset) < self.track_width / 4:  # if within half of track width
            reward += 0.2  # small bonus
        
        # ===== MINIMAL PENALTIES (only for truly bad behavior) =====
        
        # Penalty for staying in one spot (not moving forward)
        if abs(speed) < 0.1:  # if barely moving
            reward -= 1.0  # penalty for standing still
        
        # Penalty for being near start line (encourage forward movement)
        if self.current_x < 2.0:  # if within 2 units of start
            distance_from_start = self.current_x - self.start_line_x
            if distance_from_start < 2.0:
                # Penalty increases the closer you are to start
                start_penalty = (2.0 - distance_from_start) * 0.5  # max -1.0 at start
                reward -= start_penalty
        
        # Only penalize if going WAY backward (past start line significantly)
        if self.current_x < self.start_line_x - 0.5:
            reward -= 1.0  # small penalty only
        
        # Only penalize if EXTREMELY close to wall (about to crash)
        min_dist = min(left_dist, right_dist)
        if min_dist < 0.1:  # only when almost touching wall (reduced threshold)
            reward -= 1.0  # reduced penalty (from 2.0)
        
        # NO penalty for:
        # - Spinning (let it learn naturally)
        # - Being off-center (not critical)
        # - Going backward slightly (might be necessary for learning)
        
        return reward
    
    def _check_terminated(self, observation):
        """
        check if episode is done (cat went off track, crashed, or reached finish line).
        
        parameters:
            observation - current state
        
        returns:
            done - boolean
        """
        # SUCCESS: episode ends if cat reaches finish line
        if self.current_x >= self.finish_line_x:
            print(f"SUCCESS! Cat reached finish line at x={self.current_x:.2f} (finish at {self.finish_line_x:.2f})")
            return True
        
        left_dist, right_dist, center_offset, speed, steering, heading = observation
        
        # episode ends if cat goes backward past start line (too far)
        if self.current_x < self.start_line_x - 1.0:  # if more than 1 unit behind start
            print(f"Terminated: Cat went too far backward (x={self.current_x:.2f}, start={self.start_line_x:.2f})")
            return True  # terminate - went too far backward
        
        # episode ends if cat goes WAY off track (negative distance means past the wall)
        # Only terminate if significantly past the wall (give some buffer)
        if left_dist < -0.2 or right_dist < -0.2:  # must be well past wall
            return True
        
        # episode ends if cat hits the wall (very close to wall, within collision distance)
        # cat halfExtents y = 0.15, so if distance < 0.15, it's colliding
        # But with wider track (10.0), we can be more lenient
        if left_dist < 0.1 or right_dist < 0.1:  # reduced from 0.15 to 0.1 (only terminate when really close)
            return True
        
        # episode ends if cat falls off (z position too low)
        pos, _ = p.getBasePositionAndOrientation(self.cat_id)
        if pos[2] < 0.0:  # fell through ground (cat center should be at ~0.1)
            return True
        
        # check if cat is in contact with walls using pybullet contact detection
        # Only terminate if there's significant contact (not just grazing)
        # Check all wall segments (since we now have multiple walls)
        total_contacts_left = 0
        total_contacts_right = 0
        
        for wall in self.left_walls:
            contacts = p.getContactPoints(bodyA=self.cat_id, bodyB=wall)
            total_contacts_left += len(contacts)
        
        for wall in self.right_walls:
            contacts = p.getContactPoints(bodyA=self.cat_id, bodyB=wall)
            total_contacts_right += len(contacts)
        
        # Only terminate if there are multiple contact points (actual collision, not just touch)
        if total_contacts_left > 2 or total_contacts_right > 2:
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
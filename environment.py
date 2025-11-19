import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import time
import sys
import importlib.util

# Import the procedural track
spec = importlib.util.spec_from_file_location("track", "track (1).py")
track_module = importlib.util.module_from_spec(spec)
sys.modules["track"] = track_module
spec.loader.exec_module(track_module)
Track = track_module.Track

class CatRacingEnv(gym.Env):
    """
    Custom environment for a cat racing on a procedural track. Follows gymnasium interface"""

    def __init__(self, render=False, track_config_path="track_config.yaml", track_seed=None):
        """
        Initialize environment.
        
        Parameters:
            render: bool - whether to render the environment
            track_config_path: str - path to track config file
            track_seed: int - random seed for track generation (None = random each time)
        """
        super(CatRacingEnv, self).__init__()

        if render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT) #headless (faster training)
            # Performance optimizations for headless mode
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # Disable rendering
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Disable GUI
            p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)  # Disable tiny renderer

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        # set physics timestep - reduced from 240 Hz to 120 Hz for faster training
        # Lower frequency = faster simulation, slightly less accurate but fine for RL
        p.setTimeStep(1.0/120.0)  # 120 hz physics simulation (was 240 hz)
        
        # Additional performance optimizations
        p.setPhysicsEngineParameter(
            numSolverIterations=10,  # Reduced from default 20 for speed
            enableFileCaching=0,  # Disable file caching
            deterministicOverlappingPairs=0  # Faster collision detection
        )

        #define the action space [steering, throttle]
        #steering: -1 (left) to 1 (right) - controls differential wheel speeds
        #throttle: 0 (stopped) to 1 (full speed) - controls wheel motor velocity
        #Actions are now wheel-based (realistic car controls)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        #define observation space: 9 values (added look-ahead features and heading)
        # [inner_dist, outer_dist, lateral_offset, speed, steering, progress, 
        #  curvature_ahead, distance_to_wall_ahead, heading]
        #because continous space, spaces.box says observations are within these ranges
        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (9,), #1d array with 9 values
            dtype=np.float32
        )

        #load environment
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(
            self.plane_id,
            -1,
            lateralFriction=1.0,
            restitution=0.0  # no bouncing on ground
        )
        
        # Load procedural track (with optional seed for reproducibility)
        self.track = Track(track_config_path, seed=track_seed)
        self.track.spawn_in_pybullet(self.client)
        self.inner_track_ids, self.outer_track_ids = self.track.get_track_ids()
        
        # Get centerline points for progress tracking
        self.centerline_points = self.track.inner_points.copy()  # Use inner points as reference
        # Actually, we need the centerline - let's compute it from inner/outer
        centerline_xy = (self.track.inner_points[:, :2] + self.track.outer_points[:, :2]) / 2.0
        self.centerline_points = np.column_stack([centerline_xy, np.full(len(centerline_xy), self.track.line_height)])
        
        # Track parameters
        self.track_width = self.track.track_width
        self.half_width = self.track.half_width
        
        # Precompute centerline arc lengths for progress tracking
        self._compute_centerline_arc_lengths()
        
        # Progress tracking along centerline
        self.total_centerline_length = self.centerline_arc_lengths[-1]
        
        # Progress tracking (full track, no segments)
        # Track absolute progress (not modulo'd) for proper reward calculation
        self.current_progress = 0.0  # Current progress along centerline (absolute arc length)
        self.max_progress = 0.0  # Maximum progress reached (absolute)
        self.prev_progress = 0.0  # Previous progress (absolute)
        self.laps_completed = 0  # Number of full laps completed
        self.cumulative_forward_progress = 0.0  # Total forward progress made (for lap detection)
        self.last_valid_progress = 0.0  # Last valid progress position
        self.min_translation_for_progress = 0.3  # minimum world displacement to count progress
        self.progress_diff_last = 0.0
        self.progress_diff_raw_last = 0.0
        self.progress_counted_last = False
        
        # Anti-spinning tracking
        self.last_positions = []  # Track recent positions to detect spinning
        self.max_position_history = 20  # Keep last 20 positions
        self.heading_history = []  # Track recent heading values
        self.progress_history = []  # Track progress values for spinning detection
        self._history_window = 30
        self.is_spinning_recent = False
        
        # Anti-spinning tracking
        self.last_positions = []  # Track recent positions to detect spinning
        self.max_position_history = 20  # Keep last 20 positions
        self._last_10_progress = []  # Track recent progress values to detect spinning
        
        # Spawn position: on the centerline, at the start (progress = 0)
        spawn_point = self.centerline_points[0]
        self.spawn_x = spawn_point[0]
        self.spawn_y = spawn_point[1]
        
        self._create_cat()
        self.last_valid_pos = np.array([self.spawn_x, self.spawn_y], dtype=np.float32)
        
        # set camera view to see full track
        if render:
            # Calculate track center for camera
            track_center = np.mean(self.centerline_points[:, :2], axis=0)
            p.resetDebugVisualizerCamera(
                cameraDistance=60.0,   # far enough to see whole track
                cameraYaw=45,          # angled view
                cameraPitch=-60,       # look down at track
                cameraTargetPosition=[track_center[0], track_center[1], 0]  # center of track
            )
    
    def _compute_centerline_arc_lengths(self):
        """Precompute cumulative arc lengths along centerline for progress tracking."""
        n = len(self.centerline_points)
        arc_lengths = np.zeros(n)
        for i in range(1, n):
            prev_point = self.centerline_points[i - 1, :2]
            curr_point = self.centerline_points[i, :2]
            segment_length = np.linalg.norm(curr_point - prev_point)
            arc_lengths[i] = arc_lengths[i - 1] + segment_length
        # Add final segment (closed loop)
        first_point = self.centerline_points[0, :2]
        last_point = self.centerline_points[-1, :2]
        final_segment = np.linalg.norm(first_point - last_point)
        arc_lengths = np.append(arc_lengths, arc_lengths[-1] + final_segment)
        self.centerline_arc_lengths = arc_lengths
    
    def _find_closest_centerline_point(self, car_pos_2d):
        """Find the closest point on centerline and return its index and distance."""
        centerline_xy = self.centerline_points[:, :2]
        distances = np.linalg.norm(centerline_xy - car_pos_2d, axis=1)
        closest_idx = np.argmin(distances)
        return closest_idx, distances[closest_idx]
    
    def _get_progress_along_centerline(self, car_pos_2d):
        """Get progress (arc length) along centerline for car position."""
        closest_idx, _ = self._find_closest_centerline_point(car_pos_2d)
        
        # Get the two segments around the closest point
        n = len(self.centerline_points)
        prev_idx = (closest_idx - 1) % n
        next_idx = (closest_idx + 1) % n
        
        # Find which segment the car is closest to
        segments = [
            (prev_idx, closest_idx),
            (closest_idx, next_idx)
        ]
        
        min_dist = float('inf')
        best_progress = self.centerline_arc_lengths[closest_idx]
        
        for start_idx, end_idx in segments:
            seg_start = self.centerline_points[start_idx, :2]
            seg_end = self.centerline_points[end_idx, :2]
            
            # Project car position onto segment
            seg_vec = seg_end - seg_start
            seg_len = np.linalg.norm(seg_vec)
            if seg_len < 1e-6:
                continue
            
            seg_unit = seg_vec / seg_len
            car_vec = car_pos_2d - seg_start
            proj_length = np.dot(car_vec, seg_unit)
            proj_length = np.clip(proj_length, 0, seg_len)
            proj_point = seg_start + seg_unit * proj_length
            
            dist = np.linalg.norm(car_pos_2d - proj_point)
            if dist < min_dist:
                min_dist = dist
                # Calculate progress along this segment
                base_progress = self.centerline_arc_lengths[start_idx]
                segment_progress = proj_length
                best_progress = base_progress + segment_progress
        
        return best_progress
    
    def _get_centerline_tangent(self, progress):
        """Get tangent direction at given progress along centerline."""
        # Find segment containing this progress
        n = len(self.centerline_arc_lengths) - 1
        progress = progress % self.total_centerline_length
        
        for i in range(n):
            if self.centerline_arc_lengths[i] <= progress < self.centerline_arc_lengths[i + 1]:
                start_idx = i % len(self.centerline_points)
                end_idx = (i + 1) % len(self.centerline_points)
                start_point = self.centerline_points[start_idx, :2]
                end_point = self.centerline_points[end_idx, :2]
                tangent = end_point - start_point
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm > 1e-6:
                    return tangent / tangent_norm
        
        # Default: use first segment
        tangent = self.centerline_points[1, :2] - self.centerline_points[0, :2]
        return tangent / np.linalg.norm(tangent)
    
    def _get_centerline_normal(self, progress):
        """Get normal direction (pointing outward) at given progress."""
        tangent = self._get_centerline_tangent(progress)
        # Normal is perpendicular to tangent (rotate 90 degrees counterclockwise)
        return np.array([-tangent[1], tangent[0]])

    def _get_heading_from_orientation(self, orientation):
        """Convert quaternion orientation to normalized heading (yaw) angle."""
        yaw = p.getEulerFromQuaternion(orientation)[2]
        return np.arctan2(np.sin(yaw), np.cos(yaw))

    def _angle_difference(self, angle_a, angle_b):
        """Compute smallest difference between two angles."""
        diff = angle_a - angle_b
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff

    def _progress_difference(self, current, previous):
        """Compute wrapped difference along track length."""
        diff = current - previous
        if diff < -self.total_centerline_length / 2:
            diff += self.total_centerline_length
        elif diff > self.total_centerline_length / 2:
            diff -= self.total_centerline_length
        return diff
       
    

    def _create_cat(self):
        """
        Create the car from URDF file with realistic wheel controls.
        """
        # Spawn on centerline at start (progress = 0), facing along the track tangent
        spawn_progress = 0.0
        tangent = self._get_centerline_tangent(spawn_progress)
        initial_yaw = np.arctan2(tangent[1], tangent[0])  # Angle of tangent direction
        
        # Load URDF car with scaling to make it bigger relative to track
        # Original car: 0.25 x 0.125 x 0.075 (length x width x height)
        # Scale factor: 6x makes car ~1.5 x 0.75 x 0.45 (more visible on track)
        self.car_scale = 6.0
        # Car body height after scaling: 0.075 * 6 = 0.45, so center at z = 0.225
        # Wheels are at z=-0.03 * 6 = -0.18, so raise car so wheels touch ground
        car_z = 0.225 + 0.18  # Raise scaled car so wheels are at ground level
        self.cat_id = p.loadURDF(
            "car.urdf",
            basePosition=[self.spawn_x, self.spawn_y, car_z],
            baseOrientation=p.getQuaternionFromEuler([0, 0, initial_yaw]),
            globalScaling=self.car_scale,  # Scale the entire car model
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )
        
        # Get wheel joint indices (wheels rotate around Y-axis)
        num_joints = p.getNumJoints(self.cat_id)
        self.wheel_joints = []
        self.wheel_names = []
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.cat_id, i)
            joint_name = joint_info[1].decode('utf-8')
            if 'wheel' in joint_name.lower():
                self.wheel_joints.append(i)
                self.wheel_names.append(joint_name)
        
        # Separate front and rear wheels for differential steering
        self.front_wheel_joints = [i for i, name in zip(self.wheel_joints, self.wheel_names) if 'front' in name.lower()]
        self.rear_wheel_joints = [i for i, name in zip(self.wheel_joints, self.wheel_names) if 'rear' in name.lower()]
        self.left_wheel_joints = [i for i, name in zip(self.wheel_joints, self.wheel_names) if 'left' in name.lower()]
        self.right_wheel_joints = [i for i, name in zip(self.wheel_joints, self.wheel_names) if 'right' in name.lower()]
        
        # Configure wheel friction and dynamics
        for wheel_joint in self.wheel_joints:
            # Enable motor control for wheels
            p.setJointMotorControl2(
                self.cat_id,
                wheel_joint,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )
            # Set wheel friction
            p.changeDynamics(
                self.cat_id,
                wheel_joint,
                lateralFriction=1.2,  # High friction for traction
                spinningFriction=0.1,
                rollingFriction=0.01,
                restitution=0.0
            )
        
        # Configure car body dynamics
        p.changeDynamics(
            self.cat_id,
            -1,  # base link
            lateralFriction=0.7,
            spinningFriction=0.1,
            rollingFriction=0.01,
            linearDamping=0.1,      # Some air resistance
            angularDamping=0.1,     # Some angular damping
            restitution=0.0,
            contactStiffness=10000,
            contactDamping=100
        )
        
        # Car control parameters (scaled for larger car)
        self.max_motor_force = 60.0  # Increased motor force for larger car (proportional to scale^2)
        self.max_steering_angle = 0.5  # Maximum steering angle in radians (~30 degrees)
        self.wheel_radius = 0.0375 * self.car_scale  # Scaled wheel radius from URDF
        
        # store initial position for reset
        self.initial_pos = [self.spawn_x, self.spawn_y, car_z]
        self.initial_orn = p.getQuaternionFromEuler([0, 0, initial_yaw])

    def _get_observation(self):
        """
        get the current observation (state) of the car for procedural track.
        
        returns:
            observation - numpy array of 9 values:
            [inner_dist, outer_dist, lateral_offset, speed, steering, progress,
             curvature_ahead, min_dist_ahead, heading]
        """
        # get car's position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.cat_id)
        
        # get car's velocity
        linear_vel, angular_vel = p.getBaseVelocity(self.cat_id)
        
        # extract useful values
        x, y, z = pos
        car_pos_2d = np.array([x, y])
        
        # Convert orientation quaternion to heading angle (yaw)
        # Quaternion to Euler angles: [roll, pitch, yaw]
        heading = self._get_heading_from_orientation(orn)
        
        # Get progress along centerline
        progress = self._get_progress_along_centerline(car_pos_2d)
        
        # Get centerline tangent and normal at this progress
        tangent = self._get_centerline_tangent(progress)
        normal = self._get_centerline_normal(progress)
        
        # Find closest point on centerline
        closest_idx, _ = self._find_closest_centerline_point(car_pos_2d)
        centerline_point = self.centerline_points[closest_idx, :2]
        
        # Calculate lateral offset (distance perpendicular to centerline)
        car_to_centerline = car_pos_2d - centerline_point
        lateral_offset = np.dot(car_to_centerline, normal)  # positive = outside, negative = inside
        
        # Distance to inner and outer walls
        # Inner wall is at -half_width, outer wall is at +half_width
        inner_dist = self.half_width + lateral_offset  # positive = outside inner wall
        outer_dist = self.half_width - lateral_offset  # positive = inside outer wall
        
        # Calculate tangential speed (speed along the track)
        velocity_2d = np.array([linear_vel[0], linear_vel[1]])
        speed = np.dot(velocity_2d, tangent)  # Project velocity onto tangent
        
        # steering/angular velocity (how fast turning)
        steering = angular_vel[2]  # rotation around z-axis
        
        # Normalize progress to [0, 1] for observation (0 = start, 1 = one full lap)
        normalized_progress = (progress % self.total_centerline_length) / self.total_centerline_length
        
        # ===== LOOK-AHEAD FEATURES (help predict upcoming turns) =====
        
        # Look ahead along the track (predict upcoming curvature)
        look_ahead_distance = 2.0  # Look 2 units ahead (adjust based on car speed)
        look_ahead_progress = (progress + look_ahead_distance) % self.total_centerline_length
        
        # Get tangent direction ahead
        tangent_ahead = self._get_centerline_tangent(look_ahead_progress)
        
        # Calculate curvature (how much the track is turning)
        # Curvature = change in tangent direction
        curvature = np.arctan2(
            tangent_ahead[1] - tangent[1],
            tangent_ahead[0] - tangent[0]
        )  # Angle difference between current and ahead tangent
        # Normalize to [-1, 1] range
        curvature = np.clip(curvature / np.pi, -1.0, 1.0)
        
        # Get minimum distance to walls ahead (predict if we'll hit a wall)
        # Project car position forward along current direction
        car_forward_dir = np.array([np.cos(np.arctan2(tangent[1], tangent[0])), 
                                    np.sin(np.arctan2(tangent[1], tangent[0]))])
        look_ahead_pos = car_pos_2d + car_forward_dir * look_ahead_distance
        
        # Find closest point on centerline ahead
        closest_idx_ahead, _ = self._find_closest_centerline_point(look_ahead_pos)
        centerline_point_ahead = self.centerline_points[closest_idx_ahead, :2]
        normal_ahead = self._get_centerline_normal(look_ahead_progress)
        
        # Calculate lateral offset ahead
        car_to_centerline_ahead = look_ahead_pos - centerline_point_ahead
        lateral_offset_ahead = np.dot(car_to_centerline_ahead, normal_ahead)
        
        # Distance to walls ahead
        inner_dist_ahead = self.half_width + lateral_offset_ahead
        outer_dist_ahead = self.half_width - lateral_offset_ahead
        min_dist_ahead = min(inner_dist_ahead, outer_dist_ahead)
        
        # return as numpy array
        observation = np.array([
            inner_dist,
            outer_dist,
            lateral_offset,
            speed,
            steering,
            normalized_progress,
            curvature,  # How much the track curves ahead
            min_dist_ahead,  # Minimum distance to walls ahead
            heading  # Car's heading angle (yaw) in radians, normalized to [-pi, pi]
        ], dtype=np.float32)
        
        return observation

    def reset(self, seed=None):
        """
        reset the environment to initial state.
        
        returns:
            observation - initial observation
            info - empty dict (required by gymnasium)
        """
        # reset car position and orientation
        p.resetBasePositionAndOrientation(
            self.cat_id,
            self.initial_pos,
            self.initial_orn
        )
        
        # reset car velocity to zero
        p.resetBaseVelocity(
            self.cat_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0]
        )
        
        # Reset all wheel velocities to zero
        for wheel_joint in self.wheel_joints:
            p.setJointMotorControl2(
                self.cat_id,
                wheel_joint,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )
            # Reset joint state
            p.resetJointState(self.cat_id, wheel_joint, targetValue=0, targetVelocity=0)
        
        # let the car settle on the ground (step simulation a few times)
        # Reduced from 10 to 5 for faster resets
        for _ in range(5):
            p.stepSimulation()
        
        # reset centerline progress tracking
        self.current_progress = 0.0
        self.max_progress = 0.0
        self.prev_progress = 0.0
        self.laps_completed = 0
        self.cumulative_forward_progress = 0.0
        self.last_valid_progress = 0.0
        self.last_valid_pos = np.array([self.spawn_x, self.spawn_y], dtype=np.float32)
        self.progress_diff_last = 0.0
        self.progress_diff_raw_last = 0.0
        self.progress_counted_last = False
        
        # Reset anti-spinning tracking
        self.last_positions = []
        self.heading_history = []
        self.progress_history = []
        self.is_spinning_recent = False
        
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
        
        # store previous progress before updating (for reward calculation)
        self.prev_progress = self.current_progress
        
        # Get car position and current progress
        pos, orn = p.getBasePositionAndOrientation(self.cat_id)
        x, y = pos[0], pos[1]
        car_pos_2d = np.array([x, y])
        current_progress = self._get_progress_along_centerline(car_pos_2d)
        
        # Get current velocity for speed calculation
        linear_vel, angular_vel = p.getBaseVelocity(self.cat_id)
        
        # ===== REALISTIC CAR CONTROLS =====
        
        # 1. THROTTLE: Convert throttle [0, 1] to target wheel velocity
        # Max wheel velocity in rad/s (adjust for desired max speed)
        # With scaled wheel radius, adjust for proportional speed
        max_wheel_velocity = 15.0  # rad/s (same angular speed, but larger linear speed due to scaling)
        base_wheel_velocity = throttle * max_wheel_velocity
        
        # 2. STEERING: Differential steering (different speeds on left/right wheels)
        # This creates turning by making one side go faster than the other
        # Increased steering factor for more responsive turning
        steering_factor = steering * 0.5  # Increased from 0.3 to 0.5 for more responsive steering
        
        # Calculate target velocities for left and right wheels
        # Left wheels: slower when steering right (positive), faster when steering left (negative)
        # Right wheels: faster when steering right (positive), slower when steering left (negative)
        left_target_velocity = base_wheel_velocity * (1.0 - steering_factor)
        right_target_velocity = base_wheel_velocity * (1.0 + steering_factor)
        
        # Apply motor control to wheels with differential steering
        for wheel_joint in self.left_wheel_joints:
            p.setJointMotorControl2(
            self.cat_id,
                wheel_joint,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=left_target_velocity,
                force=self.max_motor_force
            )
        
        for wheel_joint in self.right_wheel_joints:
            p.setJointMotorControl2(
                self.cat_id,
                wheel_joint,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=right_target_velocity,
                force=self.max_motor_force
            )
        
        # 3. Additional steering torque for sharper turns (helps with tight corners)
        # Apply stronger steering torque for more responsive turning
        steering_torque = steering * 5.0  # Increased from 2.0 to 5.0 for more responsive steering
        p.applyExternalTorque(
            self.cat_id,
            -1,  # base link
            [0, 0, steering_torque],
            p.WORLD_FRAME
        )
        
        # step the simulation - reduced from 4 to 2 for faster training
        # Fewer steps = faster training, still stable enough for RL
        for _ in range(2):  # step 2 times per action (was 4)
            p.stepSimulation()
        
        # IMPORTANT: Update progress tracking AFTER simulation steps
        # Get the NEW position and orientation after physics simulation
        pos, orn = p.getBasePositionAndOrientation(self.cat_id)
        x, y = pos[0], pos[1]
        car_pos_2d = np.array([x, y])
        new_heading = self._get_heading_from_orientation(orn)
        
        # Calculate current progress along centerline
        new_progress = self._get_progress_along_centerline(car_pos_2d)
        
        # Get car's velocity/orientation after movement
        linear_vel, angular_vel = p.getBaseVelocity(self.cat_id)
        velocity_2d = np.array([linear_vel[0], linear_vel[1]])
        velocity_magnitude = np.linalg.norm(velocity_2d)
        angular_velocity_magnitude = abs(angular_vel[2])
        new_heading = self._get_heading_from_orientation(orn)
        
        # Get the track tangent at current position to check forward direction
        tangent = self._get_centerline_tangent(new_progress)
        forward_velocity = np.dot(velocity_2d, tangent)  # Positive = forward, negative = backward
        
        # Handle wrap-around (progress can exceed total_centerline_length)
        # Calculate raw progress change
        progress_diff = new_progress - self.current_progress
        half_track = self.total_centerline_length / 2.0
        
        crossed_start_forward = (progress_diff < -half_track)
        crossed_start_backward = (progress_diff > half_track)
        
        if crossed_start_forward:
            if forward_velocity > 0.5:
                # Legitimate forward lap completion
                progress_diff += self.total_centerline_length
            else:
                # Moving backward - keep negative diff
                pass
        elif crossed_start_backward:
            if forward_velocity < -0.5:
                # Moving backward across start - make it more negative
                progress_diff -= self.total_centerline_length
            else:
                # Moving forward but small diff - leave as is
                pass
        
        # Update current progress (keep absolute, not modulo'd)
        self.current_progress = new_progress
        
        # Track position, heading, and progress history to detect spinning
        self.last_positions.append(car_pos_2d.copy())
        if len(self.last_positions) > self.max_position_history:
            self.last_positions.pop(0)
        
        self.heading_history.append(new_heading)
        if len(self.heading_history) > self._history_window:
            self.heading_history.pop(0)
        
        self.progress_history.append(new_progress)
        if len(self.progress_history) > self._history_window:
            self.progress_history.pop(0)
        
        # Detect if car is spinning in circles using multiple heuristics
        is_spinning = False
        progress_window = 0.0
        if len(self.progress_history) >= 2:
            window_index = max(0, len(self.progress_history) - 10)
            reference_progress = self.progress_history[window_index]
            progress_window = abs(self._progress_difference(new_progress, reference_progress))
        
        if len(self.last_positions) >= 10:
            position_variance = (
                np.var([p[0] for p in self.last_positions[-10:]]) +
                np.var([p[1] for p in self.last_positions[-10:]])
            )
            if position_variance > 1.0 and progress_window < self.total_centerline_length * 0.02:
                is_spinning = True
        
        if len(self.heading_history) >= 5:
            heading_change = 0.0
            start_idx = max(1, len(self.heading_history) - 10)
            for i in range(start_idx, len(self.heading_history)):
                heading_change += abs(self._angle_difference(self.heading_history[i], self.heading_history[i - 1]))
            if heading_change > 3 * np.pi and progress_window < self.total_centerline_length * 0.05:
                is_spinning = True
        
        if velocity_magnitude > 0.5:
            forward_ratio = abs(forward_velocity) / (velocity_magnitude + 1e-6)
            if forward_ratio < 0.2 and angular_velocity_magnitude > 1.5:
                is_spinning = True

        self.is_spinning_recent = is_spinning
        
        # STRICT: Only count progress if ALL conditions are met:
        # 1. Progress increased (forward along track)
        # 2. Change is VERY small (not a jump from spinning) - max 2% of track
        # 3. Car is actually moving forward (positive forward velocity > 1.0) - MUCH stricter
        # 4. Car has moved (velocity magnitude > 0.5) - MUCH stricter
        # 5. NOT spinning (position variance check)
        actual_forward_movement = (progress_diff > 0 and 
                                   progress_diff < self.total_centerline_length * 0.02 and  # VERY small steps only (2% max)
                                   forward_velocity > 1.0 and  # Must be moving forward strongly (increased from 0.2)
                                   velocity_magnitude > 0.5 and  # Must be moving (increased from 0.2)
                                   not is_spinning)  # NOT spinning in circles
        
        displacement = np.linalg.norm(car_pos_2d - self.last_valid_pos)
        progress_counted = False

        if actual_forward_movement and displacement >= self.min_translation_for_progress:
            # Valid forward progress - count it (sufficient translation)
            self.cumulative_forward_progress += progress_diff
            self.last_valid_progress = new_progress
            self.last_valid_pos = car_pos_2d.copy()
            progress_counted = True
        elif actual_forward_movement and displacement < self.min_translation_for_progress:
            # Treat as insufficient movement - do not count progress
            actual_forward_movement = False
        elif is_spinning:
            # SPINNING - reset cumulative progress to prevent false lap detection
            # Heavily penalize spinning by reducing cumulative progress
            self.cumulative_forward_progress = max(0.0, self.cumulative_forward_progress - abs(progress_diff) * 5.0)  # Strong penalty for spinning
        elif progress_diff < 0 or forward_velocity < 0:
            # Going backward or moving backward - penalize but don't reset
            pass
        # If progress_diff is large (>2% of track) or car isn't moving forward, ignore it (likely spinning)
        
        # Detect lap completion: MUST have cumulative forward progress >= 90% of track
        # AND be near start line (to prevent false positives from spinning)
        lap_just_completed = False
        normalized_progress = (new_progress % self.total_centerline_length) / self.total_centerline_length
        
        # Only count lap if we've actually moved forward 90%+ of the track length
        self.progress_counted_last = progress_counted
        self.progress_diff_raw_last = progress_diff
        self.progress_diff_last = progress_diff if progress_counted else 0.0

        if (self.cumulative_forward_progress >= self.total_centerline_length * 0.9 and 
            normalized_progress < 0.15):  # Near start line (within 15%)
            # Valid lap completion - we actually moved forward around the track
            self.laps_completed += 1
            lap_just_completed = True
            self.cumulative_forward_progress = 0.0  # Reset for next lap
            print(f"Lap completed! Total laps: {self.laps_completed} (cumulative progress: {self.cumulative_forward_progress:.2f})")
        
        # Track maximum progress reached
        if new_progress > self.max_progress:
            self.max_progress = new_progress
        
        # get new observation
        observation = self._get_observation()
        
        # calculate reward (includes progress and lap completion rewards)
        reward = self._calculate_reward(observation)
        
        # BIG REWARD for completing a lap (give reward immediately after detection)
        if lap_just_completed:
            reward += 500.0  # MUCH larger completion bonus (increased from 100.0)
        
        # check if episode is done (crash)
        terminated = self._check_terminated(observation)
        
        truncated = False  # we'll add step limit later if needed
        
        # LARGE PENALTY for crashing
        if terminated:
            reward -= 100.0  # Very large penalty for crashing
        
        # extra info (normalize progress for display)
        normalized_progress = (self.current_progress % self.total_centerline_length) / self.total_centerline_length
        max_normalized_progress = (self.max_progress % self.total_centerline_length) / self.total_centerline_length
        
        info = {
            'progress': self.current_progress,
            'max_progress': self.max_progress,
            'laps_completed': self.laps_completed,
            'progress_ratio': normalized_progress,  # normalized progress (0-1)
            'max_progress_ratio': max_normalized_progress,  # normalized max progress (0-1)
            'finished': terminated,
            'progress_counted': progress_counted,
            'progress_displacement': float(displacement),
            'progress_threshold': self.min_translation_for_progress,
            'is_spinning': self.is_spinning_recent,
            'forward_velocity': float(forward_velocity),
            'velocity_magnitude': float(velocity_magnitude)
        }
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, observation):
        """
        Simplified reward function focused on track completion.
        Continuous rewards for progress and speed, minimal competing signals.
        
        parameters:
            observation - current state [inner_dist, outer_dist, lateral_offset, speed, steering, progress, curvature_ahead, min_dist_ahead, heading]
        
        returns:
            reward - scalar reward value
        """
        inner_dist, outer_dist, lateral_offset, speed, steering, normalized_progress, curvature_ahead, min_dist_ahead, heading = observation
        
        reward = 0.0
        
        # ===== PRIMARY REWARD: Forward Progress (validated, continuous) =====
        linear_vel, _ = p.getBaseVelocity(self.cat_id)
        velocity_2d = np.array([linear_vel[0], linear_vel[1]])
        velocity_magnitude = np.linalg.norm(velocity_2d)
        tangent = self._get_centerline_tangent(self.current_progress)
        forward_velocity = np.dot(velocity_2d, tangent)
        _, angular_vel = p.getBaseVelocity(self.cat_id)
        angular_velocity_magnitude = abs(angular_vel[2])
        is_spinning = getattr(self, 'is_spinning_recent', False)
        
        progress_step = self.progress_diff_last / self.total_centerline_length
        raw_progress_step = self.progress_diff_raw_last / self.total_centerline_length
        
        if self.progress_counted_last and progress_step > 0:
            alignment = np.dot(velocity_2d, tangent) / (velocity_magnitude + 1e-6)
            alignment = max(0.0, alignment)
            reward += progress_step * 500.0 * alignment
        else:
            reward -= 1.0  # failed to make validated progress
        
        if raw_progress_step < -1e-3:
            reward -= abs(raw_progress_step) * 200.0
        
        if is_spinning:
            reward -= 50.0
        
        # Reward forward speed ONLY if moving forward and not spinning
        if forward_velocity > 1.0 and not is_spinning:
            reward += forward_velocity * 10.0
        elif forward_velocity < -0.2:
            reward += forward_velocity * 20.0
        elif velocity_magnitude < 0.5 or is_spinning:
            reward -= 2.0
        
        # ===== CENTERING REWARD: Encourage staying in the middle of the track =====
        # Reward for being centered (lateral_offset close to 0)
        # Track width is 2 * half_width, so lateral_offset ranges from -half_width to +half_width
        # Normalize to [-1, 1] range where 0 = perfectly centered
        normalized_lateral = lateral_offset / self.half_width if self.half_width > 0 else 0.0
        normalized_lateral = np.clip(normalized_lateral, -1.0, 1.0)
        
        # Reward for being centered: maximum reward at center (0), decreasing as you move away
        # Use a smooth curve (1 - abs(normalized_lateral)^2) so it's not too harsh
        centering_reward = (1.0 - abs(normalized_lateral) ** 2) * 2.0  # Max 2.0 at center, 0.0 at edges
        reward += centering_reward
        
        # ===== PROGRESSIVE WALL PENALTY: Discourage wall tracing =====
        min_dist = min(inner_dist, outer_dist)
        
        # Progressive penalty as you get closer to walls (not just when very close)
        # This discourages wall tracing without being too harsh
        if min_dist < 0.5:  # Within 0.5 units of a wall
            # Progressive penalty: -10.0 at 0.0 (touching wall), 0.0 at 0.5 (safe distance)
            # Smooth curve so it's not too harsh
            wall_penalty = -10.0 * (1.0 - min_dist / 0.5) ** 2  # Quadratic penalty (smoother)
            reward += wall_penalty
        elif min_dist < 1.0:  # Within 1.0 units of a wall (warning zone)
            # Small penalty to encourage staying away from walls
            wall_penalty = -1.0 * (1.0 - min_dist / 1.0)  # Linear penalty: -1.0 at 0.5, 0.0 at 1.0
            reward += wall_penalty
        
        # Penalty for going off track (outside boundaries)
        if inner_dist < 0 or outer_dist < 0:  # outside track boundaries
            reward -= 20.0  # Stronger penalty for going off track (increased from 10.0)
        
        # Additional penalty if actually touching/colliding with wall
        if min_dist < 0.1:  # Very close to wall (collision imminent)
            reward -= 5.0  # Additional penalty for being too close
        
        # Note: Lap completion reward (+100) is handled in step() function after lap detection
        # This keeps the reward function focused on continuous progress
        
        return reward
    
    def _check_terminated(self, observation):
        """
        check if episode is done (car went off track, crashed, or completed lap).
        
        parameters:
            observation - current state [inner_dist, outer_dist, lateral_offset, speed, steering, progress, curvature_ahead, min_dist_ahead, heading]
        
        returns:
            done - boolean
        """
        inner_dist, outer_dist, lateral_offset, speed, steering, normalized_progress, curvature_ahead, min_dist_ahead, heading = observation
        
        # SUCCESS: episode ends if car completes a lap (optional - can be success state)
        # For now, we'll let it continue after completing laps
        # Uncomment below if you want to terminate on lap completion:
        # if self.laps_completed >= 1:
        #     print(f"SUCCESS! Car completed {self.laps_completed} lap(s)!")
        #     return True
        
        # episode ends if car goes WAY off track (negative distance means past the wall)
        # Only terminate if significantly past the wall (give some buffer)
        if inner_dist < -0.2 or outer_dist < -0.2:  # must be well past wall
            # Car went off track (collision detected)
            return True
        
        # episode ends if car hits the wall (very close to wall, within collision distance)
        if inner_dist < 0.1 or outer_dist < 0.1:  # only terminate when really close
            # Car hit wall (collision detected)
            return True
        
        # episode ends if car falls off (z position too low)
        pos, _ = p.getBasePositionAndOrientation(self.cat_id)
        if pos[2] < 0.0:  # fell through ground
            # Car fell off (collision detected)
            return True
        
        # Removed wall collision termination - let the car learn from near-misses
        # Only terminate on going off track or hitting walls (distance-based checks above)
        
        return False

    def close(self):
        '''clean up pybullet connection'''
        p.disconnect()

if __name__ == "__main__":
    env = CatRacingEnv(render=True, track_config_path="track_config.yaml")
    print("Environment created with procedural track!")
    
    obs, info = env.reset()
    print("Initial observation:", obs)
    print("Observation format: [inner_dist, outer_dist, lateral_offset, speed, steering, progress, curvature_ahead, min_dist_ahead, heading]")
    
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
            # get position from observation (new format for procedural track)
            inner_dist, outer_dist, lateral_offset, speed, steering_vel, progress, curvature_ahead, min_dist_ahead, heading = obs
            print(f"Step {i}: Progress={progress:.3f} ({progress*100:.1f}%), Speed={speed:.2f}, "
                  f"Curvature={curvature_ahead:.2f}, DistAhead={min_dist_ahead:.2f}, "
                  f"Laps={info['laps_completed']}, Reward={reward:.2f}, "
                  f"Action=[{steering:.2f}, {throttle:.2f}], Terminated={terminated}")

        time.sleep(0.01)  # reduced sleep for faster testing
        
        if terminated:
            print(f"Episode ended at step {i}!")
            obs, info = env.reset()
    
    input("Press Enter to close...")
    env.close()
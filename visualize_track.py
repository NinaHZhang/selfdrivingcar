"""
Simple script to visualize the track in PyBullet GUI.
"""
import pybullet as p
import pybullet_data
import time
import sys
import importlib.util
import numpy as np

# Import the procedural track (track (1).py)
spec = importlib.util.spec_from_file_location("track", "track (1).py")
track_module = importlib.util.module_from_spec(spec)
sys.modules["track"] = track_module
spec.loader.exec_module(track_module)
Track = track_module.Track

def visualize_track(config_path="track_config.yaml", seed=None, segment_start=0.0, segment_end=None):
    """
    Load and visualize the track in PyBullet GUI.
    
    Parameters:
        config_path: str - path to track config file
        seed: int - random seed for track generation (None = random)
        segment_start: float - start of segment to highlight (0.0 to 1.0)
        segment_end: float - end of segment to highlight (None = full track)
    """
    # Connect to PyBullet with GUI
    physics_client = p.connect(p.GUI)
    
    # Set up physics
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1.0/240.0)
    
    # Load ground plane
    plane_id = p.loadURDF("plane.urdf")
    
    # Load and spawn track
    print(f"Loading track from {config_path}...")
    if seed is not None:
        print(f"Using seed: {seed}")
    track = Track(config_path, seed=seed)
    track.spawn_in_pybullet(physics_client)
    
    # Calculate track center for camera
    centerline_xy = (track.inner_points[:, :2] + track.outer_points[:, :2]) / 2.0
    track_center = [np.mean(centerline_xy[:, 0]), np.mean(centerline_xy[:, 1]), 0]
    
    # Set camera to view the track nicely
    p.resetDebugVisualizerCamera(
        cameraDistance=50.0,      # Distance from target
        cameraYaw=45,              # Horizontal rotation
        cameraPitch=-60,           # Look down at an angle
        cameraTargetPosition=track_center  # Center of track
    )
    
    # Visualize segment start/end if specified
    if segment_end is not None:
        total_length = np.sum(np.linalg.norm(
            np.diff(centerline_xy, axis=0, prepend=centerline_xy[-1:]), axis=1
        ))
        segment_start_abs = segment_start * total_length
        segment_end_abs = segment_end * total_length
        
        # Find segment start/end points (simplified - would need proper arc length calculation)
        print(f"Segment: {segment_start:.1%} to {segment_end:.1%}")
    
    print("Track loaded! Close the PyBullet window to exit.")
    print("You can use mouse to rotate/zoom the view.")
    
    # Keep the window open
    try:
        while True:
            p.stepSimulation(physics_client)
            time.sleep(1.0/240.0)  # Match physics timestep
    except KeyboardInterrupt:
        print("\nClosing visualization...")
    finally:
        p.disconnect(physics_client)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize the track")
    parser.add_argument("--config", type=str, default="track_config.yaml",
                       help="Path to track config file")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for track generation")
    parser.add_argument("--segment_start", type=float, default=0.0,
                       help="Start of segment to highlight (0.0 to 1.0)")
    parser.add_argument("--segment_end", type=float, default=None,
                       help="End of segment to highlight (None = full track)")
    
    args = parser.parse_args()
    visualize_track(args.config, seed=args.seed, 
                   segment_start=args.segment_start, segment_end=args.segment_end)


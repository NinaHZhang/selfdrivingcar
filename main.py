"""
Main training script that connects the environment and PPO model.
"""
import argparse
from environment import CatRacingEnv
from ppo import PPO
import numpy as np

def train_ppo(render=False, total_timesteps=100000, save_freq=10, load_model=None,
              track_seed=None):
    """
    Train a PPO agent on the CatRacingEnv environment.
    
    Parameters:
        render: bool - whether to render the environment during training
        total_timesteps: int - total number of timesteps to train for
        save_freq: int - save checkpoint every N iterations (0 to disable)
        load_model: str - path to model to continue training from (optional)
        track_seed: int - random seed for track generation (None = random each time)
    """
    print("Creating environment...")
    print("Training on full track")
    if track_seed is not None:
        print(f"Using track seed: {track_seed} (track will be the same each time)")
    else:
        print("Warning: No track seed set - track will be different each time!")
    env = CatRacingEnv(render=render, track_config_path="track_config.yaml",
                      track_seed=track_seed)
    
    print("Initializing PPO agent...")
    try:
        model = PPO(env)
        print("PPO agent initialized successfully")
    except Exception as e:
        print(f"Error initializing PPO agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Try to auto-load the latest model if no specific model is provided
    import os
    import glob
    if load_model is None:
        # Look for the most recent model checkpoint
        model_files = glob.glob(os.path.join(model.save_dir, "ppo_model_*.pth"))
        if model_files:
            # Sort by modification time, get most recent
            latest_model = max(model_files, key=os.path.getmtime)
            print(f"Found existing model: {latest_model}")
            print("Attempting to load latest checkpoint...")
            result = model.load(latest_model)
            if result is None:
                print("Could not load model (architecture mismatch). Starting fresh training.")
            else:
                print("Successfully loaded model. Continuing training...")
        else:
            print("No existing model found. Starting training from scratch.")
    else:
        # Load specific model if provided
        print(f"Loading model from {load_model}...")
        result = model.load(load_model)
        if result is None:
            print("Could not load model (architecture mismatch). Starting fresh training.")
    
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Models will be saved to: {model.save_dir}")
    if save_freq > 0:
        print(f"Checkpoints saved every {save_freq} iterations")
    print("-" * 50)
    
    # Test environment reset to make sure it works
    print("Testing environment reset...")
    try:
        obs, info = env.reset()
        print(f"Environment reset successful! Observation shape: {obs.shape}")
    except Exception as e:
        print(f"Error during environment reset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        # Train the model
        print("Starting training loop...")
        model.learn(total_timesteps=total_timesteps, save_freq=save_freq)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model before exit...")
        model.save(iteration="interrupted")
        print("Model saved. You can resume training with --load_model")
    
    print("\nTraining completed!")
    env.close()

def test_agent(model_path=None, render=True, n_episodes=5, track_seed=None):
    """
    Test a trained PPO agent.
    
    Parameters:
        model_path: str - path to saved model (required for testing)
        render: bool - whether to render the environment
        n_episodes: int - number of episodes to test
        track_seed: int - random seed for track generation (None = random each time)
    """
    if model_path is None:
        print("Error: --model_path is required for testing!")
        print("Example: python main.py --mode test --model_path models/ppo_model_final.pth")
        return
    
    print("Creating environment...")
    print("Testing on full track")
    if track_seed is not None:
        print(f"Using track seed: {track_seed}")
    env = CatRacingEnv(render=render, track_config_path="track_config.yaml",
                      track_seed=track_seed)
    
    print("Initializing PPO agent...")
    model = PPO(env)
    
    # Load model weights
    print(f"Loading model from {model_path}...")
    model.load(model_path)
    
    print(f"Testing agent for {n_episodes} episodes...")
    print("-" * 50)
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Use stochastic actions during testing (temporarily enable exploration)
            action, _ = model.get_action(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # If rendering, add a small delay so visualization is visible
            if render:
                import time
                time.sleep(1.0/60.0)  # ~60 FPS for visualization
            
            # Print progress every 10 steps
            if steps % 10 == 0:
                laps = info.get('laps_completed', 0)
                progress_ratio = info.get('progress_ratio', 0)  # Normalized progress (0-1)
                max_progress_ratio = info.get('max_progress_ratio', 0)  # Max progress reached
                progress_abs = info.get('progress', 0)  # Absolute progress
                max_progress_abs = info.get('max_progress', 0)  # Max absolute progress
                
                # Get observation values for more details
                inner_dist = obs[0]
                outer_dist = obs[1]
                speed = obs[3]
                heading = obs[8] if len(obs) > 8 else 0
                
                print(f"  Step {steps}: Progress={progress_ratio*100:.1f}% (Max: {max_progress_ratio*100:.1f}%), "
                      f"Laps={laps}, Speed={speed:.2f}, "
                      f"WallDist={min(inner_dist, outer_dist):.2f}, "
                      f"Heading={heading*180/3.14159:.1f}°, "
                      f"Reward={reward:.2f}")
        
        # Final episode summary
        laps = info.get('laps_completed', 0)
        progress_ratio = info.get('progress_ratio', 0)
        max_progress_ratio = info.get('max_progress_ratio', 0)
        progress_abs = info.get('progress', 0)
        max_progress_abs = info.get('max_progress', 0)
        
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Steps: {steps}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Laps Completed: {laps}")
        print(f"  Final Progress: {progress_ratio*100:.1f}%")
        print(f"  Max Progress: {max_progress_ratio*100:.1f}%")
        print(f"  Finished: {info.get('finished', False)}")
        print("-" * 50)
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test PPO agent for self-driving car")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                       help="Mode: train or test")
    parser.add_argument("--render", action="store_true", 
                       help="Render the environment (slower but visual)")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total timesteps for training")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes for testing")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to saved model (for testing or resuming training)")
    parser.add_argument("--save_freq", type=int, default=10,
                       help="Save checkpoint every N iterations (0 to disable)")
    parser.add_argument("--load_model", type=str, default=None,
                       help="Path to model to continue training from")
    parser.add_argument("--track_seed", type=int, default=None,
                       help="Random seed for track generation (None = random each time)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_ppo(render=args.render, total_timesteps=args.timesteps, 
                 save_freq=args.save_freq, load_model=args.load_model,
                 track_seed=args.track_seed)
    elif args.mode == "test":
        # For test mode, render by default unless explicitly disabled
        render = args.render if args.render else True
        test_agent(model_path=args.model_path, render=render, n_episodes=args.episodes,
                  track_seed=args.track_seed)


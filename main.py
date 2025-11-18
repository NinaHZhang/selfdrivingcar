"""
Main training script that connects the environment and PPO model.
"""
import argparse
from environment import CatRacingEnv
from ppo import PPO
import numpy as np

def train_ppo(render=False, total_timesteps=100000, save_freq=10, load_model=None):
    """
    Train a PPO agent on the CatRacingEnv environment.
    
    Parameters:
        render: bool - whether to render the environment during training
        total_timesteps: int - total number of timesteps to train for
        save_freq: int - save checkpoint every N iterations (0 to disable)
        load_model: str - path to model to continue training from (optional)
    """
    print("Creating environment...")
    env = CatRacingEnv(render=render)
    
    print("Initializing PPO agent...")
    model = PPO(env)
    
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
            print("Auto-loading latest checkpoint to continue training...")
            model.load(latest_model)
        else:
            print("No existing model found. Starting training from scratch.")
    else:
        # Load specific model if provided
        print(f"Loading model from {load_model}...")
        model.load(load_model)
    
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Models will be saved to: {model.save_dir}")
    if save_freq > 0:
        print(f"Checkpoints saved every {save_freq} iterations")
    print("-" * 50)
    
    try:
        # Train the model
        model.learn(total_timesteps=total_timesteps, save_freq=save_freq)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model before exit...")
        model.save(iteration="interrupted")
        print("Model saved. You can resume training with --load_model")
    
    print("\nTraining completed!")
    env.close()

def test_agent(model_path=None, render=True, n_episodes=5):
    """
    Test a trained PPO agent.
    
    Parameters:
        model_path: str - path to saved model (required for testing)
        render: bool - whether to render the environment
        n_episodes: int - number of episodes to test
    """
    if model_path is None:
        print("Error: --model_path is required for testing!")
        print("Example: python main.py --mode test --model_path models/ppo_model_final.pth")
        return
    
    print("Creating environment...")
    env = CatRacingEnv(render=render)
    
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
            action, _ = model.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Print progress every 10 steps
            if steps % 10 == 0:
                print(f"  Step {steps}: x={info.get('x_position', 0):.2f}, "
                      f"max_x={info.get('max_x_reached', 0):.2f}, "
                      f"progress={info.get('progress', 0)*100:.1f}%, "
                      f"reward={reward:.2f}, finished={info.get('finished', False)}")
        
        print(f"Episode {episode + 1}: Steps={steps}, Total Reward={total_reward:.2f}, "
              f"Max X={info.get('max_x_reached', 0):.2f}, Finished={info.get('finished', False)}")
    
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
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_ppo(render=args.render, total_timesteps=args.timesteps, 
                 save_freq=args.save_freq, load_model=args.load_model)
    elif args.mode == "test":
        test_agent(model_path=args.model_path, render=args.render, n_episodes=args.episodes)


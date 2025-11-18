"""
Main training script that connects the environment and PPO model.
"""
import argparse
from environment import CatRacingEnv
from ppo import PPO
import numpy as np

def train_ppo(render=False, total_timesteps=100000):
    """
    Train a PPO agent on the CatRacingEnv environment.
    
    Parameters:
        render: bool - whether to render the environment during training
        total_timesteps: int - total number of timesteps to train for
    """
    print("Creating environment...")
    env = CatRacingEnv(render=render)
    
    print("Initializing PPO agent...")
    model = PPO(env)
    
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("-" * 50)
    
    # Train the model
    model.learn(total_timesteps=total_timesteps)
    
    print("\nTraining completed!")
    env.close()

def test_agent(model_path=None, render=True, n_episodes=5):
    """
    Test a trained PPO agent.
    
    Parameters:
        model_path: str - path to saved model (not implemented yet)
        render: bool - whether to render the environment
        n_episodes: int - number of episodes to test
    """
    print("Creating environment...")
    env = CatRacingEnv(render=render)
    
    print("Initializing PPO agent...")
    model = PPO(env)
    
    # TODO: Load model weights if model_path is provided
    # if model_path:
    #     model.load(model_path)
    
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
        
        print(f"Episode {episode + 1}: Steps={steps}, Total Reward={total_reward:.2f}")
    
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
                       help="Path to saved model (for testing)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_ppo(render=args.render, total_timesteps=args.timesteps)
    elif args.mode == "test":
        test_agent(model_path=args.model_path, render=args.render, n_episodes=args.episodes)


"""
Quick test script to verify the environment and PPO model connection.
"""
import sys

print("Testing imports...")
try:
    from environment import CatRacingEnv
    print("✓ Environment imported successfully")
except Exception as e:
    print(f"✗ Environment import failed: {e}")
    sys.exit(1)

try:
    from ppo import PPO
    print("✓ PPO imported successfully")
except Exception as e:
    print(f"✗ PPO import failed: {e}")
    sys.exit(1)

try:
    from network import FeedForwardNN
    print("✓ Network imported successfully")
except Exception as e:
    print(f"✗ Network import failed: {e}")
    sys.exit(1)

print("\nTesting environment creation...")
try:
    env = CatRacingEnv(render=False)
    print("✓ Environment created successfully")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
except Exception as e:
    print(f"✗ Environment creation failed: {e}")
    sys.exit(1)

print("\nTesting environment reset...")
try:
    obs, info = env.reset()
    print(f"✓ Environment reset successfully")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation: {obs}")
except Exception as e:
    print(f"✗ Environment reset failed: {e}")
    sys.exit(1)

print("\nTesting PPO initialization...")
try:
    model = PPO(env)
    print("✓ PPO model initialized successfully")
    print(f"  Observation dim: {model.obs_dim}")
    print(f"  Action dim: {model.act_dim}")
except Exception as e:
    print(f"✗ PPO initialization failed: {e}")
    sys.exit(1)

print("\nTesting action sampling...")
try:
    action, log_prob = model.get_action(obs)
    print(f"✓ Action sampled successfully")
    print(f"  Action: {action}")
    print(f"  Action shape: {action.shape}")
    print(f"  Log prob: {log_prob}")
except Exception as e:
    print(f"✗ Action sampling failed: {e}")
    sys.exit(1)

print("\nTesting environment step...")
try:
    obs_new, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Environment step successful")
    print(f"  Reward: {reward:.4f}")
    print(f"  Terminated: {terminated}, Truncated: {truncated}")
except Exception as e:
    print(f"✗ Environment step failed: {e}")
    sys.exit(1)

print("\nTesting a few more steps...")
try:
    for i in range(5):
        action, _ = model.get_action(obs_new)
        obs_new, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.4f}, done={terminated or truncated}")
        if terminated or truncated:
            obs_new, info = env.reset()
            print(f"  Reset after termination")
except Exception as e:
    print(f"✗ Multiple steps failed: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("✓ ALL TESTS PASSED!")
print("="*50)
print("\nThe environment and PPO model are properly connected!")
print("You can now run training with:")
print("  python main.py --mode train --timesteps 100000")

env.close()


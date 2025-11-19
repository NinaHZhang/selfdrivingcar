"""
Curriculum learning script that automatically trains on track segments sequentially.
Starts with small segments and gradually increases to full track.
"""
import argparse
import os
from main import train_ppo

def curriculum_train(total_timesteps_per_segment=50000, save_freq=10, 
                    track_seed=42, render=False, load_model=None):
    """
    Train using curriculum learning: start with small segments, gradually increase.
    
    Parameters:
        total_timesteps_per_segment: int - timesteps to train on each segment
        save_freq: int - save checkpoint every N iterations
        track_seed: int - random seed for track generation
        render: bool - whether to render during training
        load_model: str - path to model to start from (optional)
    """
    # Define curriculum: segments to train on sequentially
    # Each segment is [start, end] as normalized progress (0.0 to 1.0)
    curriculum = [
        (0.0, 0.25),    # First 25% of track
        (0.0, 0.50),    # First 50% of track
        (0.0, 0.75),    # First 75% of track
        (0.0, 1.0),     # Full track
    ]
    
    # Alternative: overlapping segments for smoother progression
    # curriculum = [
    #     (0.0, 0.25),    # First 25%
    #     (0.25, 0.50),   # Second 25%
    #     (0.0, 0.50),    # First half
    #     (0.50, 0.75),   # Third 25%
    #     (0.75, 1.0),    # Last 25%
    #     (0.0, 1.0),     # Full track
    # ]
    
    print("=" * 70)
    print("CURRICULUM LEARNING: Sequential Segment Training")
    print("=" * 70)
    print(f"Track seed: {track_seed}")
    print(f"Timesteps per segment: {total_timesteps_per_segment}")
    print(f"Total segments: {len(curriculum)}")
    print(f"Total training time: {len(curriculum) * total_timesteps_per_segment} timesteps")
    print("=" * 70)
    
    current_model = load_model
    
    for segment_idx, (segment_start, segment_end) in enumerate(curriculum, 1):
        print(f"\n{'=' * 70}")
        print(f"SEGMENT {segment_idx}/{len(curriculum)}: {segment_start:.0%} to {segment_end:.0%}")
        print(f"{'=' * 70}")
        
        # Train on this segment
        try:
            train_ppo(
                render=render,
                total_timesteps=total_timesteps_per_segment,
                save_freq=save_freq,
                load_model=current_model,  # Continue from previous segment's model
                segment_start=segment_start,
                segment_end=segment_end,
                track_seed=track_seed
            )
            
            # After training, the latest model should be saved
            # Auto-load it for the next segment
            from ppo import PPO
            from environment import CatRacingEnv
            temp_env = CatRacingEnv(render=False, track_config_path="track_config.yaml",
                                   segment_start_progress=0.0, segment_end_progress=1.0,
                                   track_seed=track_seed)
            temp_model = PPO(temp_env)
            import glob
            model_files = glob.glob(os.path.join(temp_model.save_dir, "ppo_model_*.pth"))
            if model_files:
                current_model = max(model_files, key=os.path.getmtime)
                print(f"\n✓ Segment {segment_idx} complete! Model saved: {current_model}")
                print(f"  Continuing to next segment with this model...")
            else:
                print(f"\n⚠ Warning: No model file found after segment {segment_idx}")
            
            temp_env.close()
            
        except KeyboardInterrupt:
            print(f"\n\nTraining interrupted during segment {segment_idx}!")
            print(f"Current model should be saved. You can resume with:")
            print(f"  python curriculum_train.py --load_model {current_model}")
            break
        except Exception as e:
            print(f"\n✗ Error during segment {segment_idx}: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nYou can try to continue manually from the last saved model.")
            break
    
    print(f"\n{'=' * 70}")
    print("CURRICULUM LEARNING COMPLETE!")
    print(f"{'=' * 70}")
    if current_model:
        print(f"Final model: {current_model}")
    print("You can now test the model:")
    print(f"  python main.py --mode test --model_path {current_model} --track_seed {track_seed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum learning: train on segments sequentially")
    parser.add_argument("--timesteps_per_segment", type=int, default=50000,
                       help="Timesteps to train on each segment")
    parser.add_argument("--save_freq", type=int, default=10,
                       help="Save checkpoint every N iterations")
    parser.add_argument("--track_seed", type=int, default=42,
                       help="Random seed for track generation")
    parser.add_argument("--render", action="store_true",
                       help="Render during training (slower)")
    parser.add_argument("--load_model", type=str, default=None,
                       help="Path to model to start from (optional)")
    
    args = parser.parse_args()
    
    curriculum_train(
        total_timesteps_per_segment=args.timesteps_per_segment,
        save_freq=args.save_freq,
        track_seed=args.track_seed,
        render=args.render,
        load_model=args.load_model
    )


# Training Guide for Self-Driving Car PPO Agent

## Training Strategy: Curriculum Learning

### Phase 1: Straight Track (Current Setup)
**Goal**: Learn basic forward movement and staying centered

**Training Steps**:
1. Start with 50,000-100,000 timesteps
2. Monitor average episode length - should increase as agent learns
3. Monitor average reward - should become positive and increase
4. Agent should learn to:
   - Move forward consistently
   - Stay roughly centered
   - Avoid walls

**Success Criteria**: 
- Average episode length > 200 steps
- Average reward > 0
- Agent can drive straight for extended periods

### Phase 2: Gentle Curves (Future)
**Goal**: Learn to turn while maintaining speed

**Implementation**: Modify track to have gentle curves
- Add parameter to environment for track curvature
- Start with very gentle curves (large radius)
- Gradually increase curvature as agent improves

### Phase 3: Circular Track (Future)
**Goal**: Master continuous turning

**Implementation**: Create circular track
- Full 360-degree loop
- Requires constant steering adjustment
- Ultimate test of learned behavior

## Hyperparameter Recommendations

### Current Settings (Optimized):
- **Learning Rate**: 0.003 (reduced from 0.005 for stability)
- **Discount Factor (γ)**: 0.99 (good for long episodes)
- **PPO Clip**: 0.2 (standard)
- **Timesteps per Batch**: 2048 (good balance)
- **Updates per Iteration**: 5 (standard)
- **Max Episode Length**: 500 (reduced for faster training on straight track)

### If Training is Unstable:
- Reduce learning rate to 0.001
- Increase `n_updates_per_iteration` to 10
- Reduce `timesteps_per_batch` to 1024

### If Training is Too Slow:
- Increase learning rate to 0.005 (but watch for instability)
- Increase `timesteps_per_batch` to 4096
- Reduce `n_updates_per_iteration` to 3

## Reward Function Improvements

### What Changed:
1. **Forward Speed Reward**: Increased from 0.1 to 0.5 (encourages movement)
2. **Center Reward**: Added positive reward for staying centered (smooth gradient)
3. **Progressive Wall Penalty**: Distance-based instead of binary (better learning signal)
4. **Reduced Steering Penalty**: Less harsh, only penalizes excessive steering

### Reward Components:
- `forward_speed * 0.5`: Primary driver for movement
- `center_reward * 0.3`: Encourages staying in lane center
- `-abs(center_offset) * 0.2`: Small penalty for being off-center
- `-wall_penalty`: Progressive penalty as you approach walls
- `-abs(steering) * 0.05`: Only if steering > 0.5

## Network Architecture

### Current: 3-Layer Network
- Input: 6 (observation dimensions)
- Hidden 1: 64 neurons (ReLU)
- Hidden 2: 64 neurons (ReLU)
- Output: 2 (action dimensions: steering, throttle)

**Increased from 32 to 64 neurons** for better learning capacity.

## Training Commands

### Basic Training (No Rendering - Faster):
```bash
python main.py --mode train --timesteps 100000
```

### Training with Visualization:
```bash
python main.py --mode train --render --timesteps 100000
```

### Testing Trained Agent:
```bash
python main.py --mode test --render --episodes 10
```

## Monitoring Training Progress

Watch for these metrics in the training output:
- **Average Episode Length**: Should increase over time
- **Average Reward**: Should become positive and increase
- **Iteration Number**: Tracks training progress

### Expected Training Curve:
1. **Early (0-20k timesteps)**: 
   - Short episodes (10-50 steps)
   - Negative rewards
   - Agent crashes quickly

2. **Mid (20k-60k timesteps)**:
   - Longer episodes (50-200 steps)
   - Rewards becoming positive
   - Agent learns to stay on track

3. **Late (60k+ timesteps)**:
   - Long episodes (200-500 steps)
   - Consistently positive rewards
   - Smooth driving behavior

## Troubleshooting

### Agent Not Learning:
- Check if rewards are too sparse (increase forward speed reward)
- Verify environment is resetting properly
- Check if action clipping is too aggressive

### Training Unstable:
- Reduce learning rate
- Increase number of update iterations
- Check for NaN values in losses

### Agent Crashes Immediately:
- Check reward function (might be too harsh)
- Verify action space bounds
- Check if physics parameters are reasonable

## Next Steps After Straight Track Mastery

1. **Add Track Curvature**: Modify `_create_track()` to support curves
2. **Implement GAE**: Use the `lam` parameter for Generalized Advantage Estimation
3. **Add Mini-Batching**: Use `batch_size` and `epochs` for more efficient training
4. **Save/Load Models**: Implement model checkpointing for resuming training


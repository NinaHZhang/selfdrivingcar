import torch
import torch.nn as nn
import numpy as np
import os
from torch.optim import Adam
from network import FeedForwardNN
from torch.distributions import MultivariateNormal

class PPO:
    '''PPO class used to train agent'''

    def __init__(self, env):
        '''Initializes PPO model including hyperparameters.
        parameters: env - the environment to train on'''
       
        #extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self._init_hyperparameters()

        #initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        
        # Initialize actor network to output reasonable default actions
        # This helps with initial exploration - start with some throttle
        for param in self.actor.parameters():
            if len(param.shape) >= 2:
                # Initialize weights with smaller values for more stable learning
                torch.nn.init.xavier_uniform_(param, gain=0.5)
            else:
                # Initialize bias for throttle to be around 0.5 (moderate throttle)
                # This ensures initial actions have some throttle, not all zeros
                if param.shape[0] == self.act_dim:
                    # Bias for output layer - set throttle bias to encourage initial movement
                    param.data[1] = 0.5  # throttle bias (action[1]) - encourage initial throttle

        #create a variable for matrix - variance for exploration
        # Use different variances for steering vs throttle for better exploration
        # Steering: moderate variance (reduced to prevent erratic behavior)
        # Throttle: moderate variance (but ensure it explores positive values)
        self.cov_var = torch.tensor([1.0, 0.6])  # [steering_var, throttle_var] - reduced for more stable learning
        # Ensure we have the right shape
        if len(self.cov_var) != self.act_dim:
            self.cov_var = torch.full(size=(self.act_dim,), fill_value = 1.0)

        #create the covariance matrix, which will be used for multivariate norm distr
        self.cov_mat = torch.diag(self.cov_var)

        #backpropagate
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
        #model saving directory
        self.save_dir = "models"
        os.makedirs(self.save_dir, exist_ok=True)

    def get_action(self, obs, deterministic=False):
        '''samples an action to create a multivariate normal distribution, using the mean found through the actor network inputting an observation. 
        samples an action from the distribution and its probability, returns it as a numpy array
        
        Parameters:
            obs: observation from environment
            deterministic: if True, use mean action (no exploration). If False, sample from distribution.
        '''
        #convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        elif not isinstance(obs, torch.Tensor):
            # if it's neither numpy array nor tensor, try to convert
            obs = torch.FloatTensor(np.array(obs, dtype=np.float32))
        else:
            # if it's already a tensor, ensure it's float
            obs = obs.float()
        
        # ensure it's 1D (flatten if needed) and has correct shape
        if obs.dim() > 1:
            obs = obs.flatten()
        elif obs.dim() == 0:
            obs = obs.unsqueeze(0)
        
        # ensure it has the right number of dimensions for the network
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # add batch dimension if needed
        
        #querying the neural network for a mean action
        mean = self.actor(obs)
        
        # remove batch dimension if we added it
        if mean.dim() > 1:
            mean = mean.squeeze(0)
        
        # Ensure mean actions are in reasonable range before sampling
        # This helps prevent the network from outputting extreme values
        # Steering: clip to [-2, 2] range (will be clipped to [-1, 1] later anyway)
        # Throttle: ensure it's not too negative (we want positive throttle)
        mean[0] = torch.clamp(mean[0], -2.0, 2.0)  # steering
        mean[1] = torch.clamp(mean[1], -1.0, 2.0)  # throttle (allow some negative for exploration, but mostly positive)
        
        #create a multivariate normal distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        #sample an action from the distribution and get its log probability
        # During testing, use deterministic action (mean) instead of sampling
        if deterministic:
            action = mean  # Use mean action (deterministic)
            log_prob = dist.log_prob(action)  # Still compute log_prob for consistency
        else:
            action = dist.sample()  # Sample from distribution (for training/exploration)
            log_prob = dist.log_prob(action)

        #clip action to action space bounds and convert to numpy array
        action_clipped = torch.clamp(
            action,
            torch.FloatTensor(self.env.action_space.low),
            torch.FloatTensor(self.env.action_space.high)
        )
        return action_clipped.detach().cpu().numpy(), log_prob.detach()
    def _init_hyperparameters(self):
        #hyperparameters
        self.lr = 0.0003  #learning rate (reduced from 0.003 for more stable learning)
        self.gamma = 0.99  #discount factor
        self.lam = 0.95   #lambda for gae (currently not used, but kept for future GAE implementation)
        self.clip = 0.2  #clip parameter for ppo
        self.epochs = 10  #number of epochs per update (currently not used - full batch updates)
        self.batch_size = 64   #mini-batch size (currently not used - full batch updates)
        self.timesteps_per_batch = 4096    #timesteps per batch (increased from 2048 for longer iterations)
        self.n_updates_per_iteration = 5  #number of PPO update iterations per batch
        self.max_timesteps_per_episode = 500  #max steps per episode (track is now 40 units, need more steps to reach finish)

        # Entropy/exploration schedule
        self.entropy_coef_start = 0.02   # initial entropy bonus
        self.entropy_coef_end = 0.001    # final entropy bonus
        self.entropy_decay_iters = 150   # iterations over which to decay (longer training)
        self.entropy_coef = self.entropy_coef_start

    def rollout(self):
        batch_obs = []  #batch state/observations
        batch_acts = [] #batch actions
        batch_log_probs = [] #log probability of each action
        batch_rews = [] # batch rewards (r_t) - list of lists (one per episode)
        batch_rtgs = [] #batch rewards-to-go (aka, return R_t with discounts)
        batch_lens = [] #episode length, steps in eps
        batch_laps = []  # track lap completions per episode
        batch_max_progress = []  # track maximum progress per episode
        
        #number of timesteps run in this batch
        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = [] #rewards this episode

            #reset environment (Gymnasium returns obs, info)
            obs, info = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1

                batch_obs.append(obs) #collect observation
                action, log_prob = self.get_action(obs) #get action and action probabilities from actor
                
                #step environment (Gymnasium returns obs, reward, terminated, truncated, info)
                obs, rew, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                #collect reward, action and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                
                if done:
                    break
            
            #collect episodic length and the rewards
            batch_lens.append(ep_t + 1) #because timesteps start at 0
            batch_rews.append(ep_rews)
            # Collect lap and progress info from the final info dict
            batch_laps.append(info.get('laps_completed', 0))
            # Use max_progress_ratio to show the furthest point reached in that iteration
            # This represents how much of the track was completed (0-1, where 1 = full lap)
            batch_max_progress.append(info.get('max_progress_ratio', 0.0))

        #reshape data as tensors in shape specified (convert lists to arrays first for speed)
        batch_obs = torch.tensor(np.array(batch_obs, dtype=np.float32))
        batch_acts = torch.tensor(np.array(batch_acts, dtype=np.float32))
        batch_log_probs = torch.stack(batch_log_probs)

        #compute rewardstogo/return
        batch_rtgs = self.compute_rtgs(batch_rews)

        #return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews, batch_laps, batch_max_progress

    def compute_rtgs(self, batch_rews):
        #have to return an array of rtg per episode per batch to return
        batch_rtgs = []

        #iterate through each episode in the batch
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0

            for rew in reversed(ep_rews): #loop through each timestep in the episode
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        #convert the rtg into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def learn(self, total_timesteps, save_freq=10):
        """
        Train the PPO agent.
        
        Parameters:
            total_timesteps: int - total number of timesteps to train
            save_freq: int - save checkpoint every N iterations (0 to disable)
        """
        t_so_far = 0 #timesteps simulated so far
        i_so_far = 0 #iterations so far
        
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews, batch_laps, batch_max_progress = self.rollout()
            
            #increment timesteps simulated
            t_so_far += np.sum(batch_lens)
            i_so_far += 1

            # Update entropy coefficient (linear decay)
            decay_frac = min(i_so_far / self.entropy_decay_iters, 1.0)
            self.entropy_coef = self.entropy_coef_start + decay_frac * (self.entropy_coef_end - self.entropy_coef_start)

            #calculate  V_{phi, k} 
            V, _, _ = self.evaluate(batch_obs, batch_acts)
            
            #print training progress
            avg_ep_len = np.mean(batch_lens)
            avg_rew = np.mean([np.sum(ep_rews) for ep_rews in batch_rews])
            max_ep_len = np.max(batch_lens)
            
            # Calculate reward statistics
            all_rewards = [r for ep_rews in batch_rews for r in ep_rews]
            min_rew = np.min(all_rewards) if all_rewards else 0
            max_rew = np.max(all_rewards) if all_rewards else 0
            
            # Calculate lap completion statistics
            total_laps = sum(batch_laps)
            episodes_with_laps = sum(1 for laps in batch_laps if laps > 0)
            max_laps = max(batch_laps) if batch_laps else 0
            avg_laps = np.mean(batch_laps) if batch_laps else 0.0
            
            # Calculate progress statistics
            avg_progress = np.mean(batch_max_progress) * 100 if batch_max_progress else 0.0  # Convert to percentage
            max_progress = max(batch_max_progress) * 100 if batch_max_progress else 0.0
            episodes_completing_50pct = sum(1 for p in batch_max_progress if p >= 0.5)
            episodes_completing_90pct = sum(1 for p in batch_max_progress if p >= 0.9)
            
            print(f"Iteration {i_so_far}: Timesteps={t_so_far}/{total_timesteps}, "
                  f"Avg Episode Length={avg_ep_len:.1f}, Max={max_ep_len:.0f}, "
                  f"Avg Reward={avg_rew:.2f} (min={min_rew:.2f}, max={max_rew:.2f}), "
                  f"Episodes={len(batch_lens)}")
            print(f"  Laps: Total={total_laps}, Episodes w/ Laps={episodes_with_laps}/{len(batch_laps)}, "
                  f"Max={max_laps}, Avg={avg_laps:.2f}")
            print(f"  Progress: Avg={avg_progress:.1f}%, Max={max_progress:.1f}%, "
                  f"Episodes >50%={episodes_completing_50pct}, Episodes >90%={episodes_completing_90pct}")

            #calculate advantage at the k-ith iteration
            A_k = batch_rtgs - V.detach()

            #normalize advantage (only if std > 0 to avoid division issues)
            if A_k.std() > 1e-8:
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            else:
                # If all advantages are the same, set to zero (no learning signal)
                A_k = A_k - A_k.mean()

            for _ in range(self.n_updates_per_iteration):
                #this is to find pi_theta(a_t | s_t). we use the most current policy to represent it. also we get current v and logprobs
                V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_acts)
                
                #calculate the ratio, because both are log probs, we can do that by subtrating and then exponentiating the log out with e 
                # Clamp log_prob difference to prevent numerical instability
                log_ratio = curr_log_probs - batch_log_probs
                log_ratio = torch.clamp(log_ratio, min=-10, max=10)  # prevent extreme values
                ratios = torch.exp(log_ratio)

                #calculate surrogate losses. surr1 is not clipped, surr 2 is clipped, then you find the min one to make sure we step the least during 
                #gradient ascent
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1-self.clip, 1+ self.clip) * A_k  #basically, cannot go above 1 + self.clip and below 1-self.clip
                policy_loss = (-torch.min(surr1, surr2)).mean()
                entropy_bonus = self.entropy_coef * entropy.mean()
                actor_loss = policy_loss - entropy_bonus  # encourage exploration with entropy bonus
                #minimizes loss, so we will be minimizing negative loss to maximize performance function
                #okay in a nutshell model picks the smallest option, if A_t is positive, the clipped (surr 2) is used, if A_t is negative, surr 1 is used
                #so good actions dont create too big of steps in gradient ascent but bad actions are penalized fully i think?
                #now we backprop

                #calculate gradients and perform backprop for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optim.step()

                critic_loss = nn.MSELoss()(V, batch_rtgs) #calculate MSE of predicted values and return (rewards-to-go) then backprop
                self.critic_optim.zero_grad()
                critic_loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optim.step()
            
            # Save checkpoint periodically
            if save_freq > 0 and i_so_far % save_freq == 0:
                self.save(iteration=i_so_far)
        
        # Save final model
        self.save(iteration=None)  # Save as final model
        print(f"\nTraining completed! Final model saved.")
    
    def evaluate(self, batch_obs, batch_acts):
        """
        Evaluate the current policy on a batch of observations and actions.
        Returns value estimates and log probabilities.
        """
        # Ensure batch_acts is a tensor
        if isinstance(batch_acts, np.ndarray):
            batch_acts = torch.from_numpy(batch_acts).float()
        
        V = self.critic(batch_obs).squeeze() #queries critic network for a value V for every obs in batch_obs
        
        # Ensure V has correct shape
        if V.dim() == 0:
            V = V.unsqueeze(0)
        elif V.dim() > 1:
            V = V.squeeze()

        # Get mean action from actor
        mean = self.actor(batch_obs)
        
        # Ensure mean has correct shape (should match batch_acts)
        if mean.dim() == 1 and batch_acts.dim() == 2:
            # If batch_acts is 2D but mean is 1D, something is wrong
            pass
        elif mean.dim() == 2 and batch_acts.dim() == 1:
            # If mean is 2D but batch_acts is 1D, squeeze mean
            mean = mean.squeeze(0)
        
        #get the multivariate norm distribution using that
        dist = MultivariateNormal(mean, self.cov_mat)
        
        #get the log probabilities of all the batch actions from dist
        log_probs = dist.log_prob(batch_acts)
        entropy = dist.entropy()
        
        # Ensure log_probs is 1D
        if log_probs.dim() > 1:
            log_probs = log_probs.squeeze()
        elif log_probs.dim() == 0:
            log_probs = log_probs.unsqueeze(0)
            
        # Ensure entropy is 1D
        if entropy.dim() > 1:
            entropy = entropy.squeeze()
        elif entropy.dim() == 0:
            entropy = entropy.unsqueeze(0)
            
        return V, log_probs, entropy
    
    def save(self, filepath=None, iteration=None):
        """
        Save the actor and critic networks.
        
        Parameters:
            filepath: str - full path to save model (if None, uses default naming)
            iteration: int - iteration number for checkpoint naming
        """
        if filepath is None:
            if iteration is not None:
                filepath = os.path.join(self.save_dir, f"ppo_model_iter_{iteration}.pth")
            else:
                filepath = os.path.join(self.save_dir, "ppo_model_final.pth")
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optim.state_dict(),
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'hyperparameters': {
                'lr': self.lr,
                'gamma': self.gamma,
                'clip': self.clip,
            }
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load the actor and critic networks from a saved checkpoint.
        
        Parameters:
            filepath: str - path to saved model file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Check if observation dimensions match
        # Old models might have different obs_dim if observation space changed
        old_obs_dim = checkpoint.get('obs_dim', None)
        if old_obs_dim is not None and old_obs_dim != self.obs_dim:
            print(f"WARNING: Observation space mismatch!")
            print(f"  Old model: {old_obs_dim} dimensions")
            print(f"  Current environment: {self.obs_dim} dimensions")
            print(f"  Cannot load old model - starting fresh training...")
            return None
        
        # Check by trying to load and catching size mismatch
        try:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optim.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            print(f"Model loaded from {filepath}")
            return checkpoint
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"WARNING: Model architecture mismatch!")
                print(f"  Old model has different input size than current environment")
                print(f"  Error: {e}")
                print(f"  Starting fresh training instead...")
                return None
            else:
                raise  # Re-raise if it's a different error
       


import torch
import torch.nn as nn
import numpy as np
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

        #create a variable for matrix, 0.5 for stdev is arbitrary, used to generate the covariance matrix
        self.cov_var = torch.full(size=(self.act_dim,), fill_value = 0.5)

        #create the covariance matrix, which will be used for multivariate norm distr
        self.cov_mat = torch.diag(self.cov_var)

        #backpropagate
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    def get_action(self, obs):
        '''samples an action to create a multivariate normal distribution, using the mean found through the actor network inputting an observation. 
        samples an action from the distribution and its probability, returns it as a numpy array'''
        #convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        
        #querying the neural network for a mean action
        mean = self.actor(obs)
        
        #create a multivariate normal distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        #sample an action from the distribution and get its log probability
        action = dist.sample()
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
        self.lr = 0.003  #learning rate (reduced from 0.005 for stability)
        self.gamma = 0.99  #discount factor
        self.lam = 0.95   #lambda for gae (currently not used, but kept for future GAE implementation)
        self.clip = 0.2  #clip parameter for ppo
        self.epochs = 10  #number of epochs per update (currently not used - full batch updates)
        self.batch_size = 64   #mini-batch size (currently not used - full batch updates)
        self.timesteps_per_batch = 2048    #timesteps per batch
        self.n_updates_per_iteration = 5  #number of PPO update iterations per batch
        self.max_timesteps_per_episode = 500  #max steps per episode (reduced for faster training on straight track)

    def rollout(self):
        batch_obs = []  #batch state/observations
        batch_acts = [] #batch actions
        batch_log_probs = [] #log probability of each action
        batch_rews = [] # batch rewards (r_t) - list of lists (one per episode)
        batch_rtgs = [] #batch rewards-to-go (aka, return R_t with discounts)
        batch_lens = [] #episode length, steps in eps
        
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

        #reshape data as tensors in shape specified
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        #compute rewardstogo/return
        batch_rtgs = self.compute_rtgs(batch_rews)

        #return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews

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

    def learn(self, total_timesteps):
        t_so_far = 0 #timesteps simulated so far
        i_so_far = 0 #iterations so far
        
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews = self.rollout()
            
            #increment timesteps simulated
            t_so_far += np.sum(batch_lens)
            i_so_far += 1

            #calculate  V_{phi, k} 
            V, _ = self.evaluate(batch_obs, batch_acts)
            
            #print training progress
            avg_ep_len = np.mean(batch_lens)
            avg_rew = np.mean([np.sum(ep_rews) for ep_rews in batch_rews])
            print(f"Iteration {i_so_far}: Timesteps={t_so_far}/{total_timesteps}, "
                  f"Avg Episode Length={avg_ep_len:.1f}, Avg Reward={avg_rew:.2f}")

            #calculate advantage at the k-ith iteration
            A_k = batch_rtgs - V.detach()

            #normalize advantage
            A_k = (A_k-A_k.mean())/(A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                #this is to find pi_theta(a_t | s_t). we use the most current policy to represent it. also we get current v and logprobs
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                #calculate the ratio, because both are log probs, we can do that by subtrating and then exponentiating the log out with e 
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                #calculate surrogate losses. surr1 is not clipped, surr 2 is clipped, then you find the min one to make sure we step the least during 
                #gradient ascent
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1-self.clip, 1+ self.clip) * A_k  #basically, cannot go above 1 + self.clip and below 1-self.clip
                actor_loss = (-torch.min(surr1, surr2)).mean() #takes min of both options, also we use negative because the optimizer used is Adam which
                #minimizes loss, so we will be minimizing negative loss to maximize performance function
                #okay in a nutshell model picks the smallest option, if A_t is positive, the clipped (surr 2) is used, if A_t is negative, surr 1 is used
                #so good actions dont create too big of steps in gradient ascent but bad actions are penalized fully i think?
                #now we backprop

                

                #calculate gradients and perform backprop for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                critic_loss = nn.MSELoss()(V, batch_rtgs) #calculate MSE of predicted values and return (rewards-to-go) then backprop
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()




    
    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze() #queries critic network for a value V for every obs in batch_obs

        #pretty sure this returns a random action
        mean = self.actor(batch_obs)
        #get the multivariate norm distribution using that
        dist = MultivariateNormal(mean, self.cov_mat)
        #get the log probabilities of all the batch actions from dist
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs
       


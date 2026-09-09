import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

# Hyperparameters (you should experiment with these!)
LEARNING_RATE = 5e-4
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
BUFFER_SIZE = 500
TARGET_UPDATE_FREQ = 10  # Update target network every N episodes

# Create environment
env = gym.make('LunarLander-v3', render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env, 
    video_folder="variation_2/recordings", 
    episode_trigger=lambda x: x % 100 == 0,
    disable_logger=True
)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")

# TODO: Implement the classes described in Part B

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # Changed from Sequential to manual layers
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [] # Changed from deque to standard list
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        # Using manual list comprehension instead of zip(*)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        states = torch.FloatTensor(np.array([s[0] for s in samples]))
        actions = torch.LongTensor([s[1] for s in samples])
        rewards = torch.FloatTensor([s[2] for s in samples])
        next_states = torch.FloatTensor(np.array([s[3] for s in samples]))
        dones = torch.FloatTensor([float(s[4]) for s in samples])
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# Training loop

policy_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict()) 

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer(BUFFER_SIZE)

num_episodes = 500
rewards_history = []
lengths_history = []
loss_history = []
q_values_history = []
epsilon_history = []
epsilon = EPSILON_START

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    steps = 0
    done = False
    
    while not done:
        # TODO: Select action using epsilon-greedy
        if random.random() < epsilon:
            action = env.action_space.sample() 
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy_net(state_t)
                action = q_values.argmax().item()

        # TODO: Take action in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        done = terminated or truncated
        
        # TODO: Store experience in replay buffer
        memory.push(state, action, reward, next_state, done)
        state = next_state
        
        # TODO: Train agent if buffer has enough samples
        if len(memory) > BATCH_SIZE:
            b_state, b_action, b_reward, b_next_state, b_done = memory.sample(BATCH_SIZE)
            
            # Beginner-style Q calculation
            current_q = policy_net(b_state).gather(1, b_action.unsqueeze(1)).squeeze()
            
            with torch.no_grad():
                next_q = target_net(b_next_state).max(1)[0]
                target_q = b_reward + (GAMMA * next_q * (1 - b_done))
            
            loss = nn.MSELoss()(current_q, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())
            q_values_history.append(current_q.mean().item())

        # TODO: Update target network periodically
        pass
    
    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # TODO: Decay epsilon
    epsilon_history.append(epsilon)
    if epsilon > EPSILON_END:
        epsilon = epsilon * EPSILON_DECAY

    # TODO: Track and log statistics
    rewards_history.append(episode_reward)
    lengths_history.append(steps)
    
    if episode % 20 == 0:
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}")

# Testing
# TODO: Test your trained agent

test_rewards = []
test_env = gym.wrappers.RecordVideo(
    gym.make('LunarLander-v3', render_mode="rgb_array"), 
    video_folder="variation_2/test_recordings", 
    episode_trigger=lambda x: x < 5,
    disable_logger=True
)

for i in range(100):
    state, _ = test_env.reset()
    done = False
    ep_reward = 0
    while not done:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action = policy_net(state_t).argmax().item()
        state, reward, terminated, truncated, _ = test_env.step(action)
        ep_reward += reward
        done = terminated or truncated
    test_rewards.append(ep_reward)

torch.save(policy_net.state_dict(), "variation_2/dqn_lunar_lander_v2.pth")

fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('DQN Training and Analysis var 2 smaller buffer', fontsize=16)

axs[0, 0].plot(rewards_history)
axs[0, 0].set_title('Episode Rewards')

axs[0, 1].plot(loss_history[::100])
axs[0, 1].set_title('Loss')

axs[1, 0].plot(q_values_history[::100])
axs[1, 0].set_title('Average Q-Values')

axs[1, 1].hist(test_rewards, bins=20)
axs[1, 1].set_title('Evaluation Rewards (100 Episodes)')

plt.savefig('variation_2/dqn_analysis_v2.png')
plt.show()

test_env.close()
env.close()
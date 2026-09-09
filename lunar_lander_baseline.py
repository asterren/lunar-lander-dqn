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
BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 10  # Update target network every N episodes

# Create environment
env = gym.make('LunarLander-v3', render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env, 
    video_folder="baseline/recordings", 
    episode_trigger=lambda x: x % 30 == 0,
    disable_logger=True
)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")

# TODO: Implement the classes described in Part B

# Training loop
num_episodes = 100
rewards_history = []
lengths_history = []
success_count = 0
epsilon = EPSILON_START

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    steps = 0
    done = False
    
    while not done:
        # TODO: Select action using epsilon-greedy
        action = env.action_space.sample()
        # TODO: Take action in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        done = terminated or truncated
        state = next_state
        
        if terminated and reward >= 100:
            success_count += 1
        # TODO: Store experience in replay buffer
        # TODO: Train agent if buffer has enough samples
        # TODO: Update target network periodically
        pass
    
    rewards_history.append(episode_reward)
    lengths_history.append(steps)
    
    # TODO: Decay epsilon
    # TODO: Track and log statistics
    
    if episode % 20 == 0:
        print(f"Episode {episode}, Reward: {episode_reward:.2f}")

# Testing
# TODO: Test your trained agent

mean_reward = np.mean(rewards_history)
mean_length = np.mean(lengths_history)
success_rate = (success_count / num_episodes) * 100

fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Part A - Random Policy Baseline', fontsize=16)

axs[0, 0].plot(rewards_history, color='steelblue', alpha=0.7)
axs[0, 0].axhline(y=mean_reward, color='r', linestyle='--', label=f'Mean = {mean_reward:.1f}')
axs[0, 0].set_title('Episode Reward')
axs[0, 0].set_xlabel('Episode')
axs[0, 0].set_ylabel('Total Reward')
axs[0, 0].legend()

axs[0, 1].hist(rewards_history, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
axs[0, 1].axvline(x=mean_reward, color='r', linestyle='--', label=f'Mean = {mean_reward:.1f}')
axs[0, 1].set_title('Reward Distribution')
axs[0, 1].set_xlabel('Total Reward')
axs[0, 1].set_ylabel('Count')
axs[0, 1].legend()

axs[1, 0].plot(lengths_history, color='peru', alpha=0.7)
axs[1, 0].axhline(y=mean_length, color='r', linestyle='--', label=f'Mean = {mean_length:.1f}')
axs[1, 0].set_title('Episode Length')
axs[1, 0].set_xlabel('Episode')
axs[1, 0].set_ylabel('Steps')
axs[1, 0].legend()

axs[1, 1].axis('off') # Hide the axes for this one
stats_text = (
    f"{'Random Policy Statistics':^30}\n"
    f"{'-'*30}\n"
    f"Mean reward   : {mean_reward:>10.2f}\n"
    f"Std reward    : {np.std(rewards_history):>10.2f}\n"
    f"Min reward    : {np.min(rewards_history):>10.2f}\n"
    f"Max reward    : {np.max(rewards_history):>10.2f}\n"
    f"Mean length   : {mean_length:>10.1f} steps\n"
    f"Success rate  : {success_rate:>10.1f}%"
)
axs[1, 1].text(0.5, 0.5, stats_text, family='monospace', fontsize=12,
              verticalalignment='center', horizontalalignment='center',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.2))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('baseline/baseline_stats.png')
plt.show()

env.close()
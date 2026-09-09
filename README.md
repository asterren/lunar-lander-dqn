# Lunar Lander DQN

## Overview

This repository documents a reinforcement learning class assignment using **Deep Q-Learning** on the `LunarLander-v3` environment.

The goal was to implement a Deep Q-Network agent that learns a landing policy for the Lunar Lander environment. The assignment involved building a random-policy baseline, implementing a DQN agent, training the agent, evaluating learning behavior, and comparing multiple hyperparameter variations.

This project focuses on reinforcement learning implementation, DQN training stability, experiment comparison, and analysis of learning curves.

## Task Description

The task was to train a DQN agent for `LunarLander-v3`.

The agent observes an 8-dimensional state space, including the lander’s position, velocity, angle, angular velocity, and leg contact flags. The action space contains 4 discrete actions:

- `0`: Do nothing
- `1`: Fire left orientation engine
- `2`: Fire main engine
- `3`: Fire right orientation engine

The goal is to land the spacecraft safely between the flags while using thrust efficiently.

## Methods

### Random Policy Baseline

First, I implemented a random-policy baseline to understand the environment and establish a comparison point.

The baseline ran for 100 episodes and collected:

- Episode rewards
- Episode lengths
- Mean reward
- Reward distribution
- Success rate

### DQN Agent

The DQN implementation includes:

- Neural network Q-function approximator
- Replay buffer
- Epsilon-greedy exploration
- Target network
- Periodic target network updates
- Training loop
- Evaluation without exploration
- Reward, loss, and Q-value tracking

## Experiments

I tested multiple variations to compare how different hyperparameters affected training stability and final performance.

### Baseline

The baseline used random actions and did not learn a policy. It served as a reference point for comparing trained agents.

### Variation 1: Higher Learning Rate

This variation used a higher learning rate. In my experiments, the reward increased faster, and this version was able to reach the +200 reward area during evaluation.

### Variation 2: Smaller Replay Buffer

This variation used a smaller replay buffer. The agent showed some improvement, but performance became less stable later. My analysis suggests that the smaller buffer caused the agent to focus too much on recent experiences and forget earlier successful landing behavior.

### Variation 3: Slower Target Network Update

This variation updated the target network less frequently. The learning behavior was weaker and less stable. The target network became too outdated, causing the policy network to chase old target values that no longer represented the current learning situation.

## Results

The random baseline had poor performance and a 0% success rate.

The DQN experiments showed that hyperparameters had a large effect on training behavior. Variation 1 performed the best in my experiments because it learned faster and reached the successful reward range more often. Variation 2 and Variation 3 showed weaker or less stable learning behavior.

## Analysis

From the experiments, I found that learning rate had the strongest impact on performance. If the learning rate was too low, the agent could not improve quickly enough. If other settings caused unstable learning, the agent could fail to learn a reliable landing strategy.

Replay buffer size also mattered. A smaller replay buffer could make the model focus too heavily on recent experiences and lose useful older experiences.

Target network update frequency also affected stability. If the target network was updated too rarely, the policy network learned from outdated targets. If updated too often, training could become unstable.

## Development Note

This was a class assignment completed under a course deadline. I used course materials, documentation, online references, and AI assistance as learning and debugging support while implementing and analyzing the DQN experiments. The code, experiments, result interpretation, and report were reviewed, adapted, and organized for the assignment.

## Limitations

- The experiments were limited by training time.
- The model was not tuned exhaustively.
- The results may vary between runs because reinforcement learning training can be unstable.
- The uploaded repository does not include large model checkpoint files.
- The project focuses on class assignment implementation and analysis rather than a production-ready RL system.

## What I Learned

Through this project, I practiced:

- Reinforcement learning workflow
- Deep Q-Learning implementation
- Gymnasium environment interaction
- Epsilon-greedy exploration
- Replay buffer implementation
- Target network updates
- Training loop design
- Hyperparameter comparison
- Reward, loss, and Q-value analysis
- Interpreting failure modes in RL training

This project helped me understand that reinforcement learning performance depends heavily on training stability, exploration, replay memory, target network updates, and careful interpretation of learning curves.

## Project Information

- Project type: Class assignment
- Topic: Reinforcement Learning / Deep Q-Learning
- Environment: LunarLander-v3
- Language: Python
- Libraries: Gymnasium, PyTorch, NumPy, Matplotlib
- Main focus: DQN implementation, training analysis, and hyperparameter comparison

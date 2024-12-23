Deep Reinforcement Learning Project - Policy Gradient Methods
This project implements various policy gradient methods for reinforcement learning, focusing on the CartPole environment. The project explores different approaches to training agents using policy gradients, from basic REINFORCE to more advanced actor-critic methods.
Project Overview
The project is structured into several key components:

CartPole Environment Implementation
Deep Neural Network Policy Design
REINFORCE Algorithm Implementation
State Space Modification & Markov Property Analysis
Actor-Critic Implementation

Requirements

Python 3.x
PyTorch
Gymnasium
NumPy
SciPy

Install dependencies:
bashCopypip install gymnasium
pip install numpy
pip install scipy
pip install torch
Components
1. CartPole Environment
The project uses a custom implementation of the CartPole environment with:

State space: position, angle, velocities
Action space: discrete (left/right)
Reward structure: +1 for each step, episode terminates on failure

2. Policy Network
The policy network architecture includes:

Input layer matching state space dimension
Hidden layers with ReLU activation
Output layer with softmax activation for action probabilities
Implemented using PyTorch's neural network modules

3. REINFORCE Algorithm
Implementation features:

Monte Carlo policy gradient estimation
Trajectory sampling
Discounted reward calculation
Policy gradient updates

Key parameters:

Learning rate: 0.001
Discount factor: 0.99
Episode horizon: 500

4. Modified Environment Analysis
Includes investigation of:

Reduced state space version
Impact on Markov property
Solutions for maintaining performance with limited state information

5. Actor-Critic Implementation
Extended architecture featuring:

Separate policy (actor) and value (critic) networks
Value function approximation
Advantage estimation
Parallel network training

Usage

Train the basic REINFORCE agent:

pythonCopypython train_reinforce.py

Run the trained agent:

pythonCopypython test_policy.py
Performance Metrics
The implementation achieves:

Consistent rewards of 500+ after training
Stable policy convergence
Robust performance across different random seeds

Project Structure
Copy├── cartpole_env.py      # Custom CartPole environment
├── models/
│   ├── actor.py         # Policy network
│   ├── critic.py        # Value network
├── algorithms/
│   ├── reinforce.py     # REINFORCE implementation
│   ├── actor_critic.py  # Actor-Critic implementation
└── utils/
    └── training.py      # Training utilities
Author
Elie KADOCHE (eliekadoche78@gmail.com)
Acknowledgments
This project was developed as part of a reinforcement learning practical session, building on foundational work in policy gradient methods for deep reinforcement learning.

# MAgent2 RL Final Project
## Overview
In this final project, you will develop and train a reinforcement learning (RL) agent using the MAgent2 platform. The task is to solve a specified MAgent2 environment `battle`, and your trained agent will be evaluated on all following three types of opponents:

1. Random Agents: Agents that take random actions in the environment.
2. A Pretrained Agent: A pretrained agent provided in the repository.
3. A Final Agent: A stronger pretrained agent, which will be released in the final week of the course before the deadline.

Your agent's performance should be evaluated based on reward and win rate against each of these models. You should control *blue* agents when evaluating.

# Result
My agent vs Random
<p align="center">
  <img src="assets/vs_random.gif" width="300" alt="My agent vs random agent" />
</p>

My agent vs Red.pt agent
<p align="center">
  <img src="assets/vs_redpt.gif" width="300" alt="My agent vs red.pt agent" />
</p>

My agent vs Final Red Agent
<p align="center">
  <img src="assets/vs_final_red.gif" width="300" alt="My agent vs final red agent" />
</p>

<h2 align="center">GGWP <3</h2>

Checkout a [Colab notebook](https://colab.research.google.com/drive/1qmx_NCmzPlc-atWqexn2WueqMKB_ZTxc?usp=sharing) for running this demo.

## Installation
clone this repo and install with
```
pip install -r requirements.txt
```

## Demos
See `main.py` for a starter code.

## Evaluation
Refer to `eval.py` for the evaluation code, you might want to modify it with your specific codebase.

## References

1. [MAgent2 GitHub Repository](https://github.com/Farama-Foundation/MAgent2)
2. [MAgent2 API Documentation](https://magent2.farama.org/introduction/basic_usage/)

For further details on environment setup and agent interactions, please refer to the MAgent2 documentation.

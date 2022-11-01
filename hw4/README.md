## Setup

You can run this code on your own machine or on Google Colab. 

1. **Local option:** If you choose to run locally, you will need to install some Python packages; see [installation.md](../hw1/installation.md) from homework 1 for instructions.

2. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badges below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeleydeeprlcourse/homework_fall2022/blob/master/hw4/cs285/scripts/run_hw4_mb.ipynb)

## Complete the code

The following files have blanks to be filled with your solutions from homework 1. The relevant sections are marked with `TODO: get this from Piazza'.

- [x] [agents/sac_agent.py](cs285/agents/sac_agent.py) 
- [x] [critics/sac_critic.py](cs285/critics/sac_critic.py)
- [x] [infrastructure/rl_trainer.py](cs285/infrastructure/rl_trainer.py)
- [x] [infrastructure/utils.py](cs285/infrastructure/utils.py)
- [x] [policies/sac_policy.py](cs285/policies/sac_policy.py)

You will then need to implement code in the following files:
- [x] Problem 1
  - [x] [agents/mb_agent.py](cs285/agents/mb_agent.py)
  - [x] [models/ff_model.py](cs285/models/ff_model.py)
  - [x] [policies/MPC_policy.py](cs285/policies/MPC_policy.py)
  - [x] [infrastructure/rl_trainer.py](cs285/infrastructure/rl_trainer.py)
  - [x] [infrasturcture/utils.py](cs285/infrastructure/utils.py)
- [x] Problem 2
  - [x] [policies/MPC_policy.py](cs285/policies/MPC_policy.py)
- [x] Problem 3
- [x] Problem 4
- [x] Problem 5
  - [x] [policies/MPC_policy.py](cs285/policies/MPC_policy.py)
- [ ] Problem 6
  - [ ] [agents/mbpo_agent.py](cs285/infrastructure/rl_trainer.py)
  - [ ] [infrastructure/rl_trainer.py](cs285/infrastructure/rl_trainer.py)


The relevant sections are marked with `TODO`.

You may also want to look through [scripts/run_hw4_mb.py](cs285/scripts/run_hw4_mb.py) and [scripts/run_hw4_mbpo.py](cs285/scripts/run_hw4_mbpo.py) (if running locally) or [scripts/run_hw4_mb.ipynb](cs285/scripts/run_hw4_mb.ipynb) and [scripts/run_hw4_mbpo.ipynb](cs285/scripts/run_hw4_mbpo.ipynb) (if running on Colab), though you will not need to edit this files beyond changing runtime arguments in the Colab notebook.

See the [assignment PDF](cs285_hw4.pdf) for more details on what files to edit.


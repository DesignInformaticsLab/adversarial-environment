## Adversarial Environment

### Overview
Attacking car racing in OpenAI Gym using adversarial attacks on environment. The fully trained agent and its associated environment wrappers, networks are taken from [pytorch_car_caring](https://github.com/xtma/pytorch_car_caring).

### Installation
Clone the repo and cd into directory.

```$ git clone https://github.com/Khrylx/PyTorch-RL.git```

```$ cd adversarial-environment```
### Requirements
Here are some of the dependencies that are required. For complete dependencies check ```requirements.txt``` file
- [pytorch 1.0.0](https://pytorch.org/)
- [gym 0.15.4](https://github.com/openai/gym)
- [Box2D 2.3.2](https://box2d.org)
- [visdom 0.1.8](https://github.com/facebookresearch/visdom)
- [numpy 1.18.1](https://numpy.org)
- [matplotlib 3.1.3](https://matplotlib.org)

Note: If you are facing errors related to Box2D while running, try installing ```Box2D-kengz  v2.3.3```

### Running
To train the attack, run ```python adv_attack_train.py```. To test the attack and render the environment, run ```python adv_attack_test.py --render```

### Disclaimer
This work is highly based on the following repo:
1. [xtma/pytorch_car_caring](https://github.com/xtma/pytorch_car_caring)
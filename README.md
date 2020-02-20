## Adversarial Environment

### Overview
Attacking car racing in OpenAI Gym using adversarial attacks on environment. The fully trained agent and its associated environment wrappers, networks are taken from [pytorch_car_caring](https://github.com/xtma/pytorch_car_caring).

### Installation
Clone the repo and cd into directory.

```$ git clone https://github.com/DesignInformaticsLab/adversarial-environment.git```

```$ cd adversarial-environment```
### Requirements
Here are some of the dependencies that are required. For complete dependencies check ```requirements.txt``` file
- [pytorch 1.4.0](https://pytorch.org/)
- [gym 0.15.4](https://github.com/openai/gym)
- [Box2D 2.3.2](https://box2d.org)
- [visdom 0.1.8](https://github.com/facebookresearch/visdom)
- [numpy 1.18.1](https://numpy.org)
- [matplotlib 3.1.3](https://matplotlib.org)
- [torchvision 0.5.0](https://pytorch.org/docs/stable/torchvision/index.html)
- [tensorboard 1.14.0](https://www.tensorflow.org/tensorboard)

Note: If you are facing errors related to Box2D while running, try installing ```Box2D-kengz  v2.3.3```. Also, some versions of libraries are upgraded to support tensorboard.

### Running
#### General attack (Level 0)
To train the attack, run ```python adv_attack_train.py --attack_type=general```. To test the attack and render the environment, run ```python adv_attack_test.py --render --attack_type=general```
#### Patch attack (Level 1)
To train the attack, run ```python adv_attack_train.py --attack_type=patch --patch_type=circle```. To test the attack and render the environment, run ```python adv_attack_test.py --render --attack_type=patch --patch_type=circle```. Currently supports only two types of patches **box** and **circle**.

### Disclaimer
This work is highly based on the following repo:
1. [xtma/pytorch_car_caring](https://github.com/xtma/pytorch_car_caring)
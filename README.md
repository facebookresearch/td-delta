# Separating Value Functions Across Time-Scales

Read the full paper: https://arxiv.org/abs/1902.01883 

```
@article{separatingvalues2019,
  title={Separating value functions across time-scales},
  author={Romoff, Joshua and Henderson, Peter and Touati, Ahmed and Olliver, Yann and Brunskill, Emma and Pineau, Joelle},
  journal={arXiv preprint arXiv:1902.01883},
  year={2019}
}

```

We based our code off of [ikostrikov's pytorch-rl repo](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr). 

```
@misc{pytorchrl,
  author = {Kostrikov, Ilya},
  title = {PyTorch Implementations of Reinforcement Learning Algorithms},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ikostrikov/pytorch-a2c-ppo-acktr}},
}
```

## Installation

### PyTorch

without cuda: 

```conda install pytorch=0.4.1 -c pytorch ```

with cuda: 

```conda install pytorch=0.4.1 cuda90 -c pytorch ``` 

(or cuda92, cuda80, cuda 75. depending on what you have installed)

### Baselines for Atari preprocessing
``` git clone https://github.com/openai/baselines.git ```
``` cd baselines ```
``` pip install -e . ```

### Other requirements
``` pip install -r requirements.txt ```

## Replicating results

To replicate our atari experiments run

``` python main.py --run-index [0-720] ```

## Visualization

To visualize performance (requires Visdom) first create a visdom server:

``` python -m visdom.server```

Then run: 

``` python visualize.py ```


## License
This repo is CC-BY-NC licensed, as found in the LICENSE file.


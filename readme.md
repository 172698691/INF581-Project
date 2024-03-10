# INF581 - Apprentissage Automatique Avancé et Agents Autonomes Project

## UAV Navigation with DDPG

### Create Environment

We use PyTorch 2.2.1 with CUDA 12.1. Install PyTorch using instructions from [PyTorch Install Guide](https://pytorch.org/get-started/locally/)

crete new environment by 


```shell
# install PyTorch following https://pytorch.org/get-started/locally/

# later versions changed the API, so we use an older version
pip3 install gym==0.21.0

pip3 install pygame
```

### Run

Run the experiment by:

```shell
python main.py --param 1 1 1 1 -0.5 --render 1 --train_time_steps 10000 --test_episodes 100
```

### Cite

We referred to [Moritz Schneider's project](https://github.com/schneimo/ddpg-pytorch) in the implementation of the DDPG algorithm.
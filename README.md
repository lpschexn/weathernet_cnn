# weathernet_cnn

This repo trains and tests WeatherNet CNN using [LilaBlocks](https://github.com/TheCodez/pytorch-LiLaNet) and the [WeatherNet architecture](https://arxiv.org/abs/1912.03874).

A comparison of WeatherNet to DSOR and DROR are documented in "weathernet_v_DSOR.pdf". This document is an amended version of Kurup and Bos' manuscript ["DSOR: A Scalable Statistical Filter for Removing Falling Snow from LiDAR Point Clouds in Severe Winter Weather"](https://arxiv.org/abs/2109.07078).

## Setup

We use the [PyTorch](https://pytorch.org/) framework. This has been testing on Ubuntu 20.04 with python3.8.
To download PyTorch use python's `pip` tool: `pip3 install torch torchvision torchsummary`.

## How to use

1. Download WADS dataset and place all sequences in a desired directory
2. Go to [train.py](train.py) and update `TRAIN_DATA_PATH` to the directory that contains all the training sequences (without trailing back-slash)
3. Run `python3 train.py`
4. Go to [test.py](test.py) and update `TEST_DATA_PATH` to the directory that contains all the testing sequences
5. Run `python3 test.py`

## Run tests on stats

To ensure stats are calculated correctly, run `python3 -m pytest test_stats.py` to see results of unit tests.
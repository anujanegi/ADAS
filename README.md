# ADAS
Advanced Driver Assist System

## Structure of the project

- The **extraction** module contains utility to ge the facial features;
- The **training** module contains neural network definition, dataset, and script to train the neural network;
- The **tests** module contains tests for the trained model.

## Setup

**Python 3** is required.

- Install [Git Large File Storage](https://git-lfs.github.com/), either manually or do `git lfs install` 
- Clone the repository with submodules: `git lfs clone --recurse-submodules https://github.com/anujanegi/ADAS.git`
- Install the dependencies : `pip install -r requirements.txt`.

#### Training the network
The **train.py** script in /training can be run to train the LeNet neural netwok.
`python3 ./train.py`

#### Go live 
The **run.py** script can be run to get started.
`python3 ./run.py`

import numpy as np
import torch
import json
from Network_Builder import Network_Builder

class Trainer:

    def __init__(self):

        with open('config.json' ,'r') as fp:
            self.config = json.load(fp)

        self.net = Network_Builder(self.config)

if __name__ == '__main__':
    print('\nBEGIN THE LEARNING\n')
    trainer = Trainer()

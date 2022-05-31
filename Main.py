
from Common_Imports import *
from Network_Builder import Network_Builder
from Datasets import Obj_Dataset
from torch.utils.data import Dataset, DataLoader

class Trainer:

    def __init__(self):

        with open('config.json', 'r') as fp:
            self.config = json.load(fp)

        # Network initialization
        self.net = Network_Builder(self.config)
        print(self.net)

        # Dataset and dataloader initialization
        dataset = Obj_Dataset(self.config)
        self.train_dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=2, persistent_workers=False, pin_memory=True, drop_last=False)

    def run_episode(self, test=False):

        if not test:
            dataloader = self.train_dataloader

        for en_idx, en_val in enumerate(dataloader):

            for i in range(en_val[0].size(0)):
                print(en_val[1][i])
                plt.imshow(en_val[0][i])
                plt.show()


if __name__ == '__main__':
    print('\nBEGIN THE LEARNING\n')
    trainer = Trainer()
    trainer.run_episode()

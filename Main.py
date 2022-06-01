
from Common_Imports import *
from Network_Builder import Network_Builder
from Datasets import Obj_Dataset
from Loss import YOLO_Loss
from torch.utils.data import Dataset, DataLoader

class Trainer:

    def __init__(self):

        with open('config.json', 'r') as fp:
            self.config = json.load(fp)

        self.num_epochs = self.config['num_epochs']

        # Network initialization
        self.net = Network_Builder(self.config)
        print(self.net)



        pytorch_total_params = sum(p.numel() for p in self.net.parameters())
        training_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('Trainable parameters:', f'{training_params:,}')
        print('Total number of parameters is', f'{pytorch_total_params:,}', '\n')

        self.device = torch.device("cuda:0")
        torch.set_num_threads(1)

        self.S = 7
        self.loss = YOLO_Loss(self.S, self.config, self.device)


        # Dataset and dataloader initialization
        print('Training set')
        dataset = Obj_Dataset(self.config)
        print('Validation set')
        val_dataset = Obj_Dataset(self.config, False)

        self.train_dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=3, persistent_workers=False, pin_memory=True, drop_last=False)
        self.val_dataloader  = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=3, persistent_workers=False, pin_memory=True, drop_last=False)

        self.net.to(self.device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.config['lr'], momentum=0.9)

        if path.exists(self.config['model_path']):
            print('Found older model, will attempt loading it \n')
            loaded_model = torch.load(self.config['model_path'])
            self.net.load_state_dict(loaded_model['model_state_dict'])
            self.optimizer.load_state_dict(loaded_model['optimizer_state_dict'])
            for opt_i in range(len(self.optimizer.param_groups)):
                self.optimizer.param_groups[opt_i]['lr'] = self.config['lr']
                print('New learning rate:', self.optimizer.param_groups[opt_i]['lr'])

    def save_model(self):
        torch.save({
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config
                }, self.config['model_path'])


    def run_episode(self, test=False):

        if not test:
            print('Training')
            dataloader = self.train_dataloader
            self.net.train()
        else:
            print('Validation')
            dataloader = self.val_dataloader
            self.net.eval()

        total_loss = 0.

        for en_idx, en_val in enumerate(dataloader):
            # for i in range(en_val[0].size(0)):
            #     print(en_val[1][i])
            #     plt.imshow(en_val[0][i])
            #     plt.show()
            # if en_idx > 16 and not test: break

            if test:
                with torch.no_grad():
                    x = en_val[0].to(self.device, non_blocking=True)
                    y = en_val[1]

                    output = self.net(x)
                    loss = self.loss(output, y, x)
                    total_loss += loss.item()
            else:

                x = en_val[0].to(self.device, non_blocking=True)
                y = en_val[1]

                self.optimizer.zero_grad()
                output = self.net(x)
                loss = self.loss(output, y, x)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        print('Mean epoch loss:', total_loss/(en_idx+1))
        self.save_model()

if __name__ == '__main__':
    print('\nBEGIN THE LEARNING\n')
    trainer = Trainer()
    for i in range(trainer.num_epochs):

        start = default_timer()

        print('\nEpisode:', i)
        trainer.run_episode()
        trainer.run_episode(True)

        runtime = default_timer() - start
        print('Runtime:', runtime)

        print('Time remaining:', runtime * (trainer.num_epochs-i-1) / 60., 'minutes')

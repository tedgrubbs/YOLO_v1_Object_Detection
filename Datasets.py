from Common_Imports import *

from torch.utils.data import Dataset
from PIL import Image as im

class Obj_Dataset(Dataset):

    def __init__(self, config, train=True):

        self.train = train

        if train:
            self.img_path = config['img_path']
            self.dataset_path = config['dataset_path']
        else:
            self.img_path = config['val_img_path']
            self.dataset_path = config['val_dataset_path']

        self.images = listdir(self.img_path)
        annotations = listdir(self.dataset_path)
        self.annotations = np.zeros((len(annotations), 10, 4))

        for i in range(len(annotations)):
            index = int(annotations[i].split('.')[0])
            data = np.load(self.dataset_path + annotations[i])
            for obj in range(data.shape[0]):
                self.annotations[index][obj] = data[obj]

        print('\nDone intializing dataset')
        print('Number of datapoints:', len(annotations), '\n')


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        with im.open(self.img_path+str(idx)) as new_image:
            img = torch.from_numpy(np.asarray(new_image)).float().unsqueeze(0)
        anno = torch.from_numpy(self.annotations[idx]).long()

        return img, anno

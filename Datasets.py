from Common_Imports import *

from torch.utils.data import Dataset
from PIL import Image as im

class Obj_Dataset(Dataset):

    def __init__(self, config):

        self.img_path = config['img_path']
        self.dataset_path = config['dataset_path']

        self.images = listdir(config['img_path'])
        annotations = listdir(config['dataset_path'])
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

        img = torch.from_numpy(np.asarray(im.open(self.img_path+str(idx)))).float().unsqueeze(0)
        anno = torch.from_numpy(self.annotations[idx]).long()

        return img, anno

import numpy as np
from torch.utils.data import Dataset
import torch
from albumentations.pytorch.functional import img_to_tensor


class VideoDataset(Dataset):
    def __init__(self,
                 annotations,
                 label_smoothing=0.01,
                 normalize={"mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]},
                 mode="train",
                 balance=True,
                 transforms=None,
                 num_classes=2,
                 size = 256
                 ):
        super(VideoDataset).__init__()
        self.mode = mode
        self.label_smoothing = label_smoothing
        self.normalize = normalize
        self.transforms = transforms
        self.balance = balance
        self.num_classes = num_classes
        self.size = size
        if self.balance:
            self.data = [[x for x in annotations if x[1] == lab] for lab in [i for i in range(num_classes)]]
            print(len(self.data[0]),len(self.data[1]))
        else:
            self.data = [annotations]
            print("all: %d"%len(self.data[0]))
        self.lost = []

    def __len__(self) -> int:
        return max([len(subset) for subset in self.data])

    def __getitem__(self, index: int):
        if self.balance:
            img_paths=[]
            labs=[]
            imgs=[]

            for i in range(self.num_classes):
                safe_idx = index % len(self.data[i])
                img_path = self.data[i][safe_idx][0]
                img = self.load_sample(img_path)
                lab = self.data[i][safe_idx][1]
                img_paths.append(img_path)
                labs.append(lab)
                imgs.append(img)

            return torch.tensor(labs,dtype=torch.long), torch.cat([imgs[i].unsqueeze(0) for i in range(len(imgs))]),\
                   img_paths

        else:
            lab = self.data[0][index][1]
            img_path = self.data[0][index][0]
            img = self.load_sample(img_path)
            lab = torch.tensor(lab, dtype=torch.long)
            return lab, img, img_path

    def load_sample(self,path):
        frames = np.load(path)
        frames = [self.run_transform(frame) for frame in frames]
        frames = torch.cat([frames[i].unsqueeze(0) for i in range(len(frames))])
        #frames = img_to_tensor(np.array(frames),self.normalize)
        return frames

    def run_transform(self,img):
        img_aug = self.transforms(img, return_torch=False).data[0]
        img_aug = img_to_tensor(img_aug,self.normalize)
        return img_aug


def collate_function(data):
    transposed_data = list(zip(*data))
    lab, img, img_path = transposed_data[0], transposed_data[1], transposed_data[2]
    img = torch.stack(img, 0)
    lab = torch.stack(lab, 0)
    return lab, img, img_path

# if __name__ == '__main__':
#
#     datapath = '/disk/data/UCF-101'
#     train_dataloader = \
#         DataLoader( VideoDataset(datapath, mode='train'), batch_size=10, shuffle=True, num_workers=0)
#     for step, (buffer, label) in enumerate(train_dataloader):
#         print("label: ", label)

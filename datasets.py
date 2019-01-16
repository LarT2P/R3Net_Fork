import os
import os.path

import torch.utils.data as data
from PIL import Image


def make_dataset(root):
    if 'OPPO' in root:
        img_path = os.path.join(root, 'Train', 'Image_after')
        gt_path = os.path.join(root, 'Train', 'Mask')
    elif 'DUTS' in root:
        img_path = os.path.join(root, 'Train', 'Image')
        gt_path = os.path.join(root, 'Train', 'Mask')
    else:
        img_path = os.path.join(root, 'Image')
        gt_path = os.path.join(root, 'Mask')
    
    img_list = [os.path.splitext(f)[0]
                for f in os.listdir(img_path) if f.endswith('.jpg')]
    return [(os.path.join(img_path, img_name + '.jpg'),
             os.path.join(gt_path, img_name + '.png')) for img_name in img_list]


class ImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None,
                 target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        
        # PIL中有九种不同模式,分别为1,L,P,RGB,RGBA,CMYK,YCbCr,I,F
        # 分别代表二值,灰度,8位彩色...
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    def __len__(self):
        return len(self.imgs)

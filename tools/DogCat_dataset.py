# -*- coding: utf-8 -*-

'''
@Time    : 2020/6/9 23:08
@Author  : HHNa
@FileName: DogCat_dataset.py
@Software: PyCharm
 
'''

import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
DogCat_label = {"dog": 0, "cat": 1}


class DogCatDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        猫狗大战分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = {"dog": 0, "cat": 1}
        # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')  # 0-255

        if self.transform is not None:
            img = self.transform(img)  # 在治理做transform，转化为tensor等等

        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod   # 静态方法无需self
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('jpg'), img_names))
                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = DogCat_label[sub_dir]
                    data_info.append((path_img, int(label)))
        return data_info
import numpy as np
# 读取数据
import scipy.misc
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import imageio


class CUB(Dataset):
    def __init__(self, root, is_train=True, data_len=None, transform=None, target_transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.target_transform = target_transform
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        # 图片索引
        img_name_list = []
        for line in img_txt_file:
            # 最后一个字符为换行符
            img_name_list.append(line[:-1].split(' ')[-1])

        # 标签索引，每个对应的标签减１，标签值从0开始
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)

        # 设置训练集和测试集
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))

        # zip压缩合并，将数据与标签(训练集还是测试集)对应压缩
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，
        # 然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。
        # 我们可以使用 list() 转换来输出列表

        # 如果 i 为 1，那么设为训练集
        # １为训练集，０为测试集
        # zip压缩合并，将数据与标签(训练集还是测试集)对应压缩
        # 如果 i 为 1，那么设为训练集
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

        train_label_list = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        test_label_list = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
        if self.is_train:
            # scipy.misc.imread 图片读取出来为array类型，即numpy类型
            self.train_img = [imageio.imread(os.path.join(self.root, 'images', train_file)) for train_file in
                              train_file_list[:data_len]]
            # 读取训练集标签
            self.train_label = train_label_list
        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'images', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = test_label_list

    # 数据增强
    def __getitem__(self, index):
        # 训练集
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
        # 测试集
        else:
            img, target = self.test_img[index], self.test_label[index]

        if len(img.shape) == 2:
            # 灰度图像转为三通道
            img = np.stack([img] * 3, 2)
        # 转为 RGB 类型
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


if __name__ == '__main__':
    '''
    dataset = CUB(root='./CUB_200_2011')

    for data in dataset:
        print(data[0].size(),data[1])

    '''
    # 以pytorch中DataLoader的方式读取数据集
    transform_train = transforms.Compose([
        transforms.Resize(512),
        transforms.RandomResizedCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(512),
        transforms.RandomResizedCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = CUB(root='D:\PycharmProject\ResNet-50\CUB_200_2011\CUB_200_2011', is_train=True, transform=transform_train)
    train_len = len(train_dataset)
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

    test_dataset = CUB(root='D:\PycharmProject\ResNet-50\CUB_200_2011\CUB_200_2011', is_train=False, transform=transform_test)
    test_len = len(train_dataset)
    print(len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0)


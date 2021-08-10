import os

import torch
from torch import nn, optim
from torch.utils import model_zoo
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm

import read_data

import model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_len, train_dataloader = read_data.train_data_load()
    test_len, test_dataloader = read_data.test_data_load()

    print("训练数据集长度为:{}".format(train_len))
    print("测试数据集长度为:{}".format(test_len))

    # 创建模型
    # ---------------------
    resnet_50 = model.ResNet50
    url = "https://download.pytorch.org/models/resnet50-19c8e357.pth"
    model_dict = resnet_50.state_dict()
    pretrained_dict = model_zoo.load_url(url, progress=False)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    resnet_50.load_state_dict(model_dict)
    in_channel = resnet_50.fc.in_features
    resnet_50.fc = nn.Linear(in_channel, 200)
    resnet_50 = resnet_50.to(device)
    # -------------------------
    # resnet_50 = models.resnet50(pretrained=False)
    # model_weight_path = "resnet50-19c8e357.pth"

    # resnet_50.load_state_dict(torch.load(model_weight_path))
    # in_channel = resnet_50.fc.in_features
    # resnet_50.fc = nn.Linear(in_channel, 200)
    # resnet_50 = resnet_50.to(device)
    # ---------------------
    # 损失函数
    loss_fun = nn.CrossEntropyLoss().to(device)

    # 优化器
    learn_rate = 0.002
    optimizer = optim.SGD(resnet_50.parameters(), lr=learn_rate, weight_decay=0.00005, momentum=0.9)

    # 训练轮数
    epochs = 80

    writer = SummaryWriter("logs")
    best_acc = 0


    for epoch in range(epochs):
        print("epoch:{}".format(epoch))
        # train
        resnet_50.train()
        if epoch == 20:
            learn_rate = 0.001
        elif epoch == 30:
            learn_rate = 0.0005
        elif epoch == 50:
            learn_rate = 0.0001
        optimizer = optim.SGD(resnet_50.parameters(), lr=learn_rate, weight_decay=0.00005, momentum=0.9)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        train_bar = tqdm(train_dataloader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = resnet_50(images.to(device))
            loss = loss_fun(logits, labels.to(device))
            loss.backward()
            optimizer.step()

        # testdate
        test_acc = 0.0
        train_acc = 0.0
        test_loss = 0.0
        train_loss = 0.0
        resnet_50.eval()
        with torch.no_grad():
            test_bar = tqdm(test_dataloader)
            for train_data in train_bar:
                train_images, train_labels = train_data
                train_outputs = resnet_50(train_images.to(device))
                tmp_train_loss = loss_fun(train_outputs, train_labels.to(device))
                train_acc += (train_outputs.argmax(1) == train_labels).sum().item()
                train_loss += tmp_train_loss.item()

            for test_data in test_bar:
                test_images, test_labels = test_data
                test_outputs = resnet_50(test_images.to(device))
                tmp_test_loss = loss_fun(test_outputs, test_labels.to(device))
                test_acc += (test_outputs.argmax(1) == test_labels).sum().item()
                test_loss += tmp_test_loss.item()

        train_accurate = train_acc / train_len
        test_accurate = test_acc / test_len
        print("train_acc = {}".format(train_accurate))
        print("test_acc = {}".format(test_accurate))
        writer.add_scalar("Train_loss", train_loss / train_len, epoch + 1)
        writer.add_scalar("Test_acc", train_accurate, epoch + 1)
        writer.add_scalar("Test_loss", test_loss / test_len, epoch + 1)
        writer.add_scalar("Test_acc", test_accurate, epoch + 1)
        if test_accurate > best_acc:
            best_acc = test_accurate

    print("best_acc = {}".format(best_acc))


if __name__ == '__main__':
    main()

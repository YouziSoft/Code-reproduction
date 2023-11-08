import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# 创建一个 ResNet 模型，加载预训练权重
net = models.resnet34(pretrained=True)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
# 切换到评估模式
net.eval()

for i in range(len(train_images)):
    image_array = train_images[i]
    image = Image.fromarray(image_array)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 图像预处理
    im = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])(image)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)

    # 读取标签文件
    labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)

    str_label_orig = labels[int(label_orig)].split(',')[0]
    str_label_pert = labels[int(label_pert)].split(',')[0]

    print("Original label = ", str_label_orig)
    print("Perturbed label = ", str_label_pert)

    def clip_tensor(A, minv, maxv):
        A = torch.max(A, minv * torch.ones(A.shape))
        A = torch.min(A, maxv * torch.ones(A.shape))
        return A

    clip = lambda x: clip_tensor(x, 0, 255)

    # 创建图像转换管道
    tf = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
        transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
        transforms.Lambda(clip),
        transforms.ToPILImage(),
        transforms.CenterCrop(224)
    ])


    figure_save_path = "bd"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建

    pert_image_tensor = torch.from_numpy(pert_image.cpu()[0].numpy())
    plt.figure()
    plt.imshow(tf(pert_image_tensor))
    plt.title(str_label_pert)
    plt.savefig(os.path.join(figure_save_path , f'{i}.jpg'))#第一个是指存储路径，第二个是图片名字
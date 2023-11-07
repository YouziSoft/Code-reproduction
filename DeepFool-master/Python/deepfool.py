import collections
import numpy as np
from torch.autograd import Variable
import torch as torch
import copy

import torch

def zero_gradients(x):
    """
    重置梯度为零
    :param x: PyTorch张量或张量列表
    """
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):
    """
    :param image: 输入图像，大小为 HxWx3
    :param net: 神经网络（输入：图像，输出：softmax激活值之前的值）
    :param num_classes: 类别数量，限制测试的类别数量，默认为 10
    :param overshoot: 作为终止条件以防止微小更新的过度（默认为 0.02）
    :param max_iter: deepfool 最大迭代次数（默认为 50）
    :return: 最小扰动、所需的迭代次数、新的估计标签和扰动后的图像
    """
    # 检查是否可以使用 GPU
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        # 将输入图像和神经网络切换到 GPU
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    # 获取原始图像的网络输出
    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    # 选择前 num_classes 个类别
    I = I[0:num_classes]
    label = I[0]
    # 获取输入图像的形状
    input_shape = image.cpu().numpy().shape
    # 创建扰动图像的副本
    pert_image = copy.deepcopy(image)
    # 初始化 w 和 r_tot
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)
    loop_i = 0
    # 创建一个变量用于迭代
    x = Variable(pert_image[None, :], requires_grad=True)
    # 前向传播网络以获取原始图像的激活值
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:
        pert = np.inf

        # 对原始类别的梯度
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            # 对当前类别的梯度
            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # 计算 w_k：我们所计算的梯度向量，即告诉我们扰动的方向
            # f_k：表示当前类别 k 的网络激活值与原始类别的激活值之间的差异。这个差异越大，表示网络更容易将输入分为类别 k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            # 计算 pert_k,
            # 用于衡量相对于当前类别 k 的网络激活值 f_k 的绝对差异（abs(f_k)) 与对应的梯度向量 w_k 的范数（np.linalg.norm(w_k.flatten())）之比。
            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
            # 选择最小的 pert_k
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # 计算 r_i ：这个扰动向量表示了如何在输入空间中微调原始输入以最大限度地改变网络对于类别 k 的分类
        #  r_tot：r_tot 是一个累积扰动向量，它在每次迭代时都会被更新
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        #这里就是每次的相加
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)

        # 更新变量以进行下一次迭代
        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1
    #1 + overshoot 用于缩放每次的扰动，其中 overshoot 是作为终止条件的小值，以避免微小更新的过度
    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image


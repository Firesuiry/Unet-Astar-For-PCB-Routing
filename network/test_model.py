import os

import cv2
import numpy as np
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from network.data_loader import MyDataset
import torch

from network.unet import ResNetUNet


def test():
    a = r'D:\develop\PCB\network\dataset\\'
    b = r'/data/yinshiyuan/project/pcb/network/dataset/'
    dataset = MyDataset(a, 100)
    train_set, val_set = dataset.get_train_and_test_Dataset(0.1)
    batch_size = 4

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    num_class = 2
    model = ResNetUNet(num_class).to(device)

    # freeze backbone layers
    # for l in model.base_layers:
    #    for param in l.parameters():
    #        param.requires_grad = False

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    # load model
    model.load_state_dict(torch.load('best_val_model.pth', map_location=device))

    test_loader = dataloaders['val']
    for batch_index in range(20):
        inputs, labels = next(iter(test_loader))
        inputs = inputs.to(device)
        labels = labels.to(device)
        pred = model(inputs)

        for index in range(batch_size):
            dir_path = 'test_result/test_{}/'.format(index + batch_index * batch_size)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            # draw pic of input and label and pred
            input_data = inputs[index].cpu().numpy()
            label_data = labels[index].cpu().numpy()
            pred_data = pred[index].cpu().detach().numpy()
            for i in range(input_data.shape[0]):
                draw_data = input_data[i] / np.max(input_data[i]) * 255
                draw_data = draw_data.astype(np.uint8)
                cv2.imwrite(dir_path + 'input_{}_{}.jpg'.format(index, i), draw_data)
            for i in range(label_data.shape[0]):
                draw_data = label_data[i] * 255
                draw_data = draw_data.astype(np.uint8)
                cv2.imwrite(dir_path + 'label_{}_{}.jpg'.format(index, i), draw_data)
            for i in range(pred_data.shape[0]):
                draw_data = 1 / (1 + np.exp(-pred_data))
                draw_data = (draw_data[i] - np.min(draw_data[i])) / (np.max(draw_data[i]) - np.min(draw_data[i])) * 255
                draw_data = draw_data.astype(np.uint8)
                cv2.imwrite(dir_path + 'pred_{}_{}.jpg'.format(index, i), draw_data)
    ...


if __name__ == '__main__':
    test()

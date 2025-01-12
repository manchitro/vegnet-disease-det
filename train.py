import os
import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchsummary import summary
from torchvision import transforms

from retinanet import mobilenetv2fpn, resnetsfpn
from retinanet.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import csv_eval

print('pytorch version: {}'.format(torch.__version__))
# assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--img_dir', help='Path to folder containing images')

    parser.add_argument('--model', help='Backbone to use, must be one of resnet 18, 34, 50, 101, 152 or mobilenetv2', type=str, default='mobilenetv2')
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--batchsize', help='Batch size', type=int, default=2)

    parser = parser.parse_args(args)

    if parser.csv_train is None:
        raise ValueError('Must provide --csv_train when training on CSV,')

    if parser.csv_classes is None:
        raise ValueError('Must provide --csv_classes when training on CSV,')

    dataset_train = CSVDataset(img_dir=parser.img_dir,train_file=parser.csv_train, class_list=parser.csv_classes,
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    if parser.csv_val is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        dataset_val = CSVDataset(img_dir=parser.img_dir,train_file=parser.csv_val, class_list=parser.csv_classes,
                                    transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batchsize, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler, shuffle=True)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.model == 'resnet18':
        network = resnetsfpn.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.model == 'resnet34':
        network = resnetsfpn.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.model == 'resnet50':
        network = resnetsfpn.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.model == 'resnet101':
        network = resnetsfpn.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.model == 'resnet152':
        network = resnetsfpn.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.model == 'mobilenetv2':
        network = mobilenetv2fpn.mobilenetv2FPN(num_classes=dataset_train.num_classes(), pretrained=False)
    else:
        raise ValueError('Unsupported model, must be one of resnet 18, 34, 50, 101, 152 or mobilenetv2')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            network = network.cuda()

    if torch.cuda.is_available():
        network = torch.nn.DataParallel(network).cuda()
    else:
        network = torch.nn.DataParallel(network)

    network.training = True
    # print(network)

    optimizer = optim.Adam(network.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    network.train()
    # retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        network.train()
        # retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = network([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = network([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, network)

        scheduler.step(np.mean(epoch_loss))

        torch.save(network.module, os.path.join('snapshots', 'epoch_' + str(epoch_num) + '.pt'))

    network.eval()

    torch.save(network, os.path.join('saved_models', 'model_final.pt'))


if __name__ == '__main__':
    main()

import os
import csv
import datetime
import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchsummary import summary
from torchvision import transforms

from retinanet import mobilenetv2fpn, resnetsfpn, train_metrics
from retinanet.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from retinanet.visualize_single_image import detect_image
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

    parser.add_argument('--debug', help='Debug mode', action='store_true')

    parser = parser.parse_args(args)

    if parser.debug:
        print('Debug mode is on')
        parser.csv_train = 'small_' + parser.csv_train
        parser.csv_val = 'small_' + parser.csv_val

    if not os.path.exists('out'):
        os.makedirs('out')

    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    timestamp = str(time_string)

    exp_out_dir = 'out/{}_{}exp_{}'.format(parser.model, 'debug_' if parser.debug else '', timestamp)
    if not os.path.exists(exp_out_dir):
        os.makedirs(exp_out_dir)

    snapshots_folder = os.path.join(exp_out_dir, 'snapshots')
    if not os.path.exists(snapshots_folder):
        os.makedirs(snapshots_folder)

    # Save train history
    train_history_cols = ['epoch', 'train_c_loss', 'train_r_loss', 'running_loss']
    train_csv_file_path = os.path.join(exp_out_dir, 'train_history.csv')
    with open(train_csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(train_history_cols)

    # train_metrics_history_cols = ['epoch', 'label', 'tp', 'fp', 'eval_mAP', 'eval_precision', 'eval_recall']
    # train_metrics_csv_file_path = os.path.join(exp_out_dir, 'train_metric_history.csv')
    # with open(train_metrics_csv_file_path, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(train_metrics_history_cols)
    # eval_history_cols = ['epoch', 'label', 'tp', 'fp', 'eval_mAP', 'eval_precision', 'eval_recall']

    # Save eval history
    eval_history_cols = ['epoch', 'label', 'tp', 'fp', 'eval_mAP', 'eval_precision', 'eval_recall']
    eval_csv_file_path = os.path.join(exp_out_dir, 'eval_history.csv')
    with open(eval_csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(eval_history_cols)

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
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

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
    if dataset_val:
        print('Num validation images: {}'.format(len(dataset_val)))       

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

            except Exception as e:
                print(e)
                continue

        snapshot_path = os.path.join(snapshots_folder, 'epoch_' + str(epoch_num) + '.pt')
        torch.save(network.module, snapshot_path)

        # train_mAP = train_metrics.evaluate(dataset_train, torch.load(os.path.join(snapshots_folder, 'epoch_' + str(epoch_num) + '.pt')), save_path=exp_out_dir, epoch=epoch_num, csv_file_path=train_metrics_csv_file_path)
        network.training = False
        network.eval()

        detect_image(image_path=os.path.join(parser.img_dir, 'vis_test_train'), model=network, class_list=parser.csv_classes, exp_out_dir=exp_out_dir)
        
        with open(train_csv_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch_num, classification_loss.item(), regression_loss.item(), np.mean(loss_hist)])
            # history_cols = ['epoch', 'train_c_loss', 'train_r_loss', 'running_loss']

        if parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, torch.load(os.path.join(snapshots_folder, 'epoch_' + str(epoch_num) + '.pt')), save_path=exp_out_dir, epoch=epoch_num, csv_file_path=eval_csv_file_path)

            detect_image(image_path=os.path.join(parser.image_dir, 'vis_test_val'), model=network, class_list=parser.csv_classes, exp_out_dir=exp_out_dir)

        scheduler.step(np.mean(epoch_loss))

    network.eval()

    torch.save(network, os.path.join('saved_models', 'model_final.pt'))


if __name__ == '__main__':
    main()

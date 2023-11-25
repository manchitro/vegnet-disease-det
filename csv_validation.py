import os
import datetime
import csv
import argparse
import torch
from torchvision import transforms
from retinanet import csv_eval

# from retinanet import model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval

# assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_annotations_path', help='Path to CSV annotations')
    parser.add_argument('--model_path', help='Path to model', type=str)
    parser.add_argument('--images_path',help='Path to images directory',type=str)
    parser.add_argument('--class_list_path',help='Path to classlist csv',type=str)
    parser.add_argument('--iou_threshold',help='IOU threshold used for evaluation',type=str, default='0.5')
    parser = parser.parse_args(args)

    if not os.path.exists('out'):
        os.makedirs('out')

    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    timestamp = str(time_string)

    val_out_dir = 'out/{}_val_{}'.format(parser.model, timestamp)
    if not os.path.exists(val_out_dir):
        os.makedirs(val_out_dir)

	# Save eval history
    eval_history_cols = ['epoch', 'label', 'tp', 'fp', 'eval_mAP', 'eval_precision', 'eval_recall']
    eval_csv_file_path = os.path.join(val_out_dir, 'eval_history.csv')
    with open(eval_csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(eval_history_cols)

    dataset_val = CSVDataset(parser.images_path, parser.csv_annotations_path, parser.class_list_path, transform=transforms.Compose([Normalizer(), Resizer()]))
    # Create the model
    #retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    retinanet=torch.load(parser.model_path)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        #retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    # retinanet.module.freeze_bn()

    print(csv_eval.evaluate(dataset_val, retinanet, iou_threshold=float(parser.iou_threshold), save_path=val_out_dir, csv_file_path=eval_csv_file_path, epoch=0))

if __name__ == '__main__':
    main()

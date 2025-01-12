# pytorch-retinanet

Pytorch implementation of RetinaNet object detection with ResNet and MobileNetV2 backbones as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

This implementation is primarily designed to be easy to read and simple to modify.

## Results
Currently, this repo achieves 33.5% mAP at 600px resolution with a Resnet-50 backbone. The published result is 34.0% mAP. The difference is likely due to the use of Adam optimizer instead of SGD with weight decay.

## Installation

1) Clone this repo

2) Install the required packages:

```
apt-get install tk-dev python-tk
```

3) Install the python packages:
	
```
pip install pandas
pip install opencv-python
pip install requests
```

## Training

The network can be trained using the `train.py` script.

For training using a custom dataset, with annotations in CSV format (see below), use

```
python train.py --dataset csv --csv_train <path/to/train_annots.csv>  --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv>
```

Note that the --csv_val argument is optional, in which case no validation will be performed.

## Pre-trained model

A pre-trained model is available at: 
- https://drive.google.com/open?id=1yLmjq3JtXi841yXWBxst0coAgR26MNBS (this is a pytorch state dict)

The state dict model can be loaded using:

```
retinanet = model.resnet50(num_classes=dataset_train.num_classes(),)
retinanet.load_state_dict(torch.load(PATH_TO_WEIGHTS))
```

## Validation

For CSV Datasets (more info on those below), run the following script to validate:

`python csv_validation.py --csv_annotations_path path/to/annotations.csv --model_path path/to/model.pt --images_path path/to/images_dir --class_list_path path/to/class_list.csv   (optional) iou_threshold iou_thres (0<iou_thresh<1) `

It produces following resullts:

```
label_1 : (label_1_mAP)
Precision :  ...
Recall:  ...

label_2 : (label_2_mAP)
Precision :  ...
Recall:  ...
```

You can also configure csv_eval.py script to save the precision-recall curve on disk.



## Visualization

To visualize the network detection with a CSV dataset, use `visualize.py`:

```
python visualize.py --dataset csv --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv> --model <path/to/model.pt>
```
This will visualize bounding boxes on the validation set.

## Model

The retinanet model uses a resnet backbone. We have also developed a MobileNetV2 backbone. You can set the model to use, using the --model argument. Model must be one of 
resnet18
resnet34
resnet50
resnet101
resnet152 or
mobilenetv2
Note that deeper models are more accurate but are slower and use more memory.

## CSV datasets
The `CSVGenerator` provides an easy way to define your own datasets.
It uses two CSV files: one file containing annotations and one file containing a class name to ID mapping.

### Annotations format
The CSV file with annotations should contain one annotation per line.
Images with multiple bounding boxes should use one row per bounding box.
Note that indexing for pixel values starts at 0.
The expected format of each line is:
```
class_name, x1, y1, w, h, img_path
```

Some images may not contain any labeled objects.
To add these images to the dataset as negative examples,
add an annotation where `class_name`, `x1`, `y1`, `w`, `h` are all empty:
```
, , , , , img_path
```

A full example:
```
Bacterial Spot Rot,1763,779,657,849,Bacterial Spot. (100).jpg
Bacterial Spot Rot,1913,442,1087,1000,Bacterial Spot. (101).jpg
Bacterial Spot Rot,223,1093,2062,1845,Bacterial Spot. (101).jpg
```

This defines a dataset with 3 images.
`Bacterial Spot. (100).jpg` contains one instance of Bacterial Spot Rot.
`Bacterial Spot. (101).jpg` contains two instances of Bacterial Spot Rot.

### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```

Indexing for classes starts at 0.
Do not include a background class as it is implicit.

For example:
```
cow,0
cat,1
bird,2
```

## Acknowledgements

- Significant amounts of code are borrowed from the [keras retinanet implementation](https://github.com/fizyr/keras-retinanet)
- The NMS module used is from the [pytorch faster-rcnn implementation](https://github.com/ruotianluo/pytorch-faster-rcnn)
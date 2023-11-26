import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.ops import nms
from torchsummary import summary
import torchvision.models as models
from retinanet.utils import BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet.anchors_resnet import Anchors as AnchorsMini
from retinanet import losses


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(
                in_channels=inp,
                out_channels=inp * expand_ratio,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # depthwise convolution via groups
            nn.Conv2d(
                in_channels=inp * expand_ratio,
                out_channels=inp * expand_ratio,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=inp * expand_ratio,
                bias=False,
            ),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pointwise linear convolution
            nn.Conv2d(
                in_channels=inp * expand_ratio,
                out_channels=oup,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256, mini=False):
        super(PyramidFeatures, self).__init__()
        self.mini = mini

        # upsample C5 to get P5 from the FPN paper
        if not mini: self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        if not mini: self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        if not mini: self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        if not self.mini: 
            P5_x = self.P5_1(C5)
        else:
            P5_x = C5
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        if not self.mini: 
            P4_x = self.P4_1(C4)
        else:
            P4_x = C4
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        if not self.mini: 
            P3_x = self.P3_1(C3)
        else:
            P3_x = C3
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]

class MobileNetV2Mini(nn.Module):
    def __init__(self, width_mult=1.0, num_classes=3, num_anchors=9):
        super(MobileNetV2Mini, self).__init__()

        self.input_channel = int(32 * width_mult)
        self.width_mult = width_mult

        # First layer
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, self.input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.input_channel),
            nn.ReLU6(inplace=True),
        )

        # Inverted residual blocks (each n layers)
        self.inverted_residual_setting = [
            {"expansion_factor": 1, "width_factor": 16, "n": 1, "stride": 1},
            {"expansion_factor": 6, "width_factor": 24, "n": 2, "stride": 2},
            {"expansion_factor": 6, "width_factor": 32, "n": 3, "stride": 2},
            {"expansion_factor": 6, "width_factor": 64, "n": 4, "stride": 2}, 
            {"expansion_factor": 6, "width_factor": 96, "n": 3, "stride": 1},
            {"expansion_factor": 6, "width_factor": 160, "n": 3, "stride": 2},
            # {"expansion_factor": 6, "width_factor": 240, "n": 2, "stride": 2}, # removed extra downsampling layer
            # {"expansion_factor": 6, "width_factor": 320, "n": 1, "stride": 1}, # remove last residual block
        ]
        self.inverted_residual_blocks = nn.ModuleList(
            [
                self._make_inverted_residual_block(**setting)
                for setting in self.inverted_residual_setting
            ]
        )

        self.lateral_setting = [
            setting
            for setting in self.inverted_residual_setting
            if setting["stride"] > 1
        ]
        self.lateral_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    int(setting["width_factor"] * self.width_mult),
                    256,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for setting in self.lateral_setting
            ]
        )

        fpn_sizes = [256, 256, 256]
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], mini=True)

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = AnchorsMini()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def _make_inverted_residual_block(self, expansion_factor, width_factor, n, stride):
        inverted_residual_block = []
        output_channel = int(width_factor * self.width_mult)
        for i in range(n):
            # except the first layer, all layers have stride 1
            if i != 0:
                stride = 1
            inverted_residual_block.append(
                InvertedResidual(
                    self.input_channel, output_channel, stride, expansion_factor
                )
            )
            self.input_channel = output_channel

        return nn.Sequential(*inverted_residual_block)

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        # print('img_batch.shape: ', img_batch.shape)

        # bottom up
        x = self.first_layer(img_batch)
        # print('x.shape: ', x.shape)

        # loop through inverted_residual_blocks (mobile_netV2)
        # save lateral_connections to lateral_tensors
        # track how many lateral connections have been made
        lateral_tensors = []
        n_lateral_connections = 0
        for i, block in enumerate(self.inverted_residual_blocks):
            output = block(x)  # run block of mobile_net_V2
            if self.inverted_residual_setting[i][ "stride" ] > 1 and n_lateral_connections < len(self.lateral_layers):
                lateral_tensors.append(self.lateral_layers[n_lateral_connections](output))
                n_lateral_connections += 1
            x = output

        features = self.fpn(lateral_tensors[-3:])

        # for feature in features:
            # print('feature.shape: ', feature.shape)

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        # print('regression.shape: ', regression.shape)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        # print('classification.shape: ', classification.shape)

        anchors = self.anchors(img_batch)
        # print('anchors.shape: ', anchors.shape)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]


class MobileNetV2_dynamicFPN(nn.Module):
    def __init__(self, width_mult=1.0, num_classes=3, num_anchors=9):
        super(MobileNetV2_dynamicFPN, self).__init__()

        self.input_channel = int(32 * width_mult)
        self.width_mult = width_mult

        # First layer
        self.first_layer = nn.Sequential(
            nn.Conv2d(
                3, self.input_channel, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.input_channel),
            nn.ReLU6(inplace=True),
        )

        # Inverted residual blocks (each n layers)
        self.inverted_residual_setting = [
            {"expansion_factor": 1, "width_factor": 16, "n": 1, "stride": 1},
            {"expansion_factor": 6, "width_factor": 24, "n": 2, "stride": 2},
            {"expansion_factor": 6, "width_factor": 32, "n": 3, "stride": 2},
            {"expansion_factor": 6, "width_factor": 64, "n": 4, "stride": 2},
            {"expansion_factor": 6, "width_factor": 96, "n": 3, "stride": 1},
            {"expansion_factor": 6, "width_factor": 160, "n": 3, "stride": 2},
            {"expansion_factor": 6, "width_factor": 240, "n": 2, "stride": 2}, # addded extra downsampling layer
            # {"expansion_factor": 6, "width_factor": 320, "n": 1, "stride": 1}, # remove last residual block
        ]
        self.inverted_residual_blocks = nn.ModuleList(
            [
                self._make_inverted_residual_block(**setting)
                for setting in self.inverted_residual_setting
            ]
        )

        # reduce feature maps to one pixel
        # allows to upsample semantic information of every part of the image
        self.average_pool = nn.AdaptiveAvgPool2d(1)

        # Top layer
        # input channels = last width factor
        self.top_layer = nn.Conv2d(
            int(self.inverted_residual_setting[-1]["width_factor"] * self.width_mult),
            256,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Lateral layers
        # exclude last setting as this lateral connection is the the top layer
        # build layer only if resulution has decreases (stride > 1)
        self.lateral_setting = [
            setting
            for setting in self.inverted_residual_setting
            if setting["stride"] > 1
        ]
        self.lateral_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    int(setting["width_factor"] * self.width_mult),
                    256,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for setting in self.lateral_setting
            ]
        )

        # Smooth layers
        # n = lateral layers + 1 for top layer

        self.smooth_layer = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Number of classes for prediction
        self.num_classes = num_classes

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(
            -math.log((1.0 - prior) / prior)
        )

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self._initialize_weights()

    def _make_inverted_residual_block(self, expansion_factor, width_factor, n, stride):
        inverted_residual_block = []
        output_channel = int(width_factor * self.width_mult)
        for i in range(n):
            # except the first layer, all layers have stride 1
            if i != 0:
                stride = 1
            inverted_residual_block.append(
                InvertedResidual(
                    self.input_channel, output_channel, stride, expansion_factor
                )
            )
            self.input_channel = output_channel

        return nn.Sequential(*inverted_residual_block)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) + y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        # print('img_batch.shape: ', img_batch.shape)

        # bottom up
        x = self.first_layer(img_batch)
        # print('x.shape: ', x.shape)

        # loop through inverted_residual_blocks (mobile_netV2)
        # save lateral_connections to lateral_tensors
        # track how many lateral connections have been made
        lateral_tensors = []
        n_lateral_connections = 0
        for i, block in enumerate(self.inverted_residual_blocks):
            output = block(x)  # run block of mobile_net_V2
            if self.inverted_residual_setting[i][ "stride" ] > 1 and n_lateral_connections < len(self.lateral_layers):
                lateral_tensors.append( self.lateral_layers[n_lateral_connections](output))
                n_lateral_connections += 1
            x = output

        # reverse lateral tensor order for top down
        lateral_tensors.reverse()
        # connect m_layer with previous m_layer and lateral layers recursively
        m_layers = [lateral_tensors[0]]

        for lateral_tensor in lateral_tensors[1:]:
            m_layers.append(self._upsample_add(m_layers[-1], lateral_tensor))

        # smooth all m_layers
        features = [
            self.smooth_layer(m_layer)
            for m_layer in m_layers
        ]

        # feature_sum = 0
        # for feature in features:
        #     print('feature.shape: ', feature.shape)
        #     feature_sum += feature.shape[2] * feature.shape[3]

        # print('no. of features: ', feature_sum)
        # print('no. of anchors: ', feature_sum*9)

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        # print('regression.shape: ', regression.shape)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        # print('classification.shape: ', classification.shape)

        anchors = self.anchors(img_batch)
        # print('anchors.shape: ', anchors.shape)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = scores > 0.05
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor(
                    [i] * anchors_nms_idx.shape[0]
                )
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat(
                    (finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue)
                )
                finalAnchorBoxesCoordinates = torch.cat(
                    (finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx])
                )

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)
        out = out.contiguous().view(out.shape[0], -1, 4)
        return out

class ClassificationModel(nn.Module):
    def __init__(
        self,
        num_features_in,
        num_anchors=9,
        num_classes=80,
        prior=0.01,
        feature_size=256,
    ):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(
            feature_size, num_anchors * num_classes, kernel_size=3, padding=1
        )
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        out3 = out2.contiguous().view(x.shape[0], -1, self.num_classes)

        return out3


def mobilenetv2FPN(num_classes, pretrained=False, **kwargs):
    """Constructs a MobileNetV2 backbone FPN model."""
    model = MobileNetV2_dynamicFPN(num_classes=num_classes, **kwargs)
    # model.to(torch.device("cuda"))

    if pretrained:
        loaded_state_dict = torch.load("./pretrained_weights/mobilenetv2.pth")
        model.load_state_dict(loaded_state_dict, strict=False)
    return model

def mobilenetminiFPN(num_classes, pretrained=False, **kwargs):
    """Constructs a MobileNetV2 backbone (mini version) FPN model."""
    model = MobileNetV2Mini(num_classes=num_classes, **kwargs)
    # model.to(torch.device("cuda"))

    if pretrained:
        loaded_state_dict = torch.load("./pretrained_weights/mobilenetv2.pth")
        model.load_state_dict(loaded_state_dict, strict=False)
    return model

from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck

from torch import nn as nn



class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, jigsaw_classes=1000, classes=100, domains=3, args=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.l2norm = Normalize(2)


        self.embedding = nn.Sequential(nn.Linear(512 * block.expansion, 128))

        self.embedding = nn.Sequential(nn.Linear(512 * block.expansion, 128))

        self.classifier = nn.Linear(512 * block.expansion, classes)

        self.projection_head = nn.Sequential(
                nn.Linear(512 * block.expansion, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
            # elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)


        return self.classifier(x), self.l2norm(self.projection_head(x)), self.l2norm(x)  #baseline_v2



def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        msg = model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
        print ("pretrain model loaded")
        print ("missing", set(msg.missing_keys))
        # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        # state_dict = torch.load("../pretrained/resnet18-5c106cde.pth")
        # del state_dict["classifier.fc8.weight"]
        # del state_dict["classifier.fc8.bias"]
        # model.load_state_dict(state_dict, strict=False)
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model
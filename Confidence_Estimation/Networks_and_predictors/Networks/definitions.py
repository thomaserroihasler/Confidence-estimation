import math as mt
import torch as tr
import warnings
import torch.nn as nn
import torch.nn.functional as F

class TemperatureScalingBlock(nn.Module):
    """
    TemperatureScalingBlock: A module for temperature scaling, used to calibrate model predictions.
    """
    def __init__(self, Temperature=1.0):
        super(TemperatureScalingBlock, self).__init__()
        self.Temperature = nn.Parameter(tr.ones(1) * Temperature, requires_grad=True)

    def forward(self, input):
        # Divide each logit by the temperature parameter
        output = input / self.Temperature
        return output


# SOFTMAX BLOCKS


class SquareDiagonalLinear(nn.Module):
    """
    SquareDiagonalLinear: A linear layer with a diagonal weight matrix, implemented using matrix multiplication.
    """
    def __init__(self, in_features):
        super(SquareDiagonalLinear, self).__init__()
        self.in_features = in_features
        self.weight = nn.Parameter((tr.randn(1)),requires_grad=True)
        self.bias = nn.Parameter(tr.zeros(in_features))
    def forward(self, input):
        # Compute the diagonal matrix from the weight vector
        # Compute the output using the diagonal weight and bias
        output = tr.matmul(input,tr.eye(self.in_features)*self.weight)
        output += self.bias
        return output

class SquareHeavysideLinear(nn.Module):
    """
    SquareHeavysideLinear: A linear layer with a Heaviside step function activation.
    """
    def __init__(self, in_features):
        super(SquareHeavysideLinear, self).__init__()
        self.in_features = in_features
        self.bias = nn.Parameter(tr.zeros(in_features))

    def forward(self, input):
        output = tr.heaviside(input - 0.5,tr.zeros_like(input))
        return output

class RectangularOneLinear(nn.Module):
    """
    RectangularOneLinear: A linear layer with custom weight initialization and matrix operations.
    """
    def __init__(self, in_features,out_features):
        super(RectangularOneLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        temp = 2*(tr.rand(in_features,out_features))
        temp[:,0] = -1*temp[:,0]
        self.weight = nn.Parameter(temp,requires_grad=True)
        self.bias = nn.Parameter(tr.randn(self.out_features))

    def forward(self, input):
        output = tr.matmul(input,tr.ones(self.in_features,self.out_features)*self.weight)
        output += self.bias
        return output

class RoundAndCount(nn.Module):
    """
    RoundAndCount: A module that rounds the input tensor and counts the number of zeros and ones.
    """
    def __init__(self):
        super(RoundAndCount, self).__init__()

    def forward(self, t):
        rounded = t.round()
        num_zeros = tr.count_nonzero(rounded == 0, dim=-1)
        num_ones = tr.count_nonzero(rounded == 1, dim=-1)
        return (num_zeros < num_ones).long()

#### Neural network approximations ####

### Convention that for classification the final layer is without the sofmax and the maximum

# Simpler approximating network

class GradientSelectiveSigmoidRoundAndCount(nn.Module):
    """
    GradientSelectiveSigmoidRoundAndCount: A neural network model that applies sigmoid activation selectively.
    """
    def __init__(self, height_and_width):
        super(GradientSelectiveSigmoidRoundAndCount, self).__init__()
        self.height_and_width = height_and_width
        # Initialize the first layer as SquareDiagonalLinear
        self.layer1 = SquareDiagonalLinear(height_and_width ** 2)
        # Set bias for the first layer
        self.layer1.bias = nn.Parameter(tr.ones(height_and_width**2) * (-1)*0.5, requires_grad=False)
        # Define the second linear layer
        self.layer2 = nn.Linear(height_and_width * height_and_width, 2, bias=True)

    def forward(self, x):
        # Forward pass through the network with sigmoid activation
        x = self.layer1(x)
        x = nn.functional.sigmoid(x)
        x = self.layer2(x)
        return x

class Identity(nn.Module):
    """
    Identity: A simple identity module that returns its input unchanged.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        # Returns the input as output
        return x

class SigmoidRoundAndCount(nn.Module):
    """
    SigmoidRoundAndCount: A neural network using sigmoid activations.
    """
    def __init__(self, Height_and_width):
        super(SigmoidRoundAndCount, self).__init__()
        self.Height_and_width = Height_and_width
        # Initialize two fully connected layers
        self.fc1 = nn.Linear(Height_and_width*Height_and_width, Height_and_width*Height_and_width)
        self.fc2 = nn.Linear(Height_and_width*Height_and_width, 2)

    def forward(self, x):
        # Flatten the input and apply sigmoid activation
        x = x.view(-1, self.Height_and_width*self.Height_and_width)
        x = tr.sigmoid(self.fc1(x))
        x = tr.sigmoid(self.fc2(x))
        return x


class FullyConnectedNet(nn.Module):
    """
    FullyConnectedNet: A fully connected neural network with customizable layers.
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FullyConnectedNet, self).__init__()
        # Create a sequence of layers
        self.layers = nn.ModuleList()
        # Add layers to the network
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        # Forward pass through each layer with ReLU activation
        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))
        x = self.layers[-1](x)
        return x


class Simple_linear_sigmoid(nn.Module):
    """
    Simple_linear_sigmoid: A linear model with a sigmoid activation function.
    """
    def __init__(self, input_size, output_size):
        super(Simple_linear_sigmoid, self).__init__()
        # Initialize a linear layer followed by sigmoid activation
        self.linear = nn.Linear(input_size, output_size)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # Apply linear transformation followed by sigmoid
        x = self.linear(inputs)
        x = self.Sigmoid(x)
        return x


class BatchedNet(nn.Module):
    """
    BatchedNet: A neural network with one hidden layer using ReLU activation.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(BatchedNet, self).__init__()
        # Initialize layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        # Forward pass with ReLU activation
        x = self.fc1(inputs)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class SimpleCNN(nn.Module):
    """
    SimpleCNN: A convolutional neural network with a sequence of convolutional, ReLU, BatchNorm, and MaxPool layers,
    followed by fully connected layers. It's designed for image classification tasks.
    """
    def __init__(self, input_size = (3,128,128), num_classes=7):
        super(SimpleCNN, self).__init__()

        # Assuming input_size is a tuple in the form (channels, height, width)
        self.input_channels, self.input_height, self.input_width = input_size

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )

        # Calculating the size of the output of the conv layer
        # Assuming stride of 2 for each MaxPool2d layer, and padding does not affect output size due to being same as kernel_size-1.
        conv_output_size = self.calculate_conv_output_size()

        self.fc_layer = nn.Sequential(
            nn.Linear(conv_output_size, 512),  # input dimensions of Linear layer depend on output of conv layers
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def calculate_conv_output_size(self):
        # Calculate the size of the feature map after the conv layers
        # We are assuming that the kernel_size is 3 and stride is 2 for all the MaxPool2d layers for simplification.
        size = (self.input_height // 2 // 2 // 2) * (self.input_width // 2 // 2 // 2) * 128  # division by 2 is due to max pooling
        return size

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.fc_layer(x)
        return x

# VGG & all necessary blocks

warnings.filterwarnings(
    "ignore", message="Setting attributes on ParameterList is not supported."
)

cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [ 64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M",],
}

class VGG(nn.Module):
    """
    VGG: Implementation of the VGG neural network architecture.
    """
    def __init__(self, vgg_name, num_ch=3, num_classes=10, input_shape=(3, 32, 32), bias=True, batch_norm=True,
                 pooling="max", pooling_size=2, param_list=False, width_factor=1, stride=2):
        super(VGG, self).__init__()
        if pooling == True:
            pooling = "max"
        self.features = self._make_layers(cfg[vgg_name], ch=num_ch, bias=bias, bn=batch_norm,
                                          pooling=pooling, ps=pooling_size, param_list=param_list,
                                          width_factor=width_factor, stride=stride)

        # Initialize classifier with dummy input to determine the required number of features
        dummy_input = tr.randn((1,) + input_shape)  # Prepend batch size of 1
        out = self.features(dummy_input)
        out_flat = out.view(out.size(0), -1)
        num_features = out_flat.size(1)

        self.classifier = nn.Linear(num_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        out = self.features(x)
        out_flat = out.view(out.size(0), -1)
        out = self.classifier(out_flat)
        return out

    def _make_layers(self, cfg, ch, bias, bn, pooling, ps, param_list, width_factor, stride):
        layers = []
        in_channels = ch
        if ch == 1:
            layers.append(nn.ZeroPad2d(2))
        if param_list:
            convLayer = Conv2dList
        else:
            convLayer = nn.Conv2d
        for x in cfg:
            if x == "M":
                if pooling == "max":
                    layers += [
                        nn.MaxPool2d(
                            kernel_size=ps, stride=stride, padding=ps // 2 + ps % 2 - 1
                        )
                    ]
                elif pooling == "avg":
                    layers += [
                        nn.AvgPool2d(
                            kernel_size=ps, stride=stride, padding=ps // 2 + ps % 2 - 1
                        )
                    ]
                else:
                    layers += [SubSampling(kernel_size=ps, stride=stride)]
            else:
                x = int(x * width_factor)
                if bn:
                    layers += [
                        convLayer(in_channels, x, kernel_size=3, padding=1, bias=bias),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True),
                    ]
                else:
                    layers += [
                        convLayer(in_channels, x, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                    ]
                in_channels = x
        return nn.Sequential(*layers)

class Conv2dList(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        padding_mode="constant",
    ):
        super().__init__()

        weight = tr.empty(out_channels, in_channels, kernel_size, kernel_size)

        nn.init.kaiming_uniform_(weight, a=mt.sqrt(5))

        if bias is not None:
            bias = nn.Parameter(
                tr.empty(
                    out_channels,
                )
            )
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / mt.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

        n = max(1, 128 * 256 // (in_channels * kernel_size * kernel_size))
        weight = nn.ParameterList(
            [nn.Parameter(weight[j : j + n]) for j in range(0, len(weight), n)]
        )

        setattr(self, "weight", weight)
        setattr(self, "bias", bias)

        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride

    def forward(self, x):

        weight = self.weight
        if isinstance(weight, nn.ParameterList):
            weight = tr.cat(list(self.weight))

        return F.conv2d(x, weight, self.bias, self.stride, self.padding)


class SubSampling(nn.Module):
    """
    SubSampling: A custom subsampling layer for downscaling feature maps.
    """
    def __init__(self, kernel_size, stride=None):
        super(SubSampling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size

    def forward(self, input: tr.Tensor) -> tr.Tensor:
        return input.unfold(2, self.kernel_size, self.stride).unfold(
            3, self.kernel_size, self.stride
        )[..., 0, 0]


class BasicBlock(nn.Module):
    """
    BasicBlock: A basic building block for ResNet architectures.
    """
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck: An advanced building block for ResNet architectures with bottleneck design.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ResNet & all necessary blocks

class ResNet(nn.Module):
    """
    ResNet: A class implementing the ResNet architecture.
    """
    def __init__(self, block, num_blocks, num_classes=10, num_ch=3, input_shape=(1, 3, 32, 32)):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(num_ch, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Use dummy input to determine the number of features after convolutional layers
        dummy_input = tr.randn(input_shape)
        out = self.forward_features(dummy_input)
        out_flat = out.view(out.size(0), -1)
        num_features = out_flat.size(1)

        self.linear = nn.Linear(num_features, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[2:])
        return out

    def forward(self, x):
        out = self.forward_features(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# Factory functions for creating specific ResNet architectures

def ResNet18(num_ch=3, num_classes=10, input_shape=(1, 3, 32, 32)):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, num_ch=num_ch, input_shape=input_shape)

def ResNet34(num_ch=3, num_classes=10, input_shape=(1, 3, 32, 32)):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, num_ch=num_ch, input_shape=input_shape)

def ResNet50(num_ch=3, num_classes=10, input_shape=(1, 3, 32, 32)):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, num_ch=num_ch, input_shape=input_shape)

def ResNet101(num_ch=3, num_classes=10, input_shape=(1, 3, 32, 32)):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, num_ch=num_ch, input_shape=input_shape)

def ResNet152(num_ch=3, num_classes=10, input_shape=(1, 3, 32, 32)):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, num_ch=num_ch, input_shape=input_shape)

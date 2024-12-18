import torch
import torch.nn as nn

# Same padding always used
architecture_configuration = [
    # Tuple: (out_channels (num_filters), kernel_size, stride)
    # List: ["B", num_repeats]
    # "S": branch to one of the three scale predictions
    # "U": upsample (using bilinear interpolation) to larger size
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2), 
    ["B", 4], # Up to here is the full Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S", 
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S"
]

class CNNBlock(nn.Module):
    # bn_act: indicates if block will use batchnorm
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        # using batch norm --> no need for bias
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky_relu(self.batch_norm(self.conv(x)))
        else:
            return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    # first Convolutional input channels is always double the number of output channels in the paper
                    CNNBlock(in_channels, in_channels//2, kernel_size=1),
                    CNNBlock(in_channels//2, in_channels, kernel_size=3, padding=1)
                )
            ]
        
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            # x has same dimensions due to padding --> can just add original x to x after residual block
            # NOTE: the two conv blocks are in a Sequential --> x will be ran through BOTH of them first before original x is added to result
            x = layer(x) + x if self.use_residual else layer(x)
        return x

class ScalePrediction(nn.Module):
    # Make an actual prediction (happens 3 times total)
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.prediction = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
            # there are 3 anchor boxes to predict 3 objects per grid cell
            # for each of the boxes we need to have the number of classes as well as the 5 additional values (prob_of_object, t_x, t_y, t_w, t_h)
            CNNBlock(2*in_channels, 3 * (num_classes + 5), bn_act=False, kernel_size=1)
        )
    def forward(self, x):
        # first dimension of x is batch_size
        # want to simply split the 3*(num_classes+5) into 2 separate dimensions
        # .permute() simply rearranges the dimensions of a tensor according to specified order
        # moves the 3rd dimension to become the last dimension
        return self.prediction(x).reshape(x.shape[0], 3, self.num_classes+5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels 
        self.layers = self._create_conv_layers()

    def forward(self, x):
        # outputs list contains the 3 scaled outputs
        outputs = []
        # skip connections (concatenation)
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                # after every 8-repeated residual block, we have a residual connection that should be concatenated with the upsample layer
                route_connections.append(x)
            
            elif isinstance(layer, nn.Upsample):
                # concatenate along the channels dimension
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs


    def _create_conv_layers(self):
        # use module list instead of regular list so pytorch can keep track of the layers when doing forward and backward propagation
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in architecture_configuration:
            if isinstance(module, tuple):
                num_filters, kernel_size, stride = module
                layers.append(CNNBlock(in_channels, out_channels=num_filters, kernel_size=kernel_size, stride=stride, padding=1 if kernel_size==3 else 0))
                in_channels = num_filters
            
            elif isinstance(module, list):
                num_repeats = module[1]
                # in_channels stays same due to same padding in residual block --> no need to modify it in this case
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))
            
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels//2, kernel_size=1),
                        ScalePrediction(in_channels//2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    # we want to concatenate right after upsampling
                    in_channels *= 3
            
        return layers



from torch import nn

class CustomModel(nn.Module):
    """
    from alexnet
    """
    def __init__(self):
        super(CustomModel, self).__init__()
        num_classes = 3
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(576, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, num_classes),
        )

    def forward(self, input):
        input = self.features(input)
        input = input.view(input.size(0), -1)
        return self.classifier(input)


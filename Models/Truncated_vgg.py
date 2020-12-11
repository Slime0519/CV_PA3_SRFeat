import torch.nn as nn
import torchvision.models as models

from torchsummary import summary
import os

class truncated_vgg(nn.Module):
    def __init__(self,truncate_layer=35):
        super(truncated_vgg, self).__init__()
        self.model = models.vgg.vgg19(pretrained=True)
        self.feature_extraction_model = nn.Sequential(*list(self.model.features)[:truncate_layer]).eval()
        self.truncated_layer = truncate_layer

        for param in self.feature_extraction_model.parameters():
            param.requires_grad = False
        self.MSE_Loss = nn.MSELoss()

    # test용 forward 함수
    def forward(self, x):
        out = self.feature_extraction_model(x)
        return out

    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, std=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

if __name__ == "__main__":

    Test = truncated_vgg().to('cuda:0')

    summary(Test,(3,296,296))
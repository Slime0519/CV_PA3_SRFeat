
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from torchsummary import summary
import os

PATCH_DIR ="Result_image/patchimage"
def savepatch(DIRPATH, batch,layer):

    DIRPATH_LAYER = os.path.join(DIRPATH,"layer{}".format(layer))
    if not os.path.isdir(DIRPATH_LAYER):
        os.mkdir(DIRPATH_LAYER)

    for i,patch in enumerate(batch):
        plt.imshow(patch)
        plt.savefig(os.path.join(DIRPATH_LAYER,"{}th_feature.png".format(i,layer)))

class truncated_vgg(nn.Module):
    def __init__(self,truncate_layer=35):
        super(truncated_vgg, self).__init__()
        self.model = models.vgg.vgg19(pretrained=True)
        self.feature_extraction_model = nn.Sequential(*list(self.model.features)[:truncate_layer]).eval()
        #self.feature_extraction_model = nn.Sequential(*list(self.model.features)).eval()
       # print(list(self.model.features)[:truncate_layer])
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
    """
    def forward(self,input,show = False):
        out = self.feature_extraction_model(input)
      #  target_extracted_fatch = self.feature_extraction_model(target)
       # perception_loss = self.MSE_Loss(input_extracted_fatch,target_extracted_fatch)
        
        if show:
            patch_image_input = np.array(input_extracted_fatch.cpu().detach()).squeeze()
           # print(patch_image_input.shape)
          #  savepatch(os.path.join(PATCH_DIR,"input"),patch_image_input,self.truncated_layer)
          #  plt.imshow(patch_image_input[:,:,0])
           # plt.show()

            patch_image_target = np.array(target_extracted_fatch.cpu().detach()).squeeze()
           # print(patch_image_target.shape)
           # savepatch(os.path.join(PATCH_DIR,"target"), patch_image_target, self.truncated_layer)
           # plt.imshow(patch_image_target[:, :, 0])
          #  plt.show()
        
        return out
    """


if __name__ == "__main__":

    Test = vggloss().to('cuda:0')

    summary(Test,(3,384,384))
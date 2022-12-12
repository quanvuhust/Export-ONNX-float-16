import torch
from torch import nn
# from senet import legacy_seresnext50_32x4d, legacy_seresnext101_32x4d
# from vision_transformer import vit_base_patch16_384
import timm



def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    permute = [2, 1, 0]
    # print(image.shape)
    image = image[:,:,:,permute]
    # print(image.shape)
    image = image.transpose(1, 2).transpose(1, 3).contiguous()
    return image

def bgr_to_bgr(image: torch.Tensor) -> torch.Tensor:
    image = image.transpose(1, 2).transpose(1, 3).contiguous()
    return image

class Net(nn.Module):
    def __init__(self, back_bone):
        super().__init__()
        self.model = timm.create_model(back_bone, num_classes=2, pretrained=False, in_chans=3)

    def forward(self, x):
        logit = self.model(x)
        return logit

class EnsembleNet(nn.Module):
    def __init__(self, image_mode, model_0, model_1, model_2, model_3, model_4):
        super().__init__()
        self.model_0 = model_0
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_3 = model_3
        self.model_4 = model_4
        self.IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406]).cuda()
        self.IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225]).cuda()
        
        if image_mode == "rgb":
            self.transform = bgr_to_rgb
        elif image_mode == "bgr":
            self.transform = bgr_to_bgr

    def forward(self, x):
        x = x/255.0
        x = self.transform(x)
        # print(x.shape)
        
        x = (x - self.IMAGENET_DEFAULT_MEAN[None,:, None, None])/self.IMAGENET_DEFAULT_STD[None,:, None, None]
        # print(x.shape)
        logit_0 = self.model_0(x)
        logit_1 = self.model_1(x)
        logit_2 = self.model_2(x)
        logit_3 = self.model_3(x)
        logit_4 = self.model_4(x)
        probs_0 = nn.Softmax(dim=1)(logit_0)
        probs_1 = nn.Softmax(dim=1)(logit_1)
        probs_2 = nn.Softmax(dim=1)(logit_2)
        probs_3 = nn.Softmax(dim=1)(logit_3)
        probs_4 = nn.Softmax(dim=1)(logit_4)
        probs = (probs_0 + probs_1 + probs_2 + probs_3 + probs_4)/5
        return probs[:,1]

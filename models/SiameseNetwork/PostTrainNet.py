import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import timm
from collections import OrderedDict

def rename_dict(load_dict):
    new_state_dict = OrderedDict()
    for k, v in load_dict.items():
        name = k[13:] #从classification网络读取
        #name = k[27:] # 从embedding读取remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。

    return new_state_dict

class Xception_loaded(nn.Module):
    def __init__(self, load_path):
        super(Xception_loaded, self).__init__()
        self.model = timm.create_model('xception', pretrained=True, num_classes=2)
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(rename_dict(checkpoint['net_state_dict']))


    def forward(self, x):
        out = self.model(x)
        return out


class PostTrainNet_xception(nn.Module):
    def __init__(self, load_path, freeze=True):
        super(PostTrainNet_xception, self).__init__()
        embed_model = timm.create_model('xception', pretrained=True, num_classes=2)

        checkpoint = torch.load(load_path)
        embed_model.load_state_dict(rename_dict(checkpoint['net_state_dict']))

        if freeze:
            for n,p in embed_model.named_parameters():
                if "fc" not in n:
                    p.requires_grad = False

        self.net = embed_model

    def forward(self, x):
        x = self.net(x)
        return x

    def get_embed(self,x):
        out = self.net.forward_features(x)
        out = self.net.forward_head(out, pre_logits=True)
        return out


#rand_input = torch.FloatTensor(16,3,224,224)
#load_path = '/hdd7/yinjie/deepfake_ckpt/Xception_pretrained__deepfake_faces_2022-12-21_13:55:43/Xception_pretrained_best_params.pkl'
#state_dict1 = torch.load(load_path)['net_state_dict']
#state_dict2 = timm.create_model('xception', pretrained=True, num_classes=2)

#net = PostTrainNet_xception(load_path, freeze=True)
#res = net.get_embed(rand_input) #(16,2048)
#print("done")
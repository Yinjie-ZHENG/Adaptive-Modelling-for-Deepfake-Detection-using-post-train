import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import timm

from collections import OrderedDict

def rename_dict(load_dict):
    new_state_dict = OrderedDict()
    for k, v in load_dict.items():
        name = k[27:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。

    return new_state_dict



class FineTuneNet_resnet50(nn.Module):
    def __init__(self, load_path, freeze=True):
        super(FineTuneNet_resnet50, self).__init__()
        embed_model = models.__dict__['resnet50'](pretrained=True)
        #self.last_layer = nn.Sequential(*list(embed_model.children())[-1])
        num_ftrs = embed_model.fc.in_features

        modules = list(embed_model.children())[:-1]
        embed_model = nn.Sequential(*modules)
        checkpoint = torch.load(load_path)
        embed_model.load_state_dict(rename_dict(checkpoint['net_state_dict']))
        if freeze:
            for p in embed_model.parameters():
                p.requires_grad = False
        self.convnet = embed_model
        self.fc = nn.Linear(num_ftrs, 2)


    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output



class FineTuneNet_xception(nn.Module):
    def __init__(self, load_path, freeze=True):
        super(FineTuneNet_xception, self).__init__()
        embed_model = timm.create_model('xception', pretrained=True, num_classes=2)
        num_ftrs = embed_model.fc.in_features
        checkpoint = torch.load(load_path)
        embed_model.load_state_dict(rename_dict(checkpoint['net_state_dict']))
        if freeze:
            for n,p in embed_model.named_parameters():
                #print(n)
                p.requires_grad = False
        del embed_model.fc
        self.convnet = embed_model
        self.fc = nn.Linear(num_ftrs, 2)


    def forward(self, x):
        x = self.convnet.forward_features(x)
        x = self.convnet.forward_head(x, pre_logits=True)

        output = self.fc(x)
        return output

#rand_input = torch.FloatTensor(16,3,224,224)
#load_path = '/hdd7/yinjie/deepfake_ckpt/SiameseNet_Xception__miniFFDataset_siamese_2022-12-13_21:38:58/SiameseNet_Xception_best_params.pkl'  # None#'/home/yinjie/FYP_/torch/ckpts/resnet50_pretrained_none_tiny_imagenet_2022-11-11_19:02:29/val_acc_25.799999999999997_epoch_28.pkl'
#state_dict1 = torch.load(load_path)['net_state_dict']
#state_dict2 = timm.create_model('xception', pretrained=True, num_classes=2)

#net = FineTuneNet_xception(load_path)
#res = net(rand_input)
#print("done")
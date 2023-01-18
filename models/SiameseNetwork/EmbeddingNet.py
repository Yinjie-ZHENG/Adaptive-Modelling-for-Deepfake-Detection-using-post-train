import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import timm



class EmbeddingNet_base(nn.Module):
    def __init__(self):
        super(EmbeddingNet_base, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class EmbeddingNet_resnet50(nn.Module):
    def __init__(self):
        super(EmbeddingNet_resnet50, self).__init__()
        embed_model = models.__dict__['resnet50'](pretrained=True)
        modules = list(embed_model.children())[:-1]
        embed_model = nn.Sequential(*modules)
        #for p in embed_model.parameters():
        #    p.requires_grad = False
        self.convnet = embed_model

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNet_Xception(nn.Module):
    def __init__(self):
        super(EmbeddingNet_Xception, self).__init__()
        self.model = timm.create_model('xception', pretrained=True, num_classes=2)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.model.forward_head(x, pre_logits=True)
        return x

class EmbeddingNet_XceptionV2(nn.Module):
    def __init__(self):
        super(EmbeddingNet_XceptionV2, self).__init__()
        xceptionnet = timm.create_model('xception', pretrained=True, num_classes=2)
        features = list(xceptionnet.children())
        self.fea_extractor = nn.Sequential(*features[:-1])
        self.classifier = nn.Sequential(*features[-1:])

    def forward(self, x):
        x = self.fea_extractor(x)
        x = self.classifier(x)
        return x


#rand_input = torch.FloatTensor(16,3,224,224)
#Embed_resnet50 = EmbeddingNet_resnet50()
#Embedd_xception = EmbeddingNet_XceptionV2()
#print("Torchvision Version: ", torchvision.__version__)
#out = Embedd_xception(rand_input)
#rand_input2 = torch.FloatTensor(16,1,28,28)
#Embed_ = EmbeddingNet_base()
#out2 = Embed_(rand_input2)
#print("done")
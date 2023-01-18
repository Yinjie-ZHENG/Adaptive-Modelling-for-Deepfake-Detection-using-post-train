import torch
import torch.nn as nn
import torchvision.models as models
#from EmbeddingNet import EmbeddingNet_resnet50,EmbeddingNet_Xception


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)



#rand_input1 = torch.FloatTensor(16,3,224,224)
#rand_input2 = torch.FloatTensor(16,3,224,224)
#Embed_resnet50 = EmbeddingNet_resnet50()
#Embedd_xception = EmbeddingNet_Xception()
#SiameseNet_resnet50 = SiameseNet(Embedd_xception)
#out = SiameseNet_resnet50(rand_input1,rand_input2)
#print("done")
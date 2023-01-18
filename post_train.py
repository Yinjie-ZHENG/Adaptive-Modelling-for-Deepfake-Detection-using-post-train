
import os
import warnings
import functools
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

import torch
import torchvision
import torch.optim as optim
from torchsummary import summary
from torch import sigmoid,softmax
from torch.utils.data import  DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import shutil
from V4args import *
from utils.arg_utils import *
from utils.data_utils import *
from utils.algorithm_utils import *
from dataloder import load_dataset
from metrics import Accuracy_score, AverageMeter, accuracy, accuracy2
from torch.utils.data import Subset
import copy
from scipy import spatial


class CosineSimilarity(nn.Module):
    __constants__ = ['dim', 'eps']

    def __init__(self, dim=1, eps=1e-8):
        super(CosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        nor_x1 = F.normalize(x1, dim=1)
        nor_x2 = F.normalize(x2, dim=1)
        return F.cosine_similarity(nor_x1, nor_x2, self.dim, self.eps)

cos_similarity = CosineSimilarity(dim=1, eps=1e-6)
logger = get_logger(data_config.work_dir + '/exp.log')
shutil.copyfile("./V4args.py", data_config.work_dir+"/configs.py")


def get_train_loaders_by_class(train_dataset, batch_size):
    indices_list = [[] for _ in range(data_config.num_class)]
    for i in range(len(train_dataset)):
        label = int(train_dataset[i][1])
        indices_list[label].append(i)
    dataset_list = [Subset(train_dataset, indices) for indices in indices_list]
    train_loader_list = [
        torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,#True
            pin_memory=True,
            num_workers=0,
        ) for dataset in dataset_list
    ]
    return train_loader_list



def post_train(model, images, train_loader, train_loaders_by_class, args):
    #保证插入的images是单张
    assert len(images) == 1, "Post training algorithm only accepts test input of batch size 1"

    logger = logging.getLogger("eval")
    loss_func = nn.CrossEntropyLoss()

    device = torch.device('cuda')
    #model是可以更改权重的， fix_model是不可以更改权重的
    model = copy.deepcopy(model)
    # model.train()
    fix_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(lr=args.pt_lr,
                                params=model.parameters(),
                                momentum=0.9,
                                nesterov=True)
    images = images.detach()
    #用了with torch.enable_grad，说明要对model更改
    #test警告,此时将fix_model为eval模式，model为
    model.train()
    with torch.enable_grad():
        # 把验证集图片传入，得到输出class
        #试试看让fix_model的参数梯度固定
        for n, param in fix_model.named_parameters():
            param.requires_grad = False
        original_output = fix_model(images)
        original_class = torch.argmax(original_output).reshape(1)
        #这部分需要重新看看，统一original_class和neighbour_class的类型
        neighbour_class = 1-original_class
        assert original_class.type() == neighbour_class.type()

        loss_list = []
        acc_list = []
        original_embed = fix_model.module.get_embed(images)
        #设定post_train多少次iter,每次iter都做这样一件事:寻找相邻数据

        for _ in range(args.pt_iter):
            if args.pt_data == 'num_nearest':
                similarity_lst = []
                num = (data_config.batch_size // 2)
                batch_data, batch_label = next(iter(train_loader))
                batch_data = batch_data.detach().cuda()
                batch_embeds = fix_model.module.get_embed(batch_data)
                for i in range(batch_embeds.size(0)):
                    #注意dist不是相似度，请搞清楚，目前还是用的相似度但显示的是dist
                    #similarity_lst中(similarity[1], data[1,3,224,224], label[0])
                    similarity_lst.append([torch.cosine_similarity(original_embed, batch_embeds[i].view(1, -1)), batch_data[i].unsqueeze(0), batch_label[i]])
                simi = [x[0] for x in similarity_lst]
                simi = torch.tensor(simi)
                # 获取按照 simi 排序后的索引
                _, indices = torch.sort(simi, descending=True)
                # 根据索引对 similarity_lst 进行排序
                sorted_similarity_lst = [similarity_lst[i] for i in indices]

                data = torch.cat([x[1] for x in sorted_similarity_lst[:num]], dim=0).to(device)
                label = torch.stack([x[2] for x in sorted_similarity_lst[:num]], dim=0).to(device)

            elif args.pt_data == 'num_nearest_cosine':
                similarity_lst = []
                num = (data_config.batch_size // 2)
                batch_data, batch_label = next(iter(train_loader))
                batch_data = batch_data.detach().cuda()
                batch_embeds = fix_model.module.get_embed(batch_data)
                for i in range(batch_embeds.size(0)):
                    #注意dist不是相似度，请搞清楚，目前还是用的相似度但显示的是dist
                    #similarity_lst中(similarity[1], data[1,3,224,224], label[0])
                    original_embed = F.normalize(original_embed, dim=1)  # (bs, dim)  --->  (bs, dim)
                    batch_embed_i = F.normalize(batch_embeds[i].view(1, -1), dim=1)

                    similarity_lst.append([torch.cosine_similarity(original_embed, batch_embed_i), batch_data[i].unsqueeze(0), batch_label[i]])
                simi = [x[0] for x in similarity_lst]
                simi = torch.tensor(simi)
                # 获取按照 simi 排序后的索引
                _, indices = torch.sort(simi, descending=True)
                # 根据索引对 similarity_lst 进行排序
                sorted_similarity_lst = [similarity_lst[i] for i in indices]

                data = torch.cat([x[1] for x in sorted_similarity_lst[:num]], dim=0).to(device)
                label = torch.stack([x[2] for x in sorted_similarity_lst[:num]], dim=0).to(device)




            elif args.pt_data == 'num_nearest_searchsim':
                similarity_lst = []
                num = (data_config.batch_size // 2)
                batch_data, batch_label = next(iter(train_loader))
                batch_data = batch_data.detach().cuda()
                batch_embeds = fix_model.module.get_embed(batch_data)
                for i in range(batch_embeds.size(0)):
                    #注意dist不是相似度，请搞清楚，目前还是用的相似度但显示的是dist
                    #similarity_lst中(similarity[1], data[1,3,224,224], label[0])
                    similarity_lst.append([torch.cosine_similarity(original_embed, batch_embeds[i].view(1, -1)), batch_data[i].unsqueeze(0), batch_label[i]])
                #select according to similarity
                sorted_similarity_lst = [x for x in similarity_lst if x[0] > data_config.pt_searchsim_thread]
                if len(sorted_similarity_lst) == 0:
                    break
                data = torch.cat([x[1] for x in sorted_similarity_lst], dim=0).to(device)
                label = torch.stack([x[2] for x in sorted_similarity_lst], dim=0).to(device)
                #print("doing")




            elif args.pt_data == 'num_nearest_balanced':
                similarity_lst = []
                num = (data_config.batch_size // 2)
                original_data, original_label = next(iter(train_loaders_by_class[original_class]))
                neighbour_data, neighbour_label = next(iter(train_loaders_by_class[neighbour_class]))
                batch_data = torch.vstack([original_data, neighbour_data]).to(device)
                batch_label = torch.hstack([original_label, neighbour_label]).to(device)
                batch_data = batch_data.detach().cuda()

                batch_embeds = fix_model.module.get_embed(batch_data)
                for i in range(batch_embeds.size(0)):
                    #注意dist不是相似度，请搞清楚，目前还是用的相似度但显示的是dist
                    #similarity_lst中(similarity[1], data[1,3,224,224], label[0])
                    similarity_lst.append([torch.cosine_similarity(original_embed, batch_embeds[i].view(1, -1)), batch_data[i].unsqueeze(0), batch_label[i]])
                simi = [x[0] for x in similarity_lst]
                simi = torch.tensor(simi)
                # 获取按照 simi 排序后的索引
                _, indices = torch.sort(simi, descending=True)
                # 根据索引对 similarity_lst 进行排序
                sorted_similarity_lst = [similarity_lst[i] for i in indices]

                data = torch.cat([x[1] for x in sorted_similarity_lst[:num]], dim=0).to(device)
                label = torch.stack([x[2] for x in sorted_similarity_lst[:num]], dim=0).to(device)


            elif args.pt_data == 'ori_neigh':
                original_data, original_label = next(iter(train_loaders_by_class[original_class]))
                neighbour_data, neighbour_label = next(iter(train_loaders_by_class[neighbour_class]))
                data = torch.vstack([original_data, neighbour_data]).to(device)
                label = torch.hstack([original_label, neighbour_label]).to(device)

            elif args.pt_data == 'train':
                original_data, original_label = next(iter(train_loader))
                neighbour_data, neighbour_label = next(iter(train_loader))
                data = torch.vstack([original_data, neighbour_data]).to(device)
                label = torch.hstack([original_label, neighbour_label]).to(device)
            else:
                raise NotImplementedError

            #data = torch.vstack([original_data, neighbour_data]).to(device)
            #label = torch.hstack([original_label, neighbour_label]).to(device)

            if args.pt_method == 'adv':
                # generate fgsm adv examples
                delta = (torch.rand_like(data) * 2 - 1) * epsilon  # uniform rand from [-eps, eps]
                noise_input = data + delta
                noise_input.requires_grad = True
                noise_output = model(noise_input)
                loss = loss_func(noise_output, label)  # loss to be maximized
                input_grad = torch.autograd.grad(loss, noise_input)[0]
                delta = delta + alpha * torch.sign(input_grad)
                delta.clamp_(-epsilon, epsilon)
                adv_input = data + delta
            elif args.pt_method == 'dir_adv':
                # use fixed direction attack
                if args.adv_dir == 'pos':
                    adv_input = data + 1 * neighbour_delta
                elif args.adv_dir == 'neg':
                    adv_input = data + -1 * neighbour_delta
                elif args.adv_dir == 'both':
                    directed_delta = torch.vstack([torch.ones_like(original_data).to(device) * neighbour_delta,
                                                    torch.ones_like(neighbour_data).to(device) * -1 * neighbour_delta])
                    adv_input = data + directed_delta
            elif args.pt_method == 'normal':
                adv_input = data
            else:
                raise NotImplementedError

            for n, param in model.named_parameters():
                if data_config.is_freeze:
                    if "fc" not in n:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                        #print(param)

            adv_output = model(adv_input.detach())

            loss = loss_func(adv_output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            defense_acc = accuracy2(adv_output, label)#cal_accuracy(adv_output, label)
            loss_list.append(loss)
            acc_list.append(defense_acc)
    return model, original_class, neighbour_class, loss_list, acc_list


logger.info("***********- ***********- READ DATA and processing-*************")
train_dataset, val_dataset = load_dataset(data_config)
train_loader = DataLoader(train_dataset, batch_size=data_config.batch_size, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
train_loaders_by_class = get_train_loaders_by_class(train_dataset, data_config.batch_size)

#print("done")

logger.info("***********- loading model -*************")
if(len(data_config.gpus)==0):#cpu
    model = data_config.model_arch
    if data_config.load_from_pth is not None:
        checkpoint = torch.load(data_config.load_from_pth)
        model.load_state_dict(checkpoint)
    elif data_config.load_from_pkl is not None:
        model, _, _ = load_checkpoint(model=model, checkpoint_path=data_config.load_from_pkl)
elif(len(data_config.gpus)==1):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(data_config.gpus[0])
    model = data_config.model_arch.cuda()
    if data_config.load_from_pth is not None:
        checkpoint = torch.load(data_config.load_from_pth)
        model.load_state_dict(checkpoint)
    elif data_config.load_from_pkl is not None:
        model, _, _ = load_checkpoint(model=model, checkpoint_path=data_config.load_from_pkl)
else:#multi gpus
    gpus = ','.join(str(i) for i in data_config.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    model = data_config.model_arch.cuda()
    gpus = [i for i in range(len(data_config.gpus))]
    model = torch.nn.DataParallel(model)


model.eval()

epsilon = 8/255#0.3
alpha = 10/255#1e-2
pgd_loss = 0
pgd_acc = 0
pgd_acc_post = 0
normal_loss = 0
normal_acc = 0
normal_loss_post = 0
normal_acc_post = 0
neighbour_acc = 0

pgd_loss_rep = AverageMeter()
pgd_acc_rep = AverageMeter()

pgd_acc_post_rep = AverageMeter()
normal_loss_rep = AverageMeter()
normal_acc_rep = AverageMeter()
normal_loss_post_rep = AverageMeter()
normal_acc_post_rep = AverageMeter()


n = 0


for i, (X, y) in enumerate(test_loader):
    n += y.size(0)
    X, y = X.cuda(), y.cuda()
    logger.info("\n")
    # evaluate base model
    with torch.no_grad():
        output = model(X)
        acc_top1 = accuracy2(softmax(output, dim=-1).data, y) #softmax(output, dim=-1)是predicted
        pgd_acc_rep.update(acc_top1[0].item(), y.size(0))
        logger.info('Batch {}\tbase model acc: {:.4f}'.format(i + 1, pgd_acc_rep.avg))
        logger.info('Batch {}\toutput:{},y:{}'.format(i + 1, softmax(output, dim=-1).data, y))


    # evaluate post model
    with torch.no_grad():
        post_model, original_class, neighbour_class, _, _ = post_train(model, X, train_loader,
                                                                          train_loaders_by_class, data_config)

        # evaluate prediction acc
        #在此时post_model居然是training mode,请改正
        post_model.eval()
        output = post_model(X)
        pgd_acc_post += (output.max(1)[1] == y).sum().item()
        logger.info('Batch {}\tadv acc (post): {:.4f}'.format(i + 1, pgd_acc_post / n))
        acc_top1 = accuracy2(softmax(output, dim=-1).data, y) #softmax(output, dim=-1)是predicted
        pgd_acc_post_rep.update(acc_top1[0].item(), y.size(0))
        logger.info('Batch {}\tbase acc_rep: {:.4f}'.format(i + 1, pgd_acc_post_rep.avg))
        logger.info('Batch {}\tpostoutput:{},y:{}'.format(i + 1, softmax(output, dim=-1).data, y))

print("done")
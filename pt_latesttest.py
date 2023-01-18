
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
from V5args import *
from utils.arg_utils import *
from utils.data_utils import *
from utils.algorithm_utils import *
from dataloder import load_dataset
from metrics import Accuracy_score, AverageMeter, accuracy, accuracy2
from torch.utils.data import Subset
import copy
from scipy import spatial
import torchattacks



logger = get_logger(data_config.work_dir + '/exp.log')
shutil.copyfile("./V5args.py", data_config.work_dir+"/configs.py")


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std
norm_layer = Normalize(mean=data_config.normalized_cfg_mean, std=data_config.normalized_cfg_std)


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



def post_train(model, images, train_loader, train_loaders_by_class, args,sft_output):
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
    #原本model.train是放在这的,但是考虑到还要先攻击，攻击的时候总不能train吧
    #model.train()
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
        if args.pt_setting == "near_boundary":
            if abs(sft_output[0][0] - sft_output[0][1])> (0.2 + 1e-5):
                return model, original_class, neighbour_class, None, None

        logger.info('near_boundary == True')

        loss_list = []
        acc_list = []
        if args.normalized:
            original_embed = fix_model.module[1].get_embed(images)
        else:
            original_embed = fix_model.module.get_embed(images)
        #先攻击，再看看后面咋样
        #设定post_train多少次iter,每次iter都做这样一件事:寻找相邻数据
        if args.pt_method == "deepfool":
            attack = torchattacks.DeepFool(fix_model, steps=50, overshoot=0.02)
            adv_images = attack(images, original_class)
            adv_output = fix_model(adv_images)
            adv_class = torch.argmax(adv_output).reshape(1)
            if original_class == adv_class:
                logger.info('original class == adv class')
                return model, original_class, adv_class, None, None

            if args.normalized:
                adv_embed = fix_model.module[1].get_embed(adv_images)
            else:
                adv_embed = fix_model.module.get_embed(adv_images)

        elif args.pt_method == "pgd":
            attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=True)
            adv_images = attack(images, original_class)
            adv_output = fix_model(adv_images)
            adv_class = torch.argmax(adv_output).reshape(1)
            if original_class == adv_class:
                logger.info('original class == adv class')
                return model, original_class, adv_class, None, None

            if args.normalized:
                adv_embed = fix_model.module[1].get_embed(adv_images)
            else:
                adv_embed = fix_model.module.get_embed(adv_images)


        for _ in range(args.pt_iter):
            #选择train或者balanced_train作为训练数据,batch_data和batch_label是用到的数据和标签
            if args.pt_data == "train":
                original_data, original_label = next(iter(train_loader))
                neighbour_data, neighbour_label = next(iter(train_loader))
                batch_data = torch.vstack([original_data, neighbour_data]).to(device)
                batch_label = torch.hstack([original_label, neighbour_label]).to(device)
                batch_data = batch_data.detach().cuda()
            else:#if args.pt_data == "train"
                assert args.pt_data == "balanced_train"
                original_data, original_label = next(iter(train_loaders_by_class[original_class]))
                neighbour_data, neighbour_label = next(iter(train_loaders_by_class[neighbour_class]))
                batch_data = torch.vstack([original_data, neighbour_data]).to(device)
                batch_label = torch.hstack([original_label, neighbour_label]).to(device)
                batch_data = batch_data.detach().cuda()

            similarity_lst = []
            similarity_lst_ori = []
            similarity_lst_adv = []
            num = (data_config.batch_size // 2)

            if args.normalized:
                batch_embeds = fix_model.module[1].get_embed(batch_data)
            else:
                batch_embeds = fix_model.module.get_embed(batch_data)

            #这一步实际上就是选取search中心pt_method，normal使用原本图像作为search中心；pgd使用pgd attack,有两个searach中心；deepfool使用deepfool作为search中心
            if args.pt_method == "normal":#normal使用原本图像作为search中心
                # 在循环外部已经有original_embed了，不需要重复获取
                #original_embed = fix_model.module.get_embed(images)

                for i in range(batch_embeds.size(0)):
                    #注意dist不是相似度，请搞清楚，目前还是用的相似度但显示的是dist
                    #similarity_lst中(similarity[1], data[1,3,224,224], label[0])
                    if args.measurement == "cosine_normalized":
                        original_embed_i = F.normalize(original_embed, dim=1)  # (bs, dim)  --->  (bs, dim)
                        batch_embed_i = F.normalize(batch_embeds[i].view(1, -1), dim=1)
                    else:
                        original_embed_i = original_embed
                        batch_embed_i = batch_embeds[i].view(1, -1)
                    similarity_lst.append([torch.cosine_similarity(original_embed_i, batch_embed_i), batch_data[i].unsqueeze(0), batch_label[i]])
                if args.search_by == "num_search":
                    simi = [x[0] for x in similarity_lst]
                    simi = torch.tensor(simi)
                    # 获取按照 simi 排序后的索引
                    _, indices = torch.sort(simi, descending=True)
                    # 根据索引对 similarity_lst 进行排序
                    sorted_similarity_lst = [similarity_lst[i] for i in indices]
                    data = torch.cat([x[1] for x in sorted_similarity_lst[:num]], dim=0).to(device)
                    label = torch.stack([x[2] for x in sorted_similarity_lst[:num]], dim=0).to(device)
                elif args.search_by == "dist_search":
                    # select according to similarity
                    sorted_similarity_lst = [x for x in similarity_lst if x[0] > args.pt_searchsim_thread]
                    if len(sorted_similarity_lst) == 0:
                        break
                    data = torch.cat([x[1] for x in sorted_similarity_lst], dim=0).to(device)
                    label = torch.stack([x[2] for x in sorted_similarity_lst], dim=0).to(device)
            elif args.pt_method == "deepfool": # deepfool使用原本图像对抗攻击后的图像作为search中心
                # #在循环外部已经有adv_embed了，不需要重复获取
                #batch_embeds = fix_model.module.get_embed(batch_data)
                for i in range(batch_embeds.size(0)):
                    #注意dist不是相似度，请搞清楚，目前还是用的相似度但显示的是dist
                    #similarity_lst中(similarity[1], data[1,3,224,224], label[0])
                    if args.measurement == "cosine_normalized":
                        adv_embed_i = F.normalize(adv_embed, dim=1)  # (bs, dim)  --->  (bs, dim)
                        batch_embed_i = F.normalize(batch_embeds[i].view(1, -1), dim=1)
                    else:
                        adv_embed_i = adv_embed
                        batch_embed_i = batch_embeds[i].view(1, -1)
                    similarity_lst.append([torch.cosine_similarity(adv_embed_i, batch_embed_i), batch_data[i].unsqueeze(0), batch_label[i]])
                if args.search_by == "num_search":
                    simi = [x[0] for x in similarity_lst]
                    simi = torch.tensor(simi)
                    # 获取按照 simi 排序后的索引
                    _, indices = torch.sort(simi, descending=True)
                    # 根据索引对 similarity_lst 进行排序
                    sorted_similarity_lst = [similarity_lst[i] for i in indices]
                    data = torch.cat([x[1] for x in sorted_similarity_lst[:num]], dim=0).to(device)
                    label = torch.stack([x[2] for x in sorted_similarity_lst[:num]], dim=0).to(device)
                elif args.search_by == "dist_search":
                    # select according to similarity
                    sorted_similarity_lst = [x for x in similarity_lst if x[0] > args.pt_searchsim_thread]
                    if len(sorted_similarity_lst) == 0:
                        break
                    data = torch.cat([x[1] for x in sorted_similarity_lst], dim=0).to(device)
                    label = torch.stack([x[2] for x in sorted_similarity_lst], dim=0).to(device)

            elif args.pt_method == "pgd": # pgd 有两个search中心，一个是original img, 一个是pgd attack过后的img作为search中心
                for i in range(batch_embeds.size(0)):
                    if args.measurement == "cosine_normalized":
                        original_embed_i = F.normalize(original_embed, dim=1)
                        adv_embed_i = F.normalize(adv_embed, dim=1)  # (bs, dim)  --->  (bs, dim)
                        batch_embed_i = F.normalize(batch_embeds[i].view(1, -1), dim=1)
                    else:
                        original_embed_i = original_embed
                        adv_embed_i = adv_embed
                        batch_embed_i = batch_embeds[i].view(1, -1)

                    similarity_lst_ori.append([torch.cosine_similarity(original_embed_i, batch_embed_i), batch_data[i].unsqueeze(0),batch_label[i]])
                    similarity_lst_adv.append([torch.cosine_similarity(adv_embed_i, batch_embed_i), batch_data[i].unsqueeze(0),batch_label[i]])

                if args.search_by == "num_search":
                    #ori
                    simi_ori = [x[0] for x in similarity_lst_ori]
                    simi_ori = torch.tensor(simi_ori)
                    # 获取按照 simi 排序后的索引
                    _, indices_ori = torch.sort(simi_ori, descending=True)
                    # 根据索引对 similarity_lst_ori 进行排序
                    sorted_similarity_lst_ori = [similarity_lst_ori[i] for i in indices_ori]
                    data_ori = torch.cat([x[1] for x in sorted_similarity_lst_ori[:(num//2)]], dim=0).to(device)
                    label_ori = torch.stack([x[2] for x in sorted_similarity_lst_ori[:(num//2)]], dim=0).to(device)
                    #adv
                    simi_adv = [x[0] for x in similarity_lst_adv]
                    simi_adv = torch.tensor(simi_adv)
                    # 获取按照 simi 排序后的索引
                    _, indices_adv = torch.sort(simi_adv, descending=True)
                    # 根据索引对 similarity_lst_adv 进行排序
                    sorted_similarity_lst_adv = [similarity_lst_adv[i] for i in indices_adv]
                    data_adv = torch.cat([x[1] for x in sorted_similarity_lst_adv[:(num//2)]], dim=0).to(device)
                    label_adv = torch.stack([x[2] for x in sorted_similarity_lst_adv[:(num//2)]], dim=0).to(device)

                    data = torch.vstack([data_ori, data_adv]).to(device)
                    label = torch.hstack([label_ori, label_adv]).to(device)

                elif args.search_by == "dist_search":
                    # select according to similarity
                    sorted_similarity_lst_ori = [x for x in similarity_lst_ori if x[0] > args.pt_searchsim_thread]
                    sorted_similarity_lst_adv = [x for x in similarity_lst_adv if x[0] > args.pt_searchsim_thread]
                    if (len(sorted_similarity_lst_ori) == 0) and (len(sorted_similarity_lst_adv) == 0):
                        break
                    #data只有离adv近的
                    elif (len(sorted_similarity_lst_ori) == 0) and (len(sorted_similarity_lst_adv) > 0):
                        break
                        #data = torch.cat([x[1] for x in sorted_similarity_lst_adv], dim=0).to(device)
                        #label = torch.stack([x[2] for x in sorted_similarity_lst_adv], dim=0).to(device)
                    #data只有离ori近的
                    elif (len(sorted_similarity_lst_ori) > 0) and (len(sorted_similarity_lst_adv) == 0):
                        break
                        #data = torch.cat([x[1] for x in sorted_similarity_lst_ori], dim=0).to(device)
                        #label = torch.stack([x[2] for x in sorted_similarity_lst_ori], dim=0).to(device)
                    #都找到了
                    else:
                        data_ori = torch.cat([x[1] for x in sorted_similarity_lst_ori], dim=0).to(device)
                        label_ori = torch.stack([x[2] for x in sorted_similarity_lst_ori], dim=0).to(device)
                        data_adv = torch.cat([x[1] for x in sorted_similarity_lst_adv], dim=0).to(device)
                        label_adv = torch.stack([x[2] for x in sorted_similarity_lst_adv], dim=0).to(device)

                        data = torch.vstack([data_ori, data_adv]).to(device)
                        label = torch.hstack([label_ori, label_adv]).to(device)



            else:
                raise NotImplementedError
            #此时数据和标签已经构建好了
            model.train()
            for n, param in model.named_parameters():
                if data_config.is_freeze:
                    if "fc" not in n:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                        #print(param)
                else:
                    param.requires_grad = True

            adv_input = data
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
train_loader = DataLoader(train_dataset, batch_size=data_config.batch_size, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
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
    if data_config.normalized:
        model = nn.Sequential(norm_layer, model).cuda()
    model = torch.nn.DataParallel(model)


model.eval()

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
        softmax_output = softmax(output, dim=-1).data
        acc_top1 = accuracy2(softmax(output, dim=-1).data, y) #softmax(output, dim=-1)是predicted
        pgd_acc_rep.update(acc_top1[0].item(), y.size(0))
        logger.info('Batch {}\tbase model acc: {:.4f}'.format(i + 1, pgd_acc_rep.avg))
        logger.info('Batch {}\toutput:{},y:{}'.format(i + 1, softmax(output, dim=-1).data, y))


    # evaluate post model
    with torch.no_grad():

        post_model, original_class, neighbour_class, _, _ = post_train(model, X, train_loader,
                                                                          train_loaders_by_class, data_config, softmax_output)

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
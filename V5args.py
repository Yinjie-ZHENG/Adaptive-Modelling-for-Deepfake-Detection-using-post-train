import os
import time
import torchvision.transforms as transforms
import torch
from models.SiameseNetwork.PostTrainNet import PostTrainNet_xception, Xception_loaded




class data_config:
    '''***********- Test data settings-*************'''
    test_on_data = "fgsm" #“normal”

    '''***********- Post Train Arguments-*************'''

    pt_setting = "default"#"near_boundary" #"default"
    #data used for pt
    pt_data = "balanced_train" #"balanced_train"; "train" "num_nearest_cosine" #"num_nearest_searchsim" #num_nearest,ori_neigh, ori_rand, train
    pt_searchsim_thread = 0.90
    #attack on pt_data, 也就是对pt_data采用什么attack,attack之后就是对抗样本
    pt_method = "deepfool" #”normal“对应无attack; "deepfool"对应deepfool攻击; "pgd"对应pgd攻击；
    search_by = "dist_search" #"num_search","dist_search"，”“
    measurement = "cosine_normalized"#"cosine","cosine_normalized"
    # 一次post_train经历几次iter
    pt_iter = 20#20#50
    #post_train的学习率
    pt_lr = 0.0001
    #attack_iters用于PGD攻击的iter数
    attack_iters = 40
    #attack_restart
    attack_restart = 1


    '''***********- dataset and directory-*************'''
    WORKERS = 0
    gpus = [0,1]
    batch_size = 16#16forbalanced,32fornobalanced
    dataset= "deepfake_faces"#'tiny_imagenet'#'mnist'#'imagenet'
    #Image Normalize Configs如果是True,表示网络中需要加入normalization模块；False则表示网络中不需要加入normalization模块
    normalized = True
    normalized_cfg_mean = [0.5, 0.5, 0.5]#[0.4802, 0.4481, 0.3975]
    normalized_cfg_std = [0.5, 0.5, 0.5] #[0.2302, 0.2265, 0.2262]

    input_size = 299
    num_class = 2
    train_val_split = [0.7, 0.3]
    assert train_val_split[0] + train_val_split[1] == 1

    if normalized:#如果Normalized是True，说明normalization层会加入到网络里，则transform不需要加入normalization了
        training_transforms = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        validation_transforms = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])
    else:
        normalize = transforms.Normalize(
            mean=normalized_cfg_mean,
            std=normalized_cfg_std)
        training_transforms = transforms.Compose([
                          transforms.Resize((input_size, input_size)),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          normalize
                          ])
        validation_transforms = transforms.Compose([
                            transforms.Resize((input_size, input_size)),
                            transforms.ToTensor(),
                            normalize
                            ])
        assert normalized is False

    '''**************- Model Configs -************************'''
    load_from_pkl = '/hdd7/yinjie/deepfake_ckpt_latest/lr_001_Xception_pretrained__deepfake_faces_2023-01-14_23:53:37/lr_001_Xception_pretrained_val_acc_0.2259642384775734_epoch_10.pkl'#'/hdd7/yinjie/deepfake_ckpt_latest/lr_01Xception_pretrained__deepfake_faces_2023-01-14_16:20:19/Xception_pretrained_best_params.pkl'#'/hdd7/yinjie/deepfake_ckpt/Xception_pretrained__deepfake_faces_2022-12-21_13:55:43/Xception_pretrained_best_params.pkl'
    load_from_pth = None#"/home/yinjie/FYP_/torch/ckpts/mnist/fgsm.pth"
    # check args validity
    if load_from_pkl is not None:
        assert load_from_pth is None
    if load_from_pth is not None:
        assert load_from_pkl is None

    is_freeze = False
    model_arch = PostTrainNet_xception(load_from_pkl, freeze=is_freeze)#Xception_loaded(load_from_pkl) #
    model_save_name = "debugging"#"PostTrainNet_V2_boundary_nofreeze"#"PostTrainNet_Deepfool_boundary"#"PostTrainNet_nofreeze_nobalanceddata_noattack"#""#"PostTrainNet_xception_pretrained_lr001bz32"
    model_save_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    writer_log = model_save_name+"_"+model_save_time
    #具体到文件,load_from_pkl加载经过自己代码得到的模型dict,load_from_pth加载别人的模型权重用load_state_dict


    #model_attack_dataset_time
    MODEL_PATH = './pt_results/'
    work_dir = os.path.join(MODEL_PATH,model_save_name+"_"+dataset+"_"+model_save_time)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

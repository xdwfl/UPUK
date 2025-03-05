import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random
import sys
import torch.nn.functional as F
from torchvision.transforms import Normalize
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import warnings
warnings.filterwarnings('ignore')
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
txt_dir_name = "oda_OH_"+str(datetime.datetime.now())+".txt"
txt_save_dir = os.path.join("./log",txt_dir_name)

from train_target_mods import obtain_label,obtain_label_dpm,cal_acc, best_cluster_fit


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return  transforms.Compose([
          transforms.Resize((resize_size, resize_size)),
          transforms.RandomCrop(crop_size),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize
      ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return  transforms.Compose([
          transforms.Resize((resize_size, resize_size)),
          transforms.CenterCrop(crop_size),
          transforms.ToTensor(),
          normalize
      ])

def data_load(args):

    # prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.tar_classes)):
            label_map_s[args.tar_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders



def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_target(args):
    args.txt_save_dir = txt_save_dir
    with open(txt_save_dir, "a") as f:
        f.write('Begin Training' + '\n')
    dset_loaders = data_load(args)

    #set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    # source pre-trianed model
    args.modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    start = True
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, 256)
    score_bank = torch.randn(num_sample, args.class_num).cuda()
    netF.eval()
    netB.eval()
    netC.eval()


    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            indx = data[-1]
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            output_norm = F.normalize(output)
            outputs = netC(output)
            outputs = nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()

    tt = 0
    iter_num = 0
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval

    known_number = 0
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()

        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0:
            netF.eval()
            netB.eval()
            if iter_num // interval_iter > 1 and iter_num // interval_iter < 4: #perform diffuser after epochs, to obtain more high-confident samples
                mem_label, ENT_THRESHOLD, pred_dpm = obtain_label_dpm(dset_loaders['test'], netF, netB, netC, args)
                pred_dpm_reasgn, _, _ = best_cluster_fit(mem_label, pred_dpm.cpu().numpy())

                mem_label = torch.from_numpy(mem_label).cuda()
                if pred_dpm.max() > mem_label.max():
                    # judge if the class number perdiction is correctly perfomed;
                    #  if yes, obtain the bool result about which samples are high-confident; if not, diffuser is invalid and ignored
                    high_con_bool = torch.zeros([mem_label.cpu().shape[0]]).bool()
                    high_con_bool[torch.where(pred_dpm_reasgn == mem_label)[0]] = True
                else:
                    high_con_bool = torch.ones([mem_label.cpu().shape[0]]).bool()
            else:
                mem_label, ENT_THRESHOLD = obtain_label(dset_loaders['test'], netF, netB, netC, args)
                mem_label = torch.from_numpy(mem_label).cuda()
                high_con_bool = torch.ones([mem_label.cpu().shape[0]]).bool()
            known_bool = torch.zeros([mem_label.cpu().shape[0]]).bool()
            known_bool[mem_label< args.class_num] = True
            netF.train()
            netB.train()

        high_con_bool_iter = high_con_bool[tar_idx]
        known_bool_iter = known_bool[tar_idx]
        if torch.where(high_con_bool_iter*known_bool_iter==True)[0].shape[0]>0:
            high_known_idx = torch.where(high_con_bool_iter*known_bool_iter==True)[0]
        else:
            high_known_idx = torch.where(known_bool_iter==True)[0]

        inputs_test = inputs_test.cuda()

        iter_num += 1
        alpha =  (1 + 10 * iter_num / max_iter)**(-args.beta)
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        pred = mem_label[tar_idx]
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        softmax_out = nn.Softmax(dim=1)(outputs_test)
        outputs_test_known = outputs_test[high_known_idx,:]
        features_known = features_test[high_known_idx,:]
        known_inx = tar_idx[high_known_idx]
        pred = pred[high_known_idx]
        known_number += pred.shape[0]

        if len(pred) == 0:
            print(tt)
            del features_test
            del outputs_test
            tt += 1
            continue

        classifier_loss = 0
        with torch.no_grad():
            output_f_norm = F.normalize(features_known)
            output_f_ = output_f_norm.cpu().detach().clone()

            softmax_out_known = nn.Softmax(dim=1)(outputs_test_known)
            pred_bs = softmax_out_known

            fea_bank[known_inx] = output_f_.detach().clone().cpu()
            score_bank[known_inx] = pred_bs.detach().clone()

            distance = output_f_ @ fea_bank.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=3 + 1)
            idx_near = idx_near[:, 1:]  #batch x K
            score_near = score_bank[idx_near]  #batch x K x C


            fea_near = fea_bank[idx_near]  #batch x K x num_dim
        # nn
        softmax_out_un = softmax_out_known.unsqueeze(1).expand(
            -1, 3, -1)
        loss = torch.mean((F.kl_div(softmax_out_un,
                                    score_near,
                                    reduction='none').sum(-1)).sum(1))

        mask = torch.ones((inputs_test.shape[0], inputs_test.shape[0]))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        copy = softmax_out.T

        dot_neg = softmax_out @ copy
        dot_neg = (dot_neg * mask.cuda()).sum(-1)
        neg_pred = torch.mean(dot_neg)
        #L_con
        loss += neg_pred * alpha
        classifier_loss += loss


        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            known_number = 0

            netF.train()
            netB.train()

    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
    return netF, netB, netC




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UPUK',allow_abbrev=False)
    parser.add_argument('--gpu_id', type=str, nargs='?', default='2', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size") #64
    parser.add_argument('--worker', type=int, default=0, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['office-home'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='target_b1/')
    parser.add_argument('--output_src', type=str, default='ckps/source/')
    parser.add_argument('--da', type=str, default='oda', choices=['oda'])
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--beta', type=float, default=1, help="beta")
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art','Product','RealWorld']
        args.class_num = 65

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    with open(txt_save_dir, "a") as f:
        f.write('Start 1 running' + '\n')
    for i in range(0,len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = args.t_dset_path

        if args.dset == 'office-home':
            if args.da == 'oda':
                args.class_num = 25
                args.src_classes = [i for i in range(25)]
                args.tar_classes = [i for i in range(65)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)


        args.savename = 'par_' + str(args.cls_par)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_target(args)


import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import loss
from torch.utils.data import DataLoader
import random
import sys
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from torchvision.transforms import Normalize
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import linear_sum_assignment as linear_assignment
from os_sfda.Diffuser import Diffuser_oda
from os_sfda.Diffuser.utils import cluster_acc,cluster_acc_wedefine
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI





def best_cluster_fit(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_assignment(w.max() - w)
    best_fit = []
    for i in range(y_pred.size):
        for j in range(len(row_ind)):
            if row_ind[j] == y_pred[i]:
                best_fit.append(col_ind[j])
    return torch.tensor(np.array(best_fit)).cuda(), row_ind, col_ind



def comp_hos(all_label,pred_dpm,args):
    if args.dset == 'office-home' or args.dset == 'office': #officehome(65 and 25+1) or office(31 and 20+1)
        label_26 = all_label.clone()
        unknown_idx_dpm = torch.where(label_26 > (args.class_num-1))[0]
        known_idx_dpm = torch.where(label_26 < args.class_num)[0]
        label_26[torch.where(label_26 > (args.class_num-1))[0]] = args.class_num
        pred_dpm_reasgn_65, _, _ = best_cluster_fit(all_label.numpy(), pred_dpm)
        pred_dpm_reasgn_26 = pred_dpm_reasgn_65.clone()
        pred_dpm_reasgn_26[torch.where(pred_dpm_reasgn_26 > (args.class_num-1))[0]] = args.class_num
        known_dpm_acc = (len(torch.where(pred_dpm_reasgn_26.cpu()[known_idx_dpm] == label_26[known_idx_dpm])[0]))/len(known_idx_dpm)
        unknown_dpm_acc = (len(torch.where(pred_dpm_reasgn_26.cpu()[unknown_idx_dpm] == label_26[unknown_idx_dpm])[0]))/len(unknown_idx_dpm)
        hos_dpm = (2 * known_dpm_acc * unknown_dpm_acc) / (known_dpm_acc + unknown_dpm_acc)

        all_dpm_acc = (len(torch.where(pred_dpm_reasgn_65.cpu() == all_label)[0])) / len(torch.where(all_label)[0])
        return known_dpm_acc, unknown_dpm_acc, hos_dpm, all_dpm_acc
    elif args.dset == 'VISDA-C':
        per_class_num = np.zeros((args.class_num))
        per_class_correct = np.zeros((args.class_num)).astype(np.float32)
        per_known_class_acc = np.zeros((args.class_num))
        label_7 = all_label.clone()
        label_7[torch.where(label_7 > (args.class_num - 1))[0]] = args.class_num
        pred_dpm_reasgn_12, _, _ = best_cluster_fit(all_label.numpy(), pred_dpm)
        pred_dpm_reasgn_7 = pred_dpm_reasgn_12.clone()
        pred_dpm_reasgn_7[torch.where(pred_dpm_reasgn_7 > (args.class_num - 1))[0]] = args.class_num
        for t in range(args.class_num):
            t_ind = torch.where(label_7 == t)[0]
            if len(torch.where(pred_dpm_reasgn_7[t_ind] == t)[0])==0:
                per_class_correct[t] = 0.0
                per_class_num[t] = 0.0
                per_known_class_acc[t] = 0.0
            else:
                correct_ind = torch.where(pred_dpm_reasgn_7[t_ind] == t)[0]
                per_class_correct[t] = float(len(correct_ind))
                per_class_num[t] = float(len(t_ind))
                per_known_class_acc[t] = per_class_correct[t] / per_class_num[t]

        unknown_idx_dpm = torch.where(label_7 > (args.class_num-1))[0]
        unknown_dpm_acc = (len(torch.where(pred_dpm_reasgn_7.cpu()[unknown_idx_dpm] == label_7[unknown_idx_dpm])[0])) / len(unknown_idx_dpm)

        all_dpm_acc = (len(torch.where(pred_dpm_reasgn_12.cpu() == all_label)[0])) / len(torch.where(all_label)[0])
        known_dpm_acc = per_known_class_acc.mean()
        hos_dpm = (2 * known_dpm_acc * unknown_dpm_acc) / (known_dpm_acc + unknown_dpm_acc)
        return per_known_class_acc, unknown_dpm_acc, all_dpm_acc,known_dpm_acc,hos_dpm


def obtain_label_dpm(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_fea_dpm,all_label_dpm= all_fea.clone(),all_label.clone()

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    low_cfd_idx = torch.where(all_output.max(1)[0]<0.6)[0]    #Adjust the confidence threshold #select high_confidence pesudo_label
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()
    from sklearn.cluster import KMeans
    kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
    labels = kmeans.predict(ent.reshape(-1,1))

    idx = np.where(labels==1)[0]
    iidx = 0
    if ent[idx].mean() > ent.mean():
        iidx = 1
    known_idx = np.where(kmeans.labels_ != iidx)[0]

    all_fea = all_fea[known_idx,:]
    all_output = all_output[known_idx,:]
    predict = predict[known_idx]
    all_label_idx = all_label[known_idx]
    ENT_THRESHOLD = (kmeans.cluster_centers_).mean()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    guess_label = args.class_num * np.ones(len(all_label), )
    guess_label[known_idx] = pred_label

    guess_label_highcfd = torch.Tensor(guess_label).clone()
    guess_label_highcfd[low_cfd_idx] = args.class_num+1 #mark the low-confident samples

    _,inf_K_ideal,_ = Diffuser_oda.train_cluster_net(all_fea_dpm, all_label_dpm, None, all_output.shape[1], args.txt_save_dir, 'do_infer')
    pred_dpm_semi,_,_ = Diffuser_oda.train_cluster_net(all_fea_dpm, all_label_dpm, guess_label_highcfd.long(), inf_K_ideal, args.txt_save_dir, 'only_optimize_w_semi')


    if args.dset == 'office-home' or args.dset == 'office':
        kn_dpm_acc_semi, unkn_dpm_acc_semi, hos_dpm_semi,all_dpm_acc_semi = comp_hos(all_label, pred_dpm_semi,args)
        log_str2 = 'Task: {}; Diffuser_with_semi: os*_acc = {:.6f}, unknown_acc = {:.6f}, hos = {:.6f}, all_acc = {:.6f}.'.format(args.name, kn_dpm_acc_semi, unkn_dpm_acc_semi, hos_dpm_semi, all_dpm_acc_semi)

        with open(args.txt_save_dir, "a") as f:
            f.write(log_str2 + '\n')
        print(log_str2)

    elif args.dset == 'VISDA-C':
        per_kn_dpm_acc_semi, unkn_dpm_acc_semi, all_dpm_acc_semi,known_dpm_acc_semi,hos_dpm_semi = comp_hos(all_label, pred_dpm_semi, args)
        log_str2 = 'Diffuser_with_semi: os*_acc(per) = {:.6f}, unknown_acc = {:.6f}, hos_acc = {:.6f}, all_acc = {:.6f}.'.format(
            known_dpm_acc_semi, unkn_dpm_acc_semi, hos_dpm_semi, all_dpm_acc_semi)
        log_str20 = 'Diffuser_with_semi: known_acc(per) = {:.6f}, = {:.6f}, = {:.6f}, = {:.6f}, = {:.6f}, = {:.6f}'.format(
            per_kn_dpm_acc_semi[0], per_kn_dpm_acc_semi[1], per_kn_dpm_acc_semi[2],per_kn_dpm_acc_semi[3], per_kn_dpm_acc_semi[4], per_kn_dpm_acc_semi[5])

        with open(args.txt_save_dir, "a") as f:
            f.write(log_str2 + '\n')
            f.write(log_str20 + '\n')
        print(log_str2)
        print(log_str20)


    return guess_label.astype('int'), ENT_THRESHOLD, torch.tensor(pred_dpm_semi).cuda()



'''
borrowed from https://github.com/Albert0147/AaD_SFDA 
'''

def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()

    from sklearn.cluster import KMeans
    kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
    labels = kmeans.predict(ent.reshape(-1,1))

    idx = np.where(labels==1)[0]
    iidx = 0
    if ent[idx].mean() > ent.mean():
        iidx = 1
    known_idx = np.where(kmeans.labels_ != iidx)[0]

    all_fea = all_fea[known_idx,:]
    all_output = all_output[known_idx,:]
    predict = predict[known_idx]
    all_label_idx = all_label[known_idx]
    ENT_THRESHOLD = (kmeans.cluster_centers_).mean()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    guess_label = args.class_num * np.ones(len(all_label), )
    guess_label[known_idx] = pred_label


    acc = np.sum(guess_label == all_label.float().numpy()) / len(all_label_idx)
    log_str = 'Threshold = {:.2f}, Accuracy = {:.2f}% -> {:.2f}%'.format(ENT_THRESHOLD, accuracy*100, acc*100)

    return guess_label.astype('int'), ENT_THRESHOLD



def cal_acc(loader, netF, netB, netC,args, flag=False, threshold=0.1):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_label[torch.where(all_label > (args.class_num - 1))[0]] = args.class_num
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    if flag:
        all_output = nn.Softmax(dim=1)(all_output)
        ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)

        from sklearn.cluster import KMeans
        kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
        labels = kmeans.predict(ent.reshape(-1,1))
        idx = np.where(labels==1)[0]
        iidx = 0
        if ent[idx].mean() > ent.mean():
            iidx = 1
        predict_12 = predict.clone()
        predict[np.where(labels==iidx)[0]] = args.class_num

        if args.dset == 'VISDA-C':
            per_class_num = np.zeros((args.class_num))
            per_class_correct = np.zeros((args.class_num)).astype(np.float32)
            per_known_class_acc = np.zeros((args.class_num))
            label_7 = all_label.clone()
            label_7[torch.where(label_7 > (args.class_num - 1))[0]] = args.class_num
            for t in range(args.class_num):
                t_ind = torch.where(label_7 == t)[0]
                if len(torch.where(predict[t_ind] == t)[0]) == 0:
                    per_class_correct[t] = 0.0
                    per_class_num[t] = 0.0
                    per_known_class_acc[t] = 0.0
                else:
                    correct_ind = torch.where(predict[t_ind] == t)[0]
                    per_class_correct[t] = float(len(correct_ind))
                    per_class_num[t] = float(len(t_ind))
                    per_known_class_acc[t] = per_class_correct[t] / per_class_num[t]

            unknown_idx = torch.where(label_7 > (args.class_num - 1))[0]
            unknown_aad_acc = (len(torch.where(predict.cpu()[unknown_idx] == label_7[unknown_idx])[0])) / len(unknown_idx)
            all_aad_acc = (len(torch.where(predict_12.cpu() == all_label)[0])) / len(torch.where(all_label)[0])
            known_aad_acc = per_known_class_acc.mean()
            hos_aad = (2 * known_aad_acc * unknown_aad_acc) / (known_aad_acc + unknown_aad_acc)
            return per_known_class_acc, unknown_aad_acc, all_aad_acc, known_aad_acc, hos_aad
        else:
            matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
            matrix = matrix[np.unique(all_label).astype(int),:]
            acc = matrix.diagonal()/matrix.sum(axis=1) * 100
            unknown_acc = acc[-1:].item()
            return np.mean(acc[:-1]), np.mean(acc), unknown_acc
    else:
        return accuracy*100, mean_ent


def cal_acc_65(loader, netF, netB, netC,args, flag=False, threshold=0.1):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    if flag:
        all_output = nn.Softmax(dim=1)(all_output)
        ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)

        from sklearn.cluster import KMeans
        kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
        labels = kmeans.predict(ent.reshape(-1,1))
        idx = np.where(labels==1)[0]
        iidx = 0
        if ent[idx].mean() > ent.mean():
            iidx = 1
        predict[np.where(labels==iidx)[0]] = args.class_num

        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label).astype(int),:]
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        unknown_acc = np.mean(acc[25:])#.item()
        return np.mean(acc[:25]), np.mean(acc), unknown_acc
    else:
        return accuracy*100, mean_ent


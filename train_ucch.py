import os
import argparse
import numpy as np
import torch.cuda
import torchvision.models as models
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from dataset import CMDataset
from cmdataset import get_trans_data, get_data

from model.image_module import ImageModule, ImgNet
from model.text_module import TextModule, TxtNet
from model.NCECriterion import NCESoftmaxLoss
from model.NCEAverage import NCEAverage
from itertools import cycle
from utils import *
from losses import *
from evaluate import *

import logging

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', default='/root/dataset/')
    parser.add_argument('--log-dir', default='./log/')
    parser.add_argument('--ckpt-dir', default='./ckpt/')
    parser.add_argument('--dataset', default='nuswide')
    parser.add_argument('--n-bits', '-bit', type=int, default=16)
    parser.add_argument('--Epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--arch', default="vggnet")
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--th_up', type=float, default=0.95)
    parser.add_argument('--th_dn', type=float, default=0.05)
    parser.add_argument('--mu', type=float, default=0.7)
    parser.add_argument('--nesterov', action='store_true', default=True)
    parser.add_argument('--warmup', default=1, type=int)
    parser.add_argument('--wd', default=1e-6, type=float)
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-labeled', default=10000, type=int)
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument("--a", default=0.7, type=float)
    parser.add_argument("--hiden-layer", default=[3, 2], nargs='+')
    parser.add_argument('--margin', type=float, default=.2)
    parser.add_argument('--momentum', type=float, default=0.4)
    parser.add_argument('--shift', type=float, default=.1)
    parser.add_argument('--K', type=int, default=4096)
    parser.add_argument('--T', type=float, default=.9)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    return parser.parse_args()

def set_train(is_warmup=False):
    img_net.train()
    if is_warmup:
        img_net.features.eval()
        img_net.features.requires_grad_(False)
    else:
        img_net.features.requires_grad_(True)
    txt_net.train()

def evaluate():
    img_net.eval()
    txt_net.eval()

    query_image_code = []
    query_text_code = []
    query_label = []
    retrieval_label = []
    for labeled_image, labeled_text, label in query_loader:
        query_label.append(label)
        labeled_image = labeled_image.float().cuda()
        labeled_text = labeled_text.float().cuda()
        labeled_image_output = torch.sign(img_net(labeled_image)[0])
        labeled_text_output = torch.sign(txt_net(labeled_text)[0])
        query_image_code.append(labeled_image_output.data.cpu().numpy())
        query_text_code.append(labeled_text_output.data.cpu().numpy())
    query_image_code = np.concatenate(query_image_code)
    query_text_code = np.concatenate(query_text_code)

    retrieval_image_code = []
    retrieval_text_code = []
    for labeled_image, labeled_text, label in retrieval_loader:
        retrieval_label.append(label)
        labeled_image = labeled_image.float().cuda()
        labeled_text = labeled_text.float().cuda()
        labeled_image_output = torch.sign(img_net(labeled_image)[0])
        labeled_text_output = torch.sign(txt_net(labeled_text)[0])
        retrieval_image_code.append(labeled_image_output.data.cpu().numpy())
        retrieval_text_code.append(labeled_text_output.data.cpu().numpy())
    retrieval_image_code = np.concatenate(retrieval_image_code)
    retrieval_text_code = np.concatenate(retrieval_text_code)
    map_500_i2t = mean_average_precision(torch.from_numpy(query_image_code).float(), torch.from_numpy(retrieval_text_code).float(), torch.cat(query_label).float(), torch.cat(retrieval_label).float(), torch.device('cuda'), 500)
    map_all_i2t = mean_average_precision(torch.from_numpy(query_image_code).float(), torch.from_numpy(retrieval_text_code).float(), torch.cat(query_label).float(), torch.cat(retrieval_label).float(), torch.device('cuda'))
    map_500_t2i = mean_average_precision(torch.from_numpy(query_text_code).float(), torch.from_numpy(retrieval_image_code).float(), torch.cat(query_label).float(), torch.cat(retrieval_label).float(), torch.device('cuda'), 500)
    map_all_t2i = mean_average_precision(torch.from_numpy(query_text_code).float(), torch.from_numpy(retrieval_image_code).float(), torch.cat(query_label).float(), torch.cat(retrieval_label).float(), torch.device('cuda'))
    map_500_i2i = mean_average_precision(torch.from_numpy(query_image_code).float(), torch.from_numpy(retrieval_image_code).float(), torch.cat(query_label).float(), torch.cat(retrieval_label).float(), torch.device('cuda'), 500)
    map_all_i2i = mean_average_precision(torch.from_numpy(query_image_code).float(), torch.from_numpy(retrieval_image_code).float(), torch.cat(query_label).float(), torch.cat(retrieval_label).float(), torch.device('cuda'))
    map_500_t2t = mean_average_precision(torch.from_numpy(query_text_code).float(), torch.from_numpy(retrieval_text_code).float(), torch.cat(query_label).float(), torch.cat(retrieval_label).float(), torch.device('cuda'), 500)
    map_all_t2t = mean_average_precision(torch.from_numpy(query_text_code).float(), torch.from_numpy(retrieval_text_code).float(), torch.cat(query_label).float(), torch.cat(retrieval_label).float(), torch.device('cuda'))
    
    logger.info(f"[=>mAP info<=]")
    logger.info("__" * 50)
    logger.info(f"     I -> T mAP@500: {map_500_i2t:.4f} \t      mAP@All: {map_all_i2t:.4f}")
    logger.info(f"     T -> I mAP@500: {map_500_t2i:.4f} \t      mAP@All: {map_all_t2i:.4f}")
    logger.info(f"     I -> I mAP@500: {map_500_i2i:.4f} \t      mAP@All: {map_all_i2i:.4f}")
    logger.info(f"     T -> T mAP@500: {map_500_t2t:.4f} \t      mAP@All: {map_all_t2t:.4f}")


    global map_all_max_i2t, map_all_max_t2i, map_500_max_i2t, map_500_max_t2i
    if map_all_max_i2t + map_all_max_t2i < map_all_i2t + map_all_t2i:
        map_500_max_i2t = map_500_i2t
        map_500_max_t2i = map_500_t2i
        map_all_max_i2t = map_all_i2t
        map_all_max_t2i = map_all_t2i
        logger.info("[=>Best Update!<=]")
    logger.info(f"Best I -> T mAP@500: {map_500_max_i2t:.4f}, Best I -> T mAP@All: {map_all_max_i2t:.4f}")
    logger.info(f"Best T -> I mAP@500: {map_500_max_t2i:.4f}, Best T -> I mAP@All: {map_all_max_t2i:.4f}")

def ACC(pred, y):
    acc_num = 0
    tot_num, num_classes = pred.shape
    for i in range(tot_num):
        p = True
        for j in range(num_classes):
            if pred[i][j] != y[i][j]:
                p = False
                break
        if p: acc_num += 1
    return acc_num

def Psu(pred):
    mask_id = []
    p_lb = []
    for ui, uw_ip in enumerate(pred):
        p = True
        p_lb.append([0] * args.num_classes)
        for uj, j in enumerate(uw_ip):
            if j > args.th_up:
                p_lb[ui][uj] = 1
            elif j > args.th_dn:
                p = False
                break
        if p: mask_id.append(ui)
    return p_lb, mask_id

def CalcAcc(pred, y):
    p_lb, mask_id = Psu(pred)
    if len(mask_id) == 0: return 0, [], []
    pp_lb = [p_lb[m_id] for m_id in mask_id]
    r_lb = [y[m_id] for m_id in mask_id]
    p_lb = torch.tensor(pp_lb)
    r_lb = torch.stack(r_lb)
    Acc = ACC(p_lb, r_lb) 
    return Acc, mask_id, p_lb

args = parse_arguments()
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    filename=f"./log/ucch_{args.num_labeled}.log", 
)

def train(epoch):
    set_train(epoch < args.warmup)
    Loss = 0
    for batch_idx, (idx, imgs, txts, _) in enumerate(train_loader):
        imgs, txts, idx = imgs.cuda(), txts.cuda(), idx.cuda()
        img_code, _ = img_net(imgs.float())
        txt_code, _ = txt_net(txts.float())
        out_l, out_ab = contrast(img_code, txt_code, torch.tensor(idx), epoch=epoch-args.warmup)
        l_loss = criterion_contrast(out_l)
        ab_loss = criterion_contrast(out_ab)
        Lc = l_loss + ab_loss
        Lr = criterion(img_code, txt_code)
        loss = Lc * args.alpha + Lr * (1. - args.alpha)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(parameters, 1.)
        optimizer.step()
        Loss += loss.item()
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | LR: %g' % (Loss / (batch_idx + 1), optimizer.param_groups[0]['lr']))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    for args.n_bits in [16]:
        #train_dataset, _, _, train_loader, retrieval_loader, query_loader = get_data(args, need_loader=True, return_index=True)
        train_dataset = CMDataset(
            return_index=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

        retrieval_dataset = CMDataset(
            partition='retrieval'
        )
        retrieval_loader = torch.utils.data.DataLoader(
            retrieval_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )

        test_dataset = CMDataset(
            partition='test'
        )
        query_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        args.num_classes = train_dataset.num_classes
        args.text_dim = train_dataset.text_dim
        #train_labeled_loader, _, _ = get_trans_data(args, need_loader=True, return_index=True)
        criterion = ContrastiveLoss(args.margin, args.shift)
        n_data = len(train_loader.dataset)
        contrast = NCEAverage(args.n_bits, n_data, args.K, args.T, args.momentum)
        criterion_contrast = NCESoftmaxLoss()
        contrast = contrast.cuda()
        criterion_contrast = criterion_contrast.cuda()
        CLoss = torch.nn.BCELoss()
        img_net = ImgNet(arch=args.arch, bit=args.n_bits, num_classes=args.num_classes, pretrain=True)
        img_net.cuda()
        txt_net = TxtNet(text_dim=args.text_dim, bit=args.n_bits, num_classes=args.num_classes)
        txt_net.cuda()
        parameters = list(img_net.parameters()) + list(txt_net.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.wd)
        lr_schedu = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90, 120], gamma=0.1)
        
        logger.info("====================Start Train====================")
        logger.info(f"Code Length: {args.n_bits}, \t Text Dim: {args.text_dim}, \t Classes Num: {args.num_classes} \t Labeled Num: {args.num_labeled}")


        map_500_max_i2t = 0
        map_500_max_t2i = 0
        map_all_max_i2t = 0
        map_all_max_t2i = 0

        for epoch in range(args.Epochs):
            logger.info(f"[Epoch]: {epoch + 1}")
            train(epoch)
            set_train(epoch < args.warmup)
            
            if epoch % 5 == 4:
                evaluate()
                logger.info("__" * 50 + "\n")
            else:
                logger.info("--" * 50 + "\n")
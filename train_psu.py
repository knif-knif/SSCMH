import os
import argparse
import numpy as np
import torch.cuda
import torchvision.models as models
import torch.nn as nn
from dataset import *
from cmdataset import get_trans_data, get_data
from model.image_module import ImageModule
from model.text_module import TextModule
from itertools import cycle
from utils import *
from losses import *
from evaluate import *

import logging

def parse_arguments():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', default="NUS-WIDE-10K")

    parser.add_argument('--code_length', type=int, default=32,
                        help='length of hash code')
    parser.add_argument('--labeled_ratio', type=float, default=0.3,
                        help='ratio of labeled image-text pairs')
    parser.add_argument('--epoch_num', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--labeled_batch_size', type=int, default=32,
                        help='number of labeled image-text pairs in a batch')
    parser.add_argument('--unlabeled_batch_size', type=int, default=32,
                        help='number of unlabeled image-text pairs in a batch')

    parser.add_argument('--data-dir', default='/root/dataset/')
    parser.add_argument('--log-dir', default='./log/')
    parser.add_argument('--ckpt-dir', default='./ckpt/')
    parser.add_argument('--dataset', default='nuswide')
    parser.add_argument('--n-bits', '-bit', type=int, default=16)
    parser.add_argument('--Epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--log_interval', type=int, default=40)
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--th_up', type=float, default=0.95)
    parser.add_argument('--th_dn', type=float, default=0.05)
    parser.add_argument('--mu', type=float, default=0.7)
    parser.add_argument('--nesterov', action='store_true', default=True)
    parser.add_argument('--warmup', default=5, type=int)
    parser.add_argument('--wdecay', default=5e-4, type=float)
    parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--total-steps', default=100, type=int)
    parser.add_argument('--num-labeled', default=4000, type=int)
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument("--expand-labels", action="store_true")
    parser.add_argument("--a", default=0.7, type=float)
    parser.add_argument("--hiden-layer", default=[3, 2], nargs='+')
    parser.add_argument("--arch", default="resnet")
    parser.add_argument('--margin', type=float, default=.2)
    parser.add_argument('--momentum', type=float, default=0.4)
    parser.add_argument('--shift', type=float, default=.1)
    parser.add_argument('--K', type=int, default=4096)
    parser.add_argument('--T', type=float, default=.9)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    return parser.parse_args()


def evaluate():
    image_model.eval()
    text_model.eval()

    query_image_code = []
    query_text_code = []
    query_label = []
    retrieval_label = []
    for labeled_image, labeled_text, label in query_loader:
        query_label.append(label)
        labeled_image = labeled_image.float().cuda()
        labeled_text = labeled_text.float().cuda()
        labeled_image_output = torch.sign(image_model(labeled_image)[0])
        labeled_text_output = torch.sign(text_model(labeled_text)[0])
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
        labeled_image_output = torch.sign(image_model(labeled_image)[0])
        labeled_text_output = torch.sign(text_model(labeled_text)[0])
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
    del retrieval_image_code, retrieval_text_code, query_image_code,query_text_code
    logger.info(f"[=>mAP info<=]")
    logger.info("__" * 50)
    logger.info(f"     I -> T mAP@500: {map_500_i2t:.4f} \t\t      mAP@All: {map_all_i2t:.4f}")
    logger.info(f"     T -> I mAP@500: {map_500_t2i:.4f} \t\t      mAP@All: {map_all_t2i:.4f}")
    logger.info(f"     I -> I mAP@500: {map_500_i2i:.4f} \t\t      mAP@All: {map_all_i2i:.4f}")
    logger.info(f"     T -> T mAP@500: {map_500_t2t:.4f} \t\t      mAP@All: {map_all_t2t:.4f}")


    global map_all_max_i2t, map_all_max_t2i, map_500_max_i2t, map_500_max_t2i
    F = False
    if map_all_max_i2t + map_all_max_t2i < map_all_i2t + map_all_t2i:
        map_500_max_i2t = map_500_i2t
        map_500_max_t2i = map_500_t2i
        map_all_max_i2t = map_all_i2t
        map_all_max_t2i = map_all_t2i
        F = True
    logger.info(f"Best I -> T mAP@500: {map_500_max_i2t:.4f}, Best I -> T mAP@All: {map_all_max_i2t:.4f}")
    logger.info(f"Best T -> I mAP@500: {map_500_max_t2i:.4f}, Best T -> I mAP@All: {map_all_max_t2i:.4f}")
    if F:
        logger.info("********************[=>Best Update!<=]*******************")



def Weakacc(pred, y):
    acc_num = 0
    tot_num, num_classes = pred.shape
    for i in range(tot_num):
        p = True
        if pred[i].sum() == 0: 
            tot_num -= 1
            continue
        for j in range(num_classes):
            if pred[i][j] == 1 and y[i][j] == 0:
                p = False
                break
        if p: acc_num += 1
    if tot_num == 0: return 0
    return acc_num

def Strongacc(pred, y):
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
    StrongAcc = Strongacc(p_lb, r_lb) 
    return StrongAcc, mask_id, p_lb

args = parse_arguments()
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    filename=f"./log/psu_cat_{args.num_labeled}.log", 
)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    CLoss = nn.BCELoss()
    for args.code_length in [16]:
        train_dataset, _, _, _, retrieval_loader, query_loader = get_data(args, need_loader=True)
        args.num_classes = train_dataset.num_classes
        args.tag_dim = train_dataset.text_dim
        train_labeled_loader, train_us_loader, train_uw_loader = get_trans_data(args, need_loader=True)
        logger.info(f"Code Length: {args.code_length}, \t Tag_dim: {args.tag_dim}, \t Num_classes: {args.num_classes}")
        # 加载图像网络
        image_model = ImageModule(args.code_length)
        image_model.cuda()

        # 加载文本网络
        text_model = TextModule(args.tag_dim, args.code_length)
        text_model.cuda()

        lr_i = 0.0001
        lr_t = 0.1
        lr_decay = np.exp(np.linspace(0, -8, args.epoch_num))

        map_500_max_i2t = 0
        map_500_max_t2i = 0
        map_all_max_i2t = 0
        map_all_max_t2i = 0

        for epoch in range(args.epoch_num):
            logger.info(f"[Epoch]: {epoch + 1}")

            image_model.train()
            text_model.train()
            image_optimizer = torch.optim.Adam(image_model.parameters(), lr=lr_i * lr_decay[epoch])
            text_optimizer = torch.optim.Adam(text_model.parameters(), lr=lr_t * lr_decay[epoch])

            Mask = 0
            Mask_U = 0
            cnt = 0
            cnt_u = 0
            Acc = 0
            UAcc = 0
            Loss_SM = 0
            Loss_CY = 0
            train_loader = zip(cycle(train_labeled_loader), train_us_loader, train_uw_loader)
            for idx, ((img, txt, label), (img_us, txt_us, label_u), (img_uw, txt_uw, _)) in enumerate(train_loader):
                cnt += len(label)
                cnt_u += len(label_u)
                img = img.float().cuda()
                txt = txt.float().cuda()
                label = label.float().cuda()
                label_u = label_u.float().cuda()
                img_code, img_pred = image_model(img)
                txt_code, txt_pred = text_model(txt)
                del img, txt
                pred = (img_pred + txt_pred) / 2.
                pred = torch.sigmoid(pred)
                loss_cy = CLoss(pred, label)
                
                bSA, mask_id, _ = CalcAcc(pred, label)
                Acc += bSA
                Mask += len(mask_id)

                img_us = img_us.float().cuda()
                img_uw = img_uw.float().cuda()
                txt_us = txt_us.float().cuda()
                txt_uw = txt_uw.float().cuda()
                img_uw_code, img_uw_pred = image_model(img_uw)
                img_us_code, img_us_pred = image_model(img_us)
                txt_uw_code, txt_uw_pred = text_model(txt_us)
                del img_uw, img_us, txt_us
                pred_u = (img_uw_pred + txt_uw_pred) / 2.
                pred_u = torch.sigmoid(pred_u)
                bSA, mask_id, p_lb = CalcAcc(pred_u, label_u)
                UAcc += bSA
                Mask_U += len(mask_id)
                if len(mask_id) != 0:
                    img_psu = [img_us_code[m_id] for m_id in mask_id]
                    txt_psu = [txt_uw_code[m_id] for m_id in mask_id]
                    p_lb = p_lb.cuda()
                    img_psu = torch.stack(img_psu)
                    txt_psu = torch.stack(txt_psu)
                    img_psu = img_psu.cuda()
                    txt_psu = txt_psu.cuda()
                    img_code = torch.cat((img_code, img_psu), 0)
                    txt_code = torch.cat((txt_code, txt_psu), 0)
                    label = torch.cat((label, p_lb), 0)
                sim = calculate_similarity(label, label)
                loss_sm = negative_log_likelihood_similarity_loss(img_code, txt_code, sim)
                loss = loss_sm + loss_cy
                Loss_SM += loss_sm.data.cpu().numpy()
                Loss_CY += loss_cy.data.cpu().numpy()
                image_optimizer.zero_grad()
                text_optimizer.zero_grad()
                loss.backward()
                image_optimizer.step()
                text_optimizer.step()
            logger.info("[Num]\t: num_label: {:<4d} \t\t num_unlabel: {:<4d}".format(Mask, Mask_U))
            logger.info("[Mask]\t: Mask     : {:.4f} \t Mask_U     : {:.4f}".format(Mask / cnt, Mask_U / cnt_u))
            logger.info("[Acc]\t: Label_Acc: {:.4f} \t Unlabel_Acc: {:.4f}".format(Acc / Mask, UAcc / Mask_U))
            logger.info("[Loss]\t: Loss_SM  : {:.6f} \t Loss_CY    : {:.6f}".format(Loss_SM / cnt, Loss_CY / cnt))
            logger.info("[LR]\t: Img_LR   : {:.8f}  Txt_LR     : {:.8f}".format(image_optimizer.param_groups[0]['lr'], text_optimizer.param_groups[0]['lr']))
            if epoch % 5 == 4:
                evaluate()
                logger.info("__" * 50 + "\n")
            else:
                logger.info("--" * 50 + "\n")







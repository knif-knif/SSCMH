from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import scipy.io as sio
import os
import math

from randaugment import RandAugmentMC

img_mean = (0.485, 0.456, 0.406)
img_std = (0.229, 0.224, 0.225)

class Sampler():
    def __init__(self, root, paths):
        self.root = root
        if isinstance(paths, np.ndarray):
            if len(paths.shape) == 1 or paths.shape[0] == 1 or paths.shape[1] == 1:
                paths = paths.reshape([-1]).tolist()
        
        self.paths = paths

    def __getitem__(self, item):
        path = self.paths[item]
        if isinstance(path, np.ndarray):
            if len(path.shape) >= 2:
                return Image.fromarray(path, mode='RGB')
            else:
                path = path[0]
        return Image.open(os.path.join(self.root, path))
    
    def __len__(self):
        return len(self.paths)

class TransformStrong(object):
    def __init__(self, mean, std):
        self.trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10, size=224)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def __call__(self, x):
        return self.normalize(self.trans(x))

class TransformWeak(object):
    def __init__(self, mean, std):
        self.trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224, 
                                  padding=int(224*0.125),
                                  padding_mode='reflect')
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def __call__(self, x):
        return self.normalize(self.trans(x))

def text_transform(text):
    return text

class CMDataset(data.Dataset):
    def __init__(self,
                 data_type='train',
                 data_dir=None,
                 return_index=False):
        self.data_type = data_type
        self.data_dir = data_dir
        training = 'train' in data_type.lower()
        mean = img_mean
        std = img_std
        trans = []
        if training:
            trans.extend([transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])])
        else:
            trans.extend([transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])])
        self.trans = trans
        self.return_index = return_index
        self.open_data()

    def open_data(self):
        data = DATAS(self.data_type, self.data_dir)

        if len(data) == 3:
            (self.imgs, self.texts, self.labels) = data
        else:
            (self.imgs, self.texts, self.labels, root) = data
            self.imgs = Sampler(root, self.imgs)
        self.length = self.labels.shape[0]
        self.num_classes = self.labels.shape[1]
        self.text_dim = self.texts.shape[1]

    def __getitem__(self, index):
        image = self.imgs[index]
        text = self.texts[index]
        if isinstance(self.imgs, Sampler):
            multi_crops = list(map(lambda trans: trans(image), self.trans))
            text = list(map(lambda trans: trans(text), [text_transform] * len(self.trans)))
        else:
            multi_crops = image
            text = text
        label = self.labels[index]

        if self.return_index:
            return index, multi_crops, text, label
        return multi_crops, text, label        
    
    def __len__(self):
        return self.length

class Trans_DATASET(data.Dataset):
    def __init__(self, root, indexs=None, data_type='train',
                 transform=None, target_transform=None, return_index=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        data = DATAS(data_type, self.root)
        (self.imgs, self.txts, self.labels) = data
        self.num_classes = self.labels.shape[1]
        self.text_dim = self.txts.shape[1]
        if indexs is not None:
            self.imgs = self.imgs[indexs]
            self.txts = self.txts[indexs]
            self.labels = self.labels[indexs]
        self.length = len(self.labels)
        self.return_index = return_index
    def __getitem__(self, index):
        img, txt, target = self.imgs[index],  self.txts[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_index:
            return index, img, txt, target
        return img, txt, target
        # return img, txt, target, index
    
    def __len__(self):
        return self.length
    
def DATAS(data_type, data_dir):
    data = np.load(os.path.join(data_dir, data_type + '.npz'))
    imgs = data['img']
    tags = data['txt']
    labels = data['label']
    del data
    return imgs, tags, labels

def x_u_split(args, labels):
    labeled_idx = np.array(range(args.num_labeled))
    unlabeled_idx = np.array(range(len(labels)))
    return labeled_idx, unlabeled_idx

def x_u_split_old(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = np.array(range(len(labels)))    
    for i in range(args.num_classes):
        idxs = []
        for idx, L in enumerate(labels):
            if L[i] == 1:
                idxs.append(idx)
        idxs = np.array(idxs)
        idxs = np.random.choice(idxs, label_per_class, False)
        labeled_idx.extend(idxs)
    labeled_idx = np.array(labeled_idx)
    # assert len(labeled_idx) == args.num_labeled
    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled
        )
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

def get_trans_data(args, need_loader=False, return_index=False):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=224, 
                              padding=int(224*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])
    base_labels = np.load(os.path.join(args.data_dir, args.dataset, 'train.npz'))['label']

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_labels
    )
    train_labeled_dataset = Trans_DATASET(args.data_dir + args.dataset, train_labeled_idxs, transform=transform_labeled, return_index=return_index)
    train_unlabeled_strong_dataset = Trans_DATASET(args.data_dir + args.dataset, train_unlabeled_idxs, transform=TransformStrong(img_mean, img_std))
    train_unlabeled_weak_dataset = Trans_DATASET(args.data_dir + args.dataset, train_unlabeled_idxs, transform=TransformWeak(mean=img_mean, std=img_std))

    if need_loader:
        train_labeled_loader = data.DataLoader(
            train_labeled_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True
        )
        train_us_loader = data.DataLoader(
            train_unlabeled_strong_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False
        )
        train_uw_loader = data.DataLoader(
            train_unlabeled_weak_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return train_labeled_loader, train_us_loader, train_uw_loader
    return train_labeled_dataset, train_unlabeled_strong_dataset, train_unlabeled_weak_dataset

def get_data(args, need_loader=False, return_index=False):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])
    train_dataset = Trans_DATASET(args.data_dir + args.dataset, indexs=None, data_type='train', transform=transform_train, return_index=return_index)
    retrieval_dataset = Trans_DATASET(args.data_dir + args.dataset, indexs=None, data_type='database', transform=transform_val)
    test_dataset = Trans_DATASET(args.data_dir + args.dataset, indexs=None, data_type='test', transform=transform_val)

    if need_loader:
        train_loader = data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        retrieval_dataloader = data.DataLoader(
            retrieval_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        query_loader = data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        return train_dataset, retrieval_dataset, test_dataset, train_loader, retrieval_dataloader, query_loader
    return train_dataset, retrieval_dataset, test_dataset

def get_cmdata(args):
    train_dataset = CMDataset(
        'train',
        args.data_dir + args.dataset,
        return_index=True
    )
    train_loader = torch.utils.data.DataLoader(
        trian_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    retrieval_dataset = CMDataset(
        'database',
        args.data_dir + args.dataset,
    )
    retrieval_loader = torch.utils.data.DataLoader(
        retrieval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_dataset = CMDataset(
        'test',
        args.data_dir + args.dataset,
    )
    query_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_dataset, retrieval_dataset, test_dataset, train_loader, retrieval_loader, query_loader

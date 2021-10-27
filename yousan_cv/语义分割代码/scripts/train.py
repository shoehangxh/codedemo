import argparse
import time
import datetime
import os
import shutil

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from dataloader import get_segmentation_dataset
from models.model_zoo import get_segmentation_model
from utils.loss_function import get_segmentation_loss
from utils.distributed import *
from utils.logger import setup_logger
from utils.lr_scheduler import WarmupPolyLR
from utils.scores import SegmentationMetric

# 参数设置
def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # 模型选择
    parser.add_argument('--model', type=str, default='pspnet',
                        choices=['fcn32s', 'fcn16s', 'fcn8s',
                                 'fcn', 'psp', 'deeplabv3', 'deeplabv3_plus',
                                 'danet', 'denseaspp', 'bisenet',
                                 'encnet', 'dunet', 'icnet',
                                 'enet', 'ocnet', 'ccnet', 'psanet',
                                 'cgnet', 'espnet', 'lednet', 'dfanet'],
                        help='model name (default: fcn32s)')
    # backbone
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet18', 'resnet50',
                                 'resnet101', 'resnet152', 'densenet121',
                                 'densenet161', 'densenet169', 'densenet201'],
                        help='backbone name (default: vgg16)')
    # 数据集
    parser.add_argument('--dataset', type=str, default='lips',
                        choices=['pascal_aug','neu','lips'],
                        help='dataset name (default: pascal_voc)')
    # 输入图像尺寸
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    # 裁剪尺寸
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    # 加载线程数
    parser.add_argument('--workers', '-j', type=int, default=4,
                        metavar='N', help='dataloader threads')
    # Joint Pyramid Upsampling
    parser.add_argument('--jpu', action='store_true', default=False,
                        help='JPU')
    # OHEM loss分支
    parser.add_argument('--use-ohem', type=bool, default=False,
                        help='OHEM Loss for cityscapes dataset')
    # Auxiliary loss分支
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    # Auxiliary loss所占权重
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    # batch size
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 8)')
    # 起始epoch
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    # 总的epoch
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of epochs to train (default: 50)')
    # 学习率
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    # 动量
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    # 衰减率
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    # 学习率升温
    parser.add_argument('--warmup-iters', type=int, default=0,
                        help='warmup iters')
    # 升温因子
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,
                        help='lr = warmup_factor * lr')
    # 升温方式
    parser.add_argument('--warmup-method', type=str, default='linear',
                        help='method of warmup')
    # cuda设置
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # 多cuda排序
    parser.add_argument('--local_rank', type=int, default=0)
    # 断点加载
    parser.add_argument('--resume', type=str, default=False,
                        help='put the path to resuming file if needed')
    # 存储路径
    parser.add_argument('--save-dir', default='~/.torch/models/',
                        help='Directory for saving checkpoint models')
    # 在某个epoch保存
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    # log文件保存
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    # 在某个iter保存
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    # 验证
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    # 跳过val
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    # 默认epoch设置
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'pascal_aug': 80,
            'pascal_voc': 50,
            'pcontext': 80,
            'ade20k': 160,
            'citys': 120,
            'sbu': 160,
        }
        args.epochs = epoches[args.dataset.lower()]
    # 默认学习率设置
    if args.lr is None:
        lrs = {
            'coco': 0.004,
            'pascal_aug': 0.001,
            'pascal_voc': 0.0001,
            'pcontext': 0.001,
            'ade20k': 0.01,
            'citys': 0.01,
            'sbu': 0.001,
        }
        args.lr = lrs[args.dataset.lower()] / 8 * args.batch_size
    return args

# 训练
class Trainer(object):
    def __init__(self, args):
        # 定义参数
        self.args = args
        # cuda
        self.device = torch.device(args.device)
        # transform模块
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # 读取数据
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        # 训练数据
        train_dataset = get_segmentation_dataset(args.dataset, split='train', mode='train', **data_kwargs)
        # 验证数据
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)
        # 计算每个epoch下的iter
        args.iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
        # 计算最大iter
        args.max_iters = args.epochs * args.iters_per_epoch
        # 训练数据的随机分批次
        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iters)
        # 验证数据的随机分批次
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)
        # 读取训练数据
        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)
        # 读取验证数据
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)
        # 根据是否分布式训练决定BN层的格式
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        # 获取model
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            aux=args.aux, jpu=args.jpu, norm_layer=BatchNorm2d).to(self.device)
        # 断点训练
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))
        # 获取loss
        self.criterion = get_segmentation_loss(args.model, use_ohem=args.use_ohem, aux=args.aux,
                                               aux_weight=args.aux_weight, ignore_index=-1).to(self.device)
        # 构建参数列表
        params_list = list()
        if hasattr(self.model, 'pretrained'):
            params_list.append({'params': self.model.pretrained.parameters(), 'lr': args.lr})
        if hasattr(self.model, 'exclusive'):
            for module in self.model.exclusive:
                params_list.append({'params': getattr(self.model, module).parameters(), 'lr': args.lr * 10})

        # 设置优化器
        self.optimizer = torch.optim.SGD(params_list,
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # 学习率策略
        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters=args.max_iters,
                                         power=0.9,
                                         warmup_factor=args.warmup_factor,
                                         warmup_iters=args.warmup_iters,
                                         warmup_method=args.warmup_method)

        # 分布式训练
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank)
        # 评价指标容器
        self.metric = SegmentationMetric(train_dataset.num_class)
        # 构建最优结果记录器
        self.best_pred = 0.0

    # 训练
    def train(self):
        # 保存至哪个GPU
        save_to_disk = get_rank() == 0
        # 获取epoch和max_iters
        epochs, max_iters = self.args.epochs, self.args.max_iters
        # 写入log的iter
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.args.iters_per_epoch
        # 保存模型
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        # 开始时间
        start_time = time.time()
        # 记录
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()
        model.train()
        # 在测试集内循环
        for i, (image, target) in enumerate(self.train_loader):
            image = image.to(self.device)
            target = target.to(self.device)
            outputs = model(image)
            loss_dict = self.criterion(outputs,target)
            losses = sum(loss for loss in loss_dict.values())
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            eta_seconds = ((time.time() - start_time) / i) * (max_iters - i)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            if i % log_per_iters == 0 and save_to_disk:
                logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        i, max_iters, self.optimizer.param_groups[0]['lr'], losses_reduced.item(),
                        str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))
            if i % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.model, self.args, is_best=False)
            if not self.args.skip_val and i % val_per_iters == 0:
                self.validation()
                self.model.train()
        # 保存模型
        save_checkpoint(self.model, self.args, is_best=False)
        # 总时间
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        # 记录时间
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    # 验证
    def validation(self):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        # 指标清零
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        # 转测试
        model.eval()
        # 在测试集内循环
        for i, (image, target) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))
        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        save_checkpoint(self.model, self.args, is_best)
        synchronize()

# 保存模型
def save_checkpoint(model, args, is_best=False):
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}.pth'.format(args.model, args.backbone, args.dataset)
    filename = os.path.join(directory, filename)
    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_{}_best_model.pth'.format(args.model, args.backbone, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    args.lr = args.lr * num_gpus
    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), filename='{}_{}_{}_log.txt'.format(
        args.model, args.backbone, args.dataset))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
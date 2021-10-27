import torch
import numpy as np

__all__ = ['SegmentationMetric', 'batch_pix_accuracy', 'batch_intersection_union',
           'pixelAccuracy', 'intersectionAndUnion', 'hist_info', 'compute_score']

# 评价指标计算
class SegmentationMetric(object):
    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
        # 类别数目
        self.nclass = nclass
        # 清零
        self.reset()
    # 更新结果（参数为：1、预测值；2、标签）
    def update(self, preds, labels):
        def evaluate_worker(self, pred, label):
            # 获取批次准确率
            correct, labeled = batch_pix_accuracy(pred, label)
            # 获取批次的iou值
            inter, union = batch_intersection_union(pred, label, self.nclass)
            # 统计总的正确数
            self.total_correct += correct
            # 统计总的标签数
            self.total_label += labeled
            #
            if self.total_inter.device != inter.device:
                self.total_inter = self.total_inter.to(inter.device)
                self.total_union = self.total_union.to(union.device)
            # 统计总的inter
            self.total_inter += inter
            # 统计总的union
            self.total_union += union
        # 根据preds的不同预测值数据类型，来进行计算
        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)
    # 获取结果组成的元组
    def get(self):
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)  # remove np.spacing(1)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean().item()
        return pixAcc, mIoU

    # 清零计算结果
    def reset(self):
        self.total_inter = torch.zeros(self.nclass)
        self.total_union = torch.zeros(self.nclass)
        self.total_correct = 0
        self.total_label = 0

# Pixel Acc计算
def batch_pix_accuracy(output, target):
    # 提取21类中准确度最高的一类，+1的目的是避免背景的像素0在计算中无法计算
    predict = torch.argmax(output.long(), 1) + 1
    # 相应的标签的像素值也依次+1
    target = target.long() + 1
    # 获取批次内图像的像素点总数
    pixel_labeled = torch.sum(target > 0).item()
    # 获取预测正确像素点的总数（“==” 用以获取像素点是否匹配）
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

# mIoU计算
def batch_intersection_union(output, target, nclass):
    # 类别的最小标签数字
    mini = 1
    # 类别的最大标签数字
    maxi = nclass
    #
    nbins = nclass
    # 获取预测中得分最高的一类矩阵
    predict = torch.argmax(output, 1) + 1 # dim = 1,每一行的最大列表（dim = 0.每一列的最大行标）
    # 对于二维矩阵，0表示按行找，1表示按列找
    # 获取目标的矩阵
    target = target.float() + 1
    # 将预测矩阵转float格式
    predict = predict.float() * (target > 0).float()
    # 判定有多少相符合的点，并转为0、1格式的矩阵
    intersection = predict * (predict == target).float()
    # 构建相符合点的分布直方图
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    # 构建预测结果点的分布直方图
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    # 构建标签点的分布直方图
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    # 计算iou值
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0
    return area_inter.float(), area_union.float()



def pixelAccuracy(imPred, imLab):
    pixel_labeled = np.sum(imLab >= 0)
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return (pixel_accuracy, pixel_correct, pixel_labeled)


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = imPred * (imLab >= 0)
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)

def hist_info(pred, label, num_cls):
    assert pred.shape == label.shape
    k = (label >= 0) & (label < num_cls)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == label[k]))
    return np.bincount(num_cls * label[k].astype(int) + pred[k], minlength=num_cls ** 2).reshape(num_cls,
                                                                                                 num_cls), labeled, correct

def compute_score(hist, correct, labeled):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled
    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc
import datetime
import os

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.FGCN import FCN8s
from tools.config import (ecssd_path)
from tools.datasets import ImageFolder
from tools.misc import (
    AvgMeter, cal_fmeasure, cal_p_r_mae_fm, check_mkdir, crf_refine)

# from models.FGCN import FCNs

torch.manual_seed(2019)
torch.cuda.set_device(0)

exp_name = 'FGCN'

ckpt_path_all = {
    'R3Net': './ckpt/ckpt_R3',
    'R2Net': './ckpt/ckpt_R2',
    'FGCN': './ckpt/ckpt_FGCN'
}

ckpt_path = ckpt_path_all[exp_name]

# duts: 30000次迭带
args = {
    'snapshot': '8000',
    'crf_refine': True,
}
data_path = ['ecssd', ecssd_path]

# 只用来处理生成图片, 具体的评估, 在全部图片生成完后进行计算.
# 因为在结构中使用了反卷积, 与卷积不一定可以完全对应相加, 所以这里使用强行调整大小来生成图片
img_transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

test_set = ImageFolder(root=data_path[1],
                       isTrain=False,
                       transform=img_transform)
test_loader = DataLoader(test_set,
                         batch_size=4,
                         num_workers=12)


# R2Net, after: 6000 msra10k
# test results:{'fm_thresholds': 0.9144239202514287, 'fm': 0.8578482803577097, 'mae': 0.07251937238625916}
# (resize 640, 640)
# test results:{'fm_thresholds': 0.9045901551676869, 'fm': 0.8450927408381543, 'mae': 0.07926012772671576}
# (resize 800, 800)
# test results:{'fm_thresholds': 0.8666383530633786, 'fm': 0.8306131125151587, 'mae': 0.12597991255241972}
# batch=1, noresize

# ResNet+FCN, after: 8000 msra10k
# test: ecssd: test results:{'fm_thresholds': 0.9175770664829266, 'fm': 0.8349719393636144, 'mae': 0.07209092804935234}
def main():
    net = FCN8s(pretrained=False).cuda()
    
    # 使用保存好的已经训练好的模型来进行测试, 具体使用哪个, 由参数'sanpshot'确定
    print('load snapshot \'%s\' for testing' % args['snapshot'])
    check_mkdir(os.path.join(
        ckpt_path, f"{exp_name}:{data_path[0]}_{args['snapshot']}"))
    
    net.load_state_dict(torch.load(
        os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
    net.eval()
    
    tqdm_loader = tqdm(test_loader)
    
    with torch.no_grad():
        # 迭代数据
        for i, data in enumerate(tqdm_loader):
            # print(f"开始迭代batch{i}/{len(tqdm_loader)}")
            inputs, img_names, hs, ws = data
        
            inputs = Variable(inputs).cuda()
            prediction = net(inputs)
            for i_item, img_name in enumerate(img_names):
                pre = prediction[i_item]
                pre = to_pil(pre.data.cpu())
            
                pre = pre.resize((hs[i_item], ws[i_item]), Image.ANTIALIAS)
                pre = np.array(pre)
            
                img_path = os.path.join(
                    ckpt_path, f"{exp_name}:{data_path[0]}_"
                    f"{args['snapshot']}", img_name + '.png')
                Image.fromarray(pre).save(img_path)
    
        print("测试集预测生成结束")


def cal_criterion():
    print("评估阶段开始")
    
    precision_record = [AvgMeter() for _ in range(256)]
    recall_record = [AvgMeter() for _ in range(256)]
    mae_record = AvgMeter()
    fm_record = AvgMeter()
    
    pre_path = os.path.join(
        ckpt_path, f"{exp_name}:{data_path[0]}_{args['snapshot']}")
    pre_list = os.listdir(pre_path)
    pre_names = {name.split('.')[0] for name in pre_list}
    
    if data_path[0] == 'ecssd':
        inp_path = os.path.join(data_path[1], 'Image')
        gt_path = os.path.join(data_path[1], 'Mask')
    
    results = {}
    tqdm_names = tqdm(pre_names)
    for i, pre_name in enumerate(tqdm_names):
        pre = Image.open(os.path.join(pre_path, pre_name + '.png')).convert('L')
        gt = Image.open(os.path.join(gt_path, pre_name + '.png')).convert('L')
        pre = np.array(pre)
        gt = np.array(gt)
        
        if args['crf_refine']:
            inp = Image.open(
                os.path.join(inp_path, pre_name + '.jpg')).convert('RGB')
            pre = crf_refine(np.array(inp), pre)
        
        try:
            # 此时pre和gt都是numpy数组
            precision, recall, mae, fm = cal_p_r_mae_fm(pre, gt)
        except AssertionError:
            print(f"出现异常, img:{pre_name}")
        
        # 对batch内的每张图片的256组数据进行累计
        for pidx, pdata in enumerate(zip(precision, recall)):
            p, r = pdata
            precision_record[pidx].update(p)
            recall_record[pidx].update(r)
        
        # 对batch内的每张图片的1组数据进行累计
        mae_record.update(mae)
        fm_record.update(fm)
    
    # 使用整个数据集上的数据进行平均
    fmeasure = cal_fmeasure(
        [precord.avg for precord in precision_record],
        [rrecord.avg for rrecord in recall_record])
    
    results = {'fm_thresholds': fmeasure,
               'fm': fm_record.avg,
               'mae': mae_record.avg}
    
    print(f'test results:{results}')


if __name__ == '__main__':
    start = datetime.datetime.now()
    
    main()
    middle = datetime.datetime.now() - start
    print(f"生成预测所需时间为:{middle}")
    
    cal_criterion()
    print(f"测量指标所需时间为:{datetime.datetime.now() - middle}")
    
    print(f"整体时间为:{datetime.datetime.now() - start}")

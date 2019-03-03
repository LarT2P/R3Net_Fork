"""
1. Set the path of five benchmark datasets in config.py
2. Put the trained model in ckpt/R3Net
3. Run by python infer.py
"""

import datetime
import os

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from models.model_R2 import R2Net
from tools.config import (ecssd_path)
from tools.misc import (
    AvgMeter, cal_fmeasure, cal_p_r_mae_fm, check_mkdir, crf_refine)

exp_name = 'R2Net'

torch.manual_seed(2019)
torch.cuda.set_device(0)
ckpt_path_all = {
    'R3Net': './ckpt/ckpt_R3',
    'R2Net': './ckpt/ckpt_R2',
    'FGCN': './ckpt/ckpt_FGCN'
}

ckpt_path = ckpt_path_all[exp_name]

args = {
    'snapshot': '6000',  # your snapshot filename (exclude extension name)
    'crf_refine': False,  # whether to use crf to refine results
    'save_results': False  # whether to save the resulting masks
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
to_pil = transforms.ToPILImage()

to_test = {
    # 'oppo': oppo_path
    'ecssd': ecssd_path
}


# after msra10k 6000
# {'ecssd': {'fm_thresholds': 0.8666383530633794, 'fm': 0.8306131125151581, 'mae': 0.12597991255241994}}
def main():
    net = R2Net().cuda()
    
    # 使用保存好的已经训练好的模型来进行测试, 具体使用哪个, 由参数'sanpshot'确定
    print('load snapshot \'%s\' for testing' % args['snapshot'])
    net.load_state_dict(torch.load(
        os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
    net.eval()
    
    results = {}
    with torch.no_grad():
        
        # 一次处理一个数据集
        for name, root in to_test.items():
            # 因为有256(个阈值)组数据
            precision_record, recall_record, = \
                [AvgMeter() for _ in range(256)], \
                [AvgMeter() for _ in range(256)]
            mae_record = AvgMeter()
            fm_record = AvgMeter()
            
            if name == 'duts' or name == 'oppo':
                img_path = os.path.join(root, 'Test', 'Image')
                gt_path = os.path.join(root, 'Test', 'Mask')
            else:
                img_path = os.path.join(root, 'Image')
                gt_path = os.path.join(root, 'Mask')
            
            if args['save_results']:
                check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s'
                                         % (exp_name, name, args['snapshot'])))
            
            img_list = [os.path.splitext(f)[0] for f in os.listdir(img_path)
                        if f.endswith('.jpg')]
            
            # 一次处理一张图 #####################################################
            for idx, img_name in enumerate(img_list):
                print('predicting for %s: %d/%d, img: %s/%s' %
                      (name, idx + 1, len(img_list), img_name, name))
                
                img = Image.open(
                    os.path.join(img_path, img_name + '.jpg')).convert('RGB')
                # 给输入的图片数据插入维度, 来匹配网络
                img_var = Variable(
                    # img_transform(img).unsqueeze(0), volatile=True).cuda()
                    # pytorch 1.0
                    img_transform(img).unsqueeze(0)).cuda()
                prediction = net(img_var)
                # 对输出的数据去除多余的维度
                prediction = np.array(to_pil(prediction.data.squeeze(0).cpu()))
                print(f'{prediction.dtype}')
                
                gt = np.array(Image.open(
                    os.path.join(gt_path, img_name + '.png')).convert('L'))
                
                if args['crf_refine']:
                    prediction = crf_refine(np.array(img), prediction)
                
                try:
                    # 此时prediction和gt都是numpy数组
                    precision, recall, mae, fm = cal_p_r_mae_fm(
                        prediction, gt)
                    # print(mae, fm)
                except AssertionError:
                    print("存在真值与原图大小不一致的情况, "
                          "img:{0}/{1}".format(img_name, name))
                
                # 对所有图片进行累计, 以获得整个数据集的均值
                for pidx, pdata in enumerate(zip(precision, recall)):
                    p, r = pdata
                    precision_record[pidx].update(p)
                    recall_record[pidx].update(r)
                mae_record.update(mae)
                fm_record.update(fm)
                
                if args['save_results']:
                    Image.fromarray(prediction).save(
                        os.path.join(ckpt_path, exp_name, '(%s) %s_%s' %
                                     (exp_name, name, args['snapshot']),
                                     img_name + '.png'))
            
            # 使用一个数据集上的数据
            fmeasure = cal_fmeasure(
                [precord.avg for precord in precision_record],
                [rrecord.avg for rrecord in recall_record])
            
            results[name] = {'fm_thresholds': fmeasure,
                             'fm': fm_record.avg,
                             'mae': mae_record.avg}
    
    print(f'test results:{results}')


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print(f"整体时间为:{datetime.datetime.now() - start}")

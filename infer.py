"""
1. Set the path of five benchmark datasets in config.py
2. Put the trained model in ckpt/R3Net
3. Run by python infer.py
"""

import numpy as np
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import (
    ecssd_path, hkuis_path, pascals_path, sod_path, dutomron_path,
    duts_path, oppo_path)
from misc import (
    check_mkdir, crf_refine, AvgMeter, cal_precision_recall_mae_fm,
    cal_fmeasure)
from model import R3Net

torch.manual_seed(2018)

# set which gpu to use
torch.cuda.set_device(0)

# the following two args specify the location of the file of trained model (pth extension)
# you should have the pth file in the folder './$ckpt_path$/$exp_name$'
ckpt_path = './ckpt'
exp_name = 'R3Net'

args = {
    'snapshot': '6000',  # your snapshot filename (exclude extension name)
    'crf_refine': True,  # whether to use crf to refine results
    'save_results': True  # whether to save the resulting masks
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
to_pil = transforms.ToPILImage()

to_test = {
    'oppo': oppo_path
    # 'ecssd': ecssd_path
}


# after: 6000
# train: msta10k
# {'ecssd': {'fmeasure': 0.902749575108475, 'mae': 0.05742777294518834},
# 'duts': {'fmeasure': 0.8087214904540923, 'mae': 0.06695453740073982},
# 'oppo': {'fmeasure': 0.8017284179533116, 'mae': 0.34996716117358695}}
# after: 6000
# train: oppo: 1500, test: oppo: 200
# {'oppo': {'fmeasure': 0.9847538558012283, 'mae': 0.02405147205882353}}
# test: oppo: 1500
# {'oppo': {'fmeasure': 0.9903522147959504, 'mae': 0.017322539520697164}}
def main():
    net = R3Net().cuda()
    
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
                
                gt = np.array(Image.open(
                    os.path.join(gt_path, img_name + '.png')).convert('L'))
                
                if args['crf_refine']:
                    prediction = crf_refine(np.array(img), prediction)
                
                try:
                    # 此时prediction和gt都是numpy数组
                    precision, recall, mae, fm = cal_precision_recall_mae_fm(
                        prediction, gt)
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
    
    print('test results:')
    print(results)


if __name__ == '__main__':
    main()

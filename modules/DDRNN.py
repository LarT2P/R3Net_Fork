import torch.nn as nn
import torch
import numpy as np


class RNN(nn.Module):
    """
    The dimension of input, hidden and output units for D-RNNs is set to
    512. The two non-linear activations are ReLU and softmax functions,
    respectively

    usage:
        rnn = RNN(10, 10, 10).cuda()
        x = torch.randn(1, 10, 14, 14).cuda()
        y = rnn(x)
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        # nn.Linear可以保留batch维度, 只要另一个维度满足输入特征数量就可以
        self.in_linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, output_size)
        
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        
        # todo: 这里可以添加一个对于线性层的初始化
    
    def forward(self, input):
        """
        计算 `h^l_{v_ij}`, 对前面所有的父节点的关联隐含状态加权求和

        :param x: 整体的输入数据 (batch, num_feature, map_H, map_W)
        :return: y 当前研究的元素的隐藏状态
        """
        batch, _, H, W = input.size()
        h = torch.zeros(batch, self.hidden_size, 4, H, W).cuda()
        y = torch.zeros(batch, self.output_size, 4, H, W).cuda()
        
        modes = ['right_bottom', 'left_bottom', 'right_top', 'left_top']
        
        # 遍历整个x的位置
        for mode, name in enumerate(modes):
            # 遍历四个方向, 右下, 左下, 右上, 左上, 最简单的方式是直接调整x的数据方向
            if name == 'right_bottom':
                # 右下: 最正常
                x = input
            elif name == 'left_bottom':
                # 左下: 将x沿着H翻转, 也就是左右翻转,
                x = torch.flip(input, [3])
            elif name == 'right_top':
                # 右上: 沿着W轴翻转
                x = torch.flip(input, [2])
            elif name == 'left_top':
                # 左上: 沿着副对角线翻转, 也就是左右上下各翻一下
                x = torch.flip(input, [2])
                x = torch.flip(x, [3])
            
            # todo: 耗内存的重灾区域
            for place_x in range(H):
                for place_y in range(W):
                    # 仅考虑计算y_vi/h_vi的流程, 也就是坐标(i,j)对应的点的h的计算
                    # 仅计算左上方向([0:place_x+1), [0:place_y+1))矩形范围内的数据
                    cac_x = x[:, :, place_x, place_y]
                    # 每一次迭代都会更新
                    cac_h = h[:, :, mode, :place_x + 1, :place_y + 1]
                    
                    # 将x和h组合到一起, 一起进行计算
                    # h_vivj (batch, hidden_size, place_x, place_y)
                    h_vivj = torch.zeros(
                        batch, self.hidden_size,
                        place_x + 1, place_y + 1).cuda()
                    for i in range(place_x + 1):
                        for j in range(place_y + 1):
                            # 特征数量上的拼接
                            h_x_cat = torch.cat((cac_x, cac_h[:, :, i, j]), 1)
                            h_vivj[:, :, i, j] = self.relu(
                                self.in_linear(h_x_cat))
                            del h_x_cat
                    
                    del cac_x, cac_h
                    
                    # 先书写for-loop形式的父节点的计算
                    # w_vivj (batch, hidden_size, place_x, place_y)
                    #  w_vivj = torch.zeros(
                    #      batch, self.hidden_size, place_x + 1,
                    #                               place_y + 1).cuda()
                    # for k in range(place_x + 1):
                    #     for l in range(place_y + 1):
                    #         w_vivj[:, :, k, l] = self.softmax(h_vivj[:, :, k, l])
                    w_vivj = self.softmax(h_vivj[:, :, :, :])
                    
                    # 对来自不同父节点的加权乘积, 求和
                    # h_vi (batch, hidden_size)
                    h_vi = torch.sum(torch.mul(h_vivj, w_vivj), dim=2)
                    h_vi = torch.sum(h_vi, dim=2)
                    
                    del h_vivj, w_vivj
                    
                    # 完成了对于vi点处的计算, 也就是对与(i,j)处的计算, 收集起来
                    h[:, :, mode, i, j] = h_vi
                    
                    # y_vi (batch, out_size)
                    y_vi = self.out_linear(h_vi)
                    y[:, :, mode, i, j] = y_vi
                    
                    del y_vi
            
            del x
        
        y = self.softmax(torch.sum(y, dim=2))
        del h
        
        return y


# 模型显存占用监测函数
# model：输入的模型
# input：实际中需要输入的Tensor变量
# type_size 默认为 4 默认类型为 float32
# 这个代码有点小瑕疵, 默认是认为里面的所有的层都是输入相同的大小
def modelsize(model, input, type_size=4):
    # 从模型参数上来计算占用
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(
        model._get_name(), para * type_size / 1000 / 1000))
    
    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)
    
    mods = list(model.modules())
    out_sizes = []
    # for i in range(1, len(mods)):
    #     m = mods[i]
    #     if isinstance(m, nn.ReLU):
    #         if m.inplace:
    #             continue
    #     try:
    #         out = m(input_)
    #         out_sizes.append(np.array(out.size()))
    #         input_ = out
    #     except RuntimeError:
    #         print("当前计算的模块:", m)
    #         print("当前的数据", input_.size())
    # #
    m = mods[0]
    out = m(input_)
    out_sizes.append(np.array(out.size()))
    input_ = out
    
    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums
    
    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))


def main():
    rnn_0 = RNN(1, 3, 1).cuda()
    rnn_1 = RNN(1, 3, 1).cuda()
    rnn_2 = RNN(1, 3, 1).cuda()
    rnn_3 = RNN(1, 3, 1).cuda()
    rnn_4 = RNN(1, 3, 1).cuda()
    rnn_5 = RNN(1, 3, 1).cuda()
    rnn_6 = RNN(1, 3, 1).cuda()
    # rnn = RNN(1, 256, 1).cuda()
    x = torch.randn(2, 1, 20, 20).cuda()
    
    x = rnn_0(x)
    del rnn_0
    x = rnn_1(x)
    del rnn_1
    x = rnn_2(x)
    del rnn_2
    x = rnn_3(x)
    del rnn_3
    x = rnn_4(x)
    del rnn_4
    x = rnn_5(x)
    del rnn_5
    x = rnn_6(x)
    del rnn_6
    print(x.size())


if __name__ == '__main__':
    main()

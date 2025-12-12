import os
os.environ['DDE_BACKEND'] = 'pytorch'  # 设置 DeepXDE 使用 PyTorch 后端
print(os.environ['DDE_BACKEND'])

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定 GPU 设备

import numpy as np
import torch
import matplotlib.pyplot as plt
import random

# PyTorch 相关层
import torch.nn as nn

import deepxde as dde
from deepxde.data.data import Data
from deepxde.data.sampler import BatchSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from siren_pytorch import SinusodialRepresentationDense
from matplotlib.colors import LinearSegmentedColormap
import pyvista as pv
import imageio.v2 as imageio
import joblib
import torch.nn.functional as F

#拉丁超立方采样
from scipy.spatial.distance import cdist

print(dde.__version__)
dde.config.disable_xla_jit()  # XLA JIT 对 PyTorch 可能不适用，保留以防万一

# 查看是否可用 GPU
print("Num GPUs Available: ", torch.cuda.device_count())

class DeepONetCartesianProd(nn.Module):
    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation=None,
        kernel_initializer=None,
        regularization=None,
    ):
        super().__init__()

        self.regularizer = None

        #self.regularizer = ["l2", 1e-4]	L2 正则化（权重衰减），防止模型过拟合,模型参数更新时的正则化项


        # Branch networks
        self.geoNet1 = layer_sizes_branch[0]  # branch 前部分
        self.geoNet2 = layer_sizes_branch[1]  # branch 后部分

        # Trunk networks
        self.outNet1 = layer_sizes_trunk[0]  # trunk 前部分
        self.outNet2 = layer_sizes_trunk[1]  # trunk 后部分

        # Bias
        self.b = nn.Parameter(torch.zeros(1))  # 可学习偏置项

        # Output transform (可选)
        self._output_transform = None

    #############高斯核点积################
    def forward(self, inputs, training=False):
        x_func, x_loc = inputs  # x_func: [B, 4], x_loc: [B, N, 3]

        # Encode implicit geom
        x_func2 = self.geoNet1(x_func)  #[B, H]

        x_loc2 = self.outNet1(x_loc)    # [B, N, H]

        sigma = 0.4                            # 超参数，可调
        N = x_loc2.size(1)  # N: Number of nodes
        x_func2_expanded = x_func2.unsqueeze(1).expand(-1, N, -1)  # (B, N, H)
        diff = x_func2_expanded - x_loc2  # (B, N, h)
        K = torch.exp(-torch.sum(diff**2, dim=-1, keepdim=True)/(2*sigma**2))  # (B,N,1)

        # - Branch side: 使用平均相似性或最大相似性作为全局上下文
        global_sim = K.mean(dim=1)  # (B, 1)
        x_func3 = torch.cat([x_func2, global_sim], dim=-1)  # (B, H+1)

        # - Trunk side: 使用相似性作为局部上下文
        x_loc3 = torch.cat([x_loc2, K], dim=-1)             # (B,N,H+1)

        x_func4 = self.geoNet2(x_func3)  #[B, H]

        x_loc4 = self.outNet2(x_loc3)  # [B, N, H]

        x_loc4 = x_loc4.unsqueeze(-1)  # [B, N, H, 1]

        # Element-wise product
        x = torch.einsum("bh,bnhc->bnc", x_func4, x_loc4) # [B, N, H]

        # Add bias
        x += self.b

        # Optional output transform
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)

        # Final activation
        return torch.sigmoid(x)
        # return x
    

    # ##############爱因斯坦点积################
    # def forward(self, inputs, training=False):
    #     x_func, x_loc = inputs  # x_func: [B, 4], x_loc: [B, N, 3]

    #     # Encode implicit geom
    #     x_func2 = self.geoNet1(x_func)  #[B, H]

    #     x_loc2 = self.outNet1(x_loc)    # [B, N, H]

    #     # Element-wise product
    #     mix1 = torch.einsum("bh,bnh->bnh", x_func2, x_loc2) # [B, N, H]

    #     x_func3 = mix1.mean(dim=1)  # [B, H]

    #     x_func4 = self.geoNet2(x_func3)  #[B, H]

    #     x_loc3 = self.outNet2(mix1)  # [B, N, H]

    #     x_loc3 = x_loc3.unsqueeze(-1)  # [B, N, H, 1]

    #     # Element-wise product
    #     x = torch.einsum("bh,bnhc->bnc", x_func4, x_loc3) # [B, N, H]

    #     # Add bias
    #     x += self.b

    #     # Optional output transform
    #     if self._output_transform is not None:
    #         x = self._output_transform(inputs, x)

    #     # Final activation
    #     return torch.sigmoid(x)
    
class Test_BatchSampler:
    def __init__(self, num_samples, shuffle=False):
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(self.indices)
        self.index = 0

    def get_next(self, batch_size):
        if self.index >= self.num_samples:
            return None  # 返回 None 表示没有更多数据
        end = min(self.index + batch_size, self.num_samples)
        batch_indices = self.indices[self.index:end]
        self.index = end
        return batch_indices
    
    def reset(self):
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
    

    
class TripleCartesianProd(Data):
    """Dataset with each data point as a triple. The ordered pair of the first two
    elements are created from a Cartesian product of the first two lists. If we compute
    the Cartesian product of the first two arrays, then we have a ``Triple`` dataset.

    This dataset can be used with the network ``DeepONetCartesianProd`` for operator
    learning.

    Args:
        X_train: A tuple of two NumPy arrays. The first element has the shape (`N1`,
            `dim1`), and the second element has the shape (`N2`, `dim2`).
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        # 分别为 branch 和 trunk 创建采样器
        self.branch_sampler = BatchSampler(len(X_train[0]), shuffle=True)
        self.trunk_sampler = BatchSampler(len(X_train[1]), shuffle=True)

        # 测试集采样器（测试时不打乱）
        self.test_branch_sampler = Test_BatchSampler(len(X_test[0]), shuffle=False)
        self.test_trunk_sampler = Test_BatchSampler(len(X_test[1]), shuffle=False)

    def losses(self, targets, outputs, loss_fn, inputs=None, model=None, aux=None):
        return loss_fn(targets, outputs)
    
    # def losses(self, targets, outputs, loss_fn, inputs=None, model=None, aux=None):
    #     # 先计算原始 loss per point
    #     raw_loss = loss_fn(outputs, targets, reduction='none')  # (N1, N2)

    #     with torch.no_grad():
    #         weight = torch.abs(targets) + 0.1
    #         weight = 0.1 + 9.9 * (weight - weight.min()) / (weight.max() - weight.min() + 1e-8)

    #     return torch.mean(weight * raw_loss)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            # 返回全部训练数据（保持 NumPy 格式）
            return self.train_x, self.train_y

        if not isinstance(batch_size, (tuple, list)):
            # 单一 batch size，取相同的索引
            indices = self.branch_sampler.get_next(batch_size)
            if indices is None:
                return None, None


            return (
                self.train_x[0][indices],
                self.train_x[1][indices],
            ), self.train_y[indices]

        # 双 batch size，分别取不同索引
        indices_branch = self.branch_sampler.get_next(batch_size[0])
        indices_trunk = self.trunk_sampler.get_next(batch_size[1])
        if indices_branch is None or indices_trunk is None:
            return None, None


        return (
            self.train_x[0][indices_branch],
            self.train_x[1][indices_trunk],
        ), self.train_y[indices_branch, indices_trunk]
    

    def test_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.test_x, self.test_y

        if not isinstance(batch_size, (tuple, list)):
            indices = self.test_branch_sampler.get_next(batch_size)
            if indices is None or len(indices) == 0:
                return None, None
            return (
                self.test_x[0][indices],
                self.test_x[1][indices],
            ), self.test_y[indices]

        indices_branch = self.test_branch_sampler.get_next(batch_size[0])
        indices_trunk = self.test_trunk_sampler.get_next(batch_size[1])

        if indices_branch is None or len(indices_branch) == 0 or indices_trunk is None or len(indices_trunk) == 0:
            return None, None
    
        return (
            self.test_x[0][indices_branch],
            self.test_x[1][indices_trunk],
        ), self.test_y[indices_branch, indices_trunk]
    

    def test(self):
        # 返回测试数据（保持 NumPy 格式）
        return self.test_x, self.test_y

def normalize_and_translate(points):   #对象都被归一化到一个近似 [-0.5, 0.5] × [-0.5, 0.5] × [0, 1] 的长方体盒子内
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    bbox_size = max_coords - min_coords
    scale_factor = 1.0 / np.max(bbox_size)

    normalized = (points - min_coords) * scale_factor

    # Z=0 对齐（接地）
    min_z = np.min(normalized[:, 2])
    normalized[:, 2] -= min_z

    # 底面中心移到原点
    bottom_mask = normalized[:, 2] <= (min_z + 1e-6)  # 加容差避免浮点误差
    bottom_center_x = np.mean(normalized[bottom_mask, 0])
    bottom_center_y = np.mean(normalized[bottom_mask, 1])

    normalized[:, 0] -= bottom_center_x
    normalized[:, 1] -= bottom_center_y

    return normalized, min_coords, scale_factor  # 返回参数用于反归一化



def inverse_normalize_coord(normalized_points, min_coords, scale_factor):
    """
    反归一化：恢复原始尺寸，但保持底面中心在原点
    """
    return normalized_points / scale_factor + min_coords


# 设置随机种子（PyTorch + NumPy + Python）
seed = 2024
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Parameters
N_Geom = 9              # Number of parameters (x1,x2,x3,x4,x5,x6,x7,t,F)
N_comp = 1              # 输出矢量分量的数量
HIDDEN = 32             # 隐藏层神经元数

num_dim = 3             # 节点坐标维数
batch_size = 16

fraction_train = 0.8    # 训练集比例
N_epoch = 50000         # 训练迭代次数
data_type = np.float32   

learning_rate = 2e-3
# learning_rate = 2e-4

w0 = 10.                # 基频参数（如在 SIREN 中使用）
# act_layer = nn.ReLU()
# act_layer = nn.SiLU()
act_layer = nn.GELU()


field = 'Dis'     # 可选值: 'Dis', 'Stress'

print('\n\nModel parameters:')

print( 'N_comp  ' , N_comp )
print( 'HIDDEN  ' , HIDDEN )
print( 'batch_size  ' , batch_size )
print( 'fraction_train  ' , fraction_train )
print( 'learning_rate  ' , learning_rate )
print( 'w0  ' , w0 )
print( 'activation  ' , act_layer )
print('\n\n\n')


# 参数设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# trunk 前部分: 输入 shape: (b, N_Node, 3)
outNet1 = nn.Sequential(
    nn.Linear(num_dim, 50),
    act_layer,

    nn.Linear(50, 50),
    act_layer,

    nn.Linear(50, HIDDEN),
    act_layer,
).to(device)

print('\n\noutNet1:')
print(outNet1)

# trunk 后部分: 输入 shape: (b, N_Node, HIDDEN)
# outNet2 = nn.Sequential(
#     nn.Linear(HIDDEN* 2 +1, HIDDEN * 4),
#     act_layer,

#     nn.Linear(HIDDEN * 4, HIDDEN * 4),
#     act_layer,

#     nn.Linear(HIDDEN * 4, HIDDEN * 2),
#     act_layer,

#     nn.Linear(HIDDEN * 2, HIDDEN * 1),
#     act_layer,
# ).to(device)

outNet2 = nn.Sequential(
    SinusodialRepresentationDense(HIDDEN +1, HIDDEN * 2, w0=w0, activation='sine'),

    SinusodialRepresentationDense(HIDDEN * 2, HIDDEN * 4, w0=w0, activation='sine'),

    SinusodialRepresentationDense(HIDDEN * 4, HIDDEN * 2, w0=w0, activation='sine'),

    SinusodialRepresentationDense(HIDDEN * 2, HIDDEN * 1, w0=w0, activation='sine'),
).to(device)

print('\n\noutNet2:')
print(outNet2)

# branch 前部分: 输入 shape: (b, N_Geom)
geoNet1 = nn.Sequential(
    nn.Linear(N_Geom, 50),
    act_layer,

    nn.Linear(50, 50),
    act_layer,

    nn.Linear(50, HIDDEN),
    act_layer,
).to(device)

print('\n\ngeoNet1:')
print(geoNet1)


# branch 后部分: 输入 shape: (b, HIDDEN)
geoNet2 = nn.Sequential(
    nn.Linear(HIDDEN +1, HIDDEN * 2),
    act_layer,

    nn.Linear(HIDDEN * 2, HIDDEN * 4),
    act_layer,

    nn.Linear(HIDDEN * 4, HIDDEN * 2),
    act_layer,

    nn.Linear(HIDDEN * 2, HIDDEN * 1),
    act_layer,
).to(device)

print('\n\ngeoNet2:')
print(geoNet2)



# ----------------------------
# ✅ DataParallel 包装
# ----------------------------
class DataParallelWithAttr(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


base_net = DeepONetCartesianProd(
    layer_sizes_branch=[geoNet1, geoNet2],
    layer_sizes_trunk=[outNet1, outNet2],
    activation=act_layer,
).to(device)

net = DataParallelWithAttr(base_net)

print("\nModel initialized (DataParallel):")
print(net)


# # 初始化 DeepONetCartesianProd 模型
# net = DeepONetCartesianProd(
#     layer_sizes_branch=[geoNet1, geoNet2],
#     layer_sizes_trunk=[outNet1, outNet2],
#     activation=act_layer,
# ).to(device)

# print("\n" + "="*50)
# print("Model Architecture:")
# print(net)
# print("="*50)



Data_path = '/home/zhangchi/data/SGMW/F410S_100/Data/'

Data_morph_path = Data_path + 'Morph/' 

Data_ratio_path = Data_path + 'Ratio/'

Data_train_morph_ratio = Data_path + 'Kernel_Train_Morph_Ratio/'

round = 'r5_2'
# Resampledata_path = Data_path + field_name + '_' + 'N' +str(num_node)+'_k' + '.npz'

# ratio_list = [0.05, 0.1, 0.2, 0.25, 0.5]
ratio_list = [0.5]

for ratio in ratio_list:
    nodes_ratio = ratio
    print(f"\n\nnodes_ratio: {nodes_ratio}")

    # for morph in ['25', '50', '75', '100']:
    for morph in ['100']:
        print(f"\n\nProcessing morph: {morph}")

        Resampledata_path = Data_morph_path + 'Time_Coord_' + field + '_m' + morph + '.npz'

        if not os.path.exists(Resampledata_path):
            print(f"⚠️ 路径不存在: {Resampledata_path}")
            continue

        print(f"🔍 加载数据: {Resampledata_path}")

        tmp = np.load(Resampledata_path)
        Coords_flt = tmp['pos']      #缩放前
        Field_flt = tmp['dis']       #缩放前
        # Field_flt = tmp['s']       #缩放前


    ################################下采样################################
        # 动态获取节点数
        n_nodes = Coords_flt.shape[1]
        print(f"原始节点数: {n_nodes}")

        index_file = Data_ratio_path + field + f'_node_k_indices_ratio{str(nodes_ratio)}.npy'

        idx_selected = np.load(index_file)

        
    ######################节点下采样########################
        Coords_flt = Coords_flt[:, idx_selected, :]  # (N_sample, ~35k, 3)
        Field_flt = Field_flt[:, idx_selected, :]

        # 动态获取节点数
        n_nodes = Coords_flt.shape[1]
        print(f"下采样后节点数: {n_nodes}")


        # 存储归一化后的坐标和参数
        Coords_normalized = np.zeros_like(Coords_flt)  # 归一化后坐标
        norm_params = []  # 存储每个样本的 min_coords, scale_factor

        for i in range(Coords_flt.shape[0]):
            pts = Coords_flt[i]  # 当前样本，shape: (139372, 3)
            
            normalized, min_coords, scale_factor = normalize_and_translate(pts)
            
            Coords_normalized[i] = normalized
            norm_params.append({
                'min_coords': min_coords,
                'scale_factor': scale_factor
            })

        print("✅ 归一化完成！")
        print("Coords_normalized.shape:", Coords_normalized.shape)  


        output_path = Data_train_morph_ratio + field + '_m' + morph + '_ratio' + str(nodes_ratio) + '_k/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)


        Geom_flt = np.load(Data_morph_path+'Geom_Time_Load_m' + morph + '.npy', allow_pickle=True) # Geom params
        Geom_flt = Geom_flt.astype(float)


        sub = field + '_e' +str(N_epoch) + '_m' + morph + '_n' + str(n_nodes) + '_' + round
        print('sub = ', sub )

        # Scale
        scaler_fun = MinMaxScaler

        Coords_scal = Coords_normalized

        Geom_scalers = scaler_fun()
        Geom_scalers.fit(Geom_flt)
        Geom_scal = Geom_scalers.transform( Geom_flt )

        Field_scalers = scaler_fun()
        ss = Field_flt.shape
        tmp = Field_flt.reshape([ss[0] * ss[1], ss[2]])
        Field_scalers.fit(tmp)
        Field_scal = Field_scalers.transform(tmp).reshape(ss)    #缩放后

        joblib.dump(Geom_scalers, output_path+ field + '_m' + morph + '_n' + str(n_nodes) + '_scaler_Geom.pkl')
        joblib.dump(Field_scalers, output_path+ field + '_m' + morph + '_n' + str(n_nodes) + '_scaler_Field.pkl')

        # -------------------------------
        # 3. 划分训练集和测试集
        # -------------------------------
        num_sample = Coords_flt.shape[0]
        N_train = int(num_sample * fraction_train)
        train_case = np.random.choice(num_sample, N_train, replace=False)
        test_case = np.setdiff1d(np.arange(num_sample), train_case)

        # 训练集
        Coords_train = Coords_scal[train_case, ::].astype(data_type)  # shape: (N_train, num_node, 3)
        Coords_testing = Coords_scal[test_case, ::].astype(data_type)  # shape: (N_test, num_node, 3)

        Geom_train = Geom_scal[train_case, :].astype(data_type) # shape: (N_train, 4)
        Geom_testing = Geom_scal[test_case, :].astype(data_type) # shape: (N_test, 4)

        Field_train = Field_scal[train_case, ::].astype(data_type)  # shape: (N_train, num_node, 1)
        Field_testing = Field_scal[test_case, ::].astype(data_type)  # shape: (N_test, num_node, 1)

        # 打印信息
        print('Coords_train.shape = ', Coords_train.shape)
        print('Coords_testing.shape = ', Coords_testing.shape)
        print('Geom_train.shape = ', Geom_train.shape)
        print('Geom_testing.shape = ', Geom_testing.shape)
        print('Field_train.shape = ', Field_train.shape)
        print('Field_testing.shape = ', Field_testing.shape)

        x_train = (Geom_train.astype(data_type), Coords_train.astype(data_type))
        y_train = Field_train.astype(data_type)
        x_test = (Geom_testing.astype(data_type), Coords_testing.astype(data_type))
        y_test = Field_testing.astype(data_type)
        data = TripleCartesianProd(x_train, y_train, x_test, y_test)

        # Build model
        model = dde.B_Model(data, net)

        def inv( data , scaler ):
            ss = data.shape
            tmp = data.reshape([ ss[0]*ss[1] , ss[2] ])
            return scaler.inverse_transform( tmp ).reshape(ss)


        def err_L2( true_vals , pred_vals ):
            return np.linalg.norm(true_vals - pred_vals , axis=1 ) / np.linalg.norm( true_vals , axis=1 )

        def err_MAE( true_vals , pred_vals ):
            return np.mean( np.abs(true_vals - pred_vals) , axis=1 )

        def u_L2( y_train , y_pred ):   #垂直位移（u分量）的L2误差（也称为均方误差）
            true_vals = inv( y_train , Field_scalers )[:,:,0]
            pred_vals = inv( y_pred , Field_scalers )[:,:,0]
            return np.mean( err_L2( true_vals , pred_vals ) )

        def u_MAE( y_train , y_pred ):  #垂直位移（u分量）的平均绝对误差（MAE）
            true_vals = inv( y_train , Field_scalers )[:,:,0]
            pred_vals = inv( y_pred , Field_scalers )[:,:,0]
            return np.mean( err_MAE( true_vals , pred_vals ) )


        def vm_L2( y_train , y_pred ):  #水平位移（v分量）的L2误差
            true_vals = inv( y_train , Field_scalers )[:,:,1]
            pred_vals = inv( y_pred , Field_scalers )[:,:,1]
            return np.mean( err_L2( true_vals , pred_vals ) )

        def vm_MAE( y_train , y_pred ):   #水平位移（v分量）的平均绝对误差（MAE）
            true_vals = inv( y_train , Field_scalers )[:,:,1]
            pred_vals = inv( y_pred , Field_scalers )[:,:,1]
            return np.mean( err_MAE( true_vals , pred_vals ) )


        if N_comp == 1:
            metrics = [ u_L2 , u_MAE ]
        else:
            metrics = [ u_L2 , u_MAE , vm_L2 , vm_MAE ]

        model.compile(
            "adam",
            lr=learning_rate,          #优化器和学习率设置，控制梯度下降的速度	控制模型参数更新的步长
            loss=F.mse_loss,  # ← 必须这样设置
            decay=("inverse time", 1, learning_rate/10.),
            metrics=metrics,
        )

        loss_path = output_path

        losshistory, train_state = model.train(iterations=N_epoch, batch_size=batch_size, model_save_path= loss_path + sub)

        loss_name = 'losshistory' + sub + '.npy'

        np.save(loss_path + loss_name,losshistory)

        losshistory_name = 'losshistory' + sub + '.png'
        dde.utils.plot_loss_history(losshistory)                                #绘制损失历史的函数
        plt.savefig(loss_path + losshistory_name, dpi=300)  #保存图像
        plt.close()  #关闭图像
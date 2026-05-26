import os
os.environ['DDE_BACKEND'] = 'pytorch'  # Set DeepXDE to use PyTorch backend
print(os.environ['DDE_BACKEND'])
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU device

import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import deepxde as dde
from deepxde.data.data import Data
from deepxde.data.sampler import BatchSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from matplotlib.colors import LinearSegmentedColormap
import pyvista as pv
import imageio.v2 as imageio
import joblib
import torch.nn.functional as F
import math

print(dde.__version__)
dde.config.disable_xla_jit()  

# Check if GPU is available
print("Num GPUs Available: ", torch.cuda.device_count())





class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SinusodialRepresentationDense(nn.Module):
    def __init__(self, in_features, out_features, w0=30.0, c=6.0, use_bias=True, activation='sine'):
        """
        SIREN Layer: Implicit Neural Representations with Periodic Activation Functions.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            w0: Sine activation frequency factor
            c: Weight initialization scaling factor
            use_bias: Whether to use bias
            activation: Activation function ('sine' or None)
        """
        super().__init__()
        self.w0 = float(w0)
        self.c = float(c)

        # Linear layer
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)

        # Initialize weights
        fan_in = in_features
        bound = math.sqrt(self.c / fan_in) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)

        # Initialize bias
        if use_bias:
            torch.nn.init.uniform_(self.linear.bias, -1e-3, 1e-3)

        # Activation function
        if activation == 'sine':
            self.activation = Sine(w0=self.w0)
        elif activation is None or activation == 'none':
            self.activation = None
        else:
            raise NotImplementedError(f"Activation {activation} not supported.")

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
    

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


        # Branch networks
        self.geoNet1 = layer_sizes_branch[0]  # branch front part
        self.geoNet2 = layer_sizes_branch[1]  # branch back part

        # Trunk networks
        self.outNet1 = layer_sizes_trunk[0]  # trunk front part
        self.outNet2 = layer_sizes_trunk[1]  # trunk back part

        # Bias
        self.b = nn.Parameter(torch.zeros(1))  # Learnable bias term

        self._output_transform = None
    

    ##############Einstein Dot Product##############
    def forward(self, inputs, training=False):
        x_func, x_loc = inputs  # x_func: [B, 4], x_loc: [B, N, 3]

        # Encode implicit geom
        x_func2 = self.geoNet1(x_func)  #[B, H]

        x_loc2 = self.outNet1(x_loc)    # [B, N, H]

        # Element-wise product
        mix1 = torch.einsum("bh,bnh->bnh", x_func2, x_loc2) # [B, N, H]

        x_func3 = mix1.mean(dim=1)  # [B, H]

        x_func4 = self.geoNet2(x_func3)  #[B, H]

        x_loc3 = self.outNet2(mix1)  # [B, N, H]

        x_loc3 = x_loc3.unsqueeze(-1)  # [B, N, H, 1]

        # Element-wise product
        x = torch.einsum("bh,bnhc->bnc", x_func4, x_loc3) # [B, N, H]

        # Add bias
        x += self.b

        # Optional output transform
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)

        return torch.sigmoid(x)
    
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
            return None  # Return None when no more data is available
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

        # Create samplers for branch and trunk respectively
        self.branch_sampler = BatchSampler(len(X_train[0]), shuffle=True)
        self.trunk_sampler = BatchSampler(len(X_train[1]), shuffle=True)

        # Test set samplers (no shuffle during testing)
        self.test_branch_sampler = Test_BatchSampler(len(X_test[0]), shuffle=False)
        self.test_trunk_sampler = Test_BatchSampler(len(X_test[1]), shuffle=False)

    def losses(self, targets, outputs, loss_fn, inputs=None, model=None, aux=None):
        return loss_fn(targets, outputs)
    

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            # Return all training data (kept in NumPy format)
            return self.train_x, self.train_y

        if not isinstance(batch_size, (tuple, list)):
            # Single batch size, use the same indices
            indices = self.branch_sampler.get_next(batch_size)
            if indices is None:
                return None, None


            return (
                self.train_x[0][indices],
                self.train_x[1][indices],
            ), self.train_y[indices]

        # Dual batch size, use different indices respectively
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
        # Return test data (kept in NumPy format)
        return self.test_x, self.test_y

def normalize_and_translate(points):   # Objects are normalized into a box approximately [-0.5, 0.5] × [-0.5, 0.5] × [0, 1]
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    bbox_size = max_coords - min_coords
    scale_factor = 1.0 / np.max(bbox_size)

    normalized = (points - min_coords) * scale_factor

    # Z=0 alignment (grounding)
    min_z = np.min(normalized[:, 2])
    normalized[:, 2] -= min_z

    # Move bottom center to origin
    bottom_mask = normalized[:, 2] <= (min_z + 1e-6)  # Add tolerance to avoid floating point errors
    bottom_center_x = np.mean(normalized[bottom_mask, 0])
    bottom_center_y = np.mean(normalized[bottom_mask, 1])

    normalized[:, 0] -= bottom_center_x
    normalized[:, 1] -= bottom_center_y

    return normalized, min_coords, scale_factor  # Return parameters for denormalization



def inverse_normalize_coord(normalized_points, min_coords, scale_factor):
    """
    Denormalize: Restore original size while keeping the bottom center at origin
    """
    return normalized_points / scale_factor + min_coords


# Set random seeds (PyTorch + NumPy + Python)
seed = 2024
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Parameters
N_Geom = 9              # Number of parameters (x1,x2,x3,x4,x5,x6,x7,t,F)
N_comp = 1              # Number of output vector components
HIDDEN = 32             # Number of hidden layer neurons

num_dim = 3             # Node coordinate dimension
batch_size = 16

fraction_train = 0.8    # Training set ratio
N_epoch = 50000         # Number of training iterations
data_type = np.float32   

learning_rate = 2e-3


w0 = 10.                # Base frequency parameter (used in SIREN)
act_layer = nn.GELU()


field = 'Dis'     # Optional values: 'Dis', 'Stress'

print('\n\nModel parameters:')

print( 'N_comp  ' , N_comp )
print( 'HIDDEN  ' , HIDDEN )
print( 'batch_size  ' , batch_size )
print( 'fraction_train  ' , fraction_train )
print( 'learning_rate  ' , learning_rate )
print( 'w0  ' , w0 )
print( 'activation  ' , act_layer )
print('\n\n\n')


# Parameter settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Trunk front part: input shape: (b, N_Node, 3)
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

outNet2 = nn.Sequential(
    SinusodialRepresentationDense(HIDDEN, HIDDEN * 2, w0=w0, activation='sine'),

    SinusodialRepresentationDense(HIDDEN * 2, HIDDEN * 4, w0=w0, activation='sine'),

    SinusodialRepresentationDense(HIDDEN * 4, HIDDEN * 2, w0=w0, activation='sine'),

    SinusodialRepresentationDense(HIDDEN * 2, HIDDEN * 1, w0=w0, activation='sine'),
).to(device)

print('\n\noutNet2:')
print(outNet2)

# Branch front part: input shape: (b, N_Geom)
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


# Branch back part: input shape: (b, HIDDEN)
geoNet2 = nn.Sequential(
    nn.Linear(HIDDEN, HIDDEN * 2),
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
# ✅ DataParallel wrapper
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



Data_path = '/home/zhangchi/data/SGMW/F410S_100/Data/'

Data_morph_path = Data_path + 'Morph/' 

Data_ratio_path = Data_path + 'Ratio/'

Data_train_morph_ratio = Data_path + 'Kernel_Train_Morph_Ratio/'

round = 'r6'                        # r6 is Einstein summation, r5 is Gaussian kernel

ratio_list = [0.05, 0.1, 0.2, 0.5]

for ratio in ratio_list:
    nodes_ratio = ratio
    print(f"\n\nnodes_ratio: {nodes_ratio}")

    for morph in ['100']:
        print(f"\n\nProcessing morph: {morph}")

        Resampledata_path = Data_morph_path + 'Time_Coord_' + field + '_m' + morph + '.npz'

        if not os.path.exists(Resampledata_path):
            print(f"⚠️ Path does not exist: {Resampledata_path}")
            continue

        print(f"🔍 Loading data: {Resampledata_path}")

        tmp = np.load(Resampledata_path)
        Coords_flt = tmp['pos']      # Before scaling
        Field_flt = tmp['dis']       # Before scaling


    ################################Downsampling################################
        # Dynamically get number of nodes
        n_nodes = Coords_flt.shape[1]
        print(f"Original number of nodes: {n_nodes}")

        index_file = Data_ratio_path + field + f'_node_k_indices_ratio{str(nodes_ratio)}.npy'

        idx_selected = np.load(index_file)

        
    ######################Node Downsampling########################
        Coords_flt = Coords_flt[:, idx_selected, :]  # (N_sample, ~35k, 3)
        Field_flt = Field_flt[:, idx_selected, :]

        # Dynamically get number of nodes
        n_nodes = Coords_flt.shape[1]
        print(f"Number of nodes after downsampling: {n_nodes}")


        # Store normalized coordinates and parameters
        Coords_normalized = np.zeros_like(Coords_flt)  # Normalized coordinates
        norm_params = []  # Store min_coords, scale_factor for each sample

        for i in range(Coords_flt.shape[0]):
            pts = Coords_flt[i]  # Current sample, shape: (139372, 3)
            
            normalized, min_coords, scale_factor = normalize_and_translate(pts)
            
            Coords_normalized[i] = normalized
            norm_params.append({
                'min_coords': min_coords,
                'scale_factor': scale_factor
            })

        print("✅ Normalization complete!")
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
        Field_scal = Field_scalers.transform(tmp).reshape(ss)    # After scaling

        joblib.dump(Geom_scalers, output_path+ field + '_m' + morph + '_n' + str(n_nodes) + '_scaler_Geom.pkl')
        joblib.dump(Field_scalers, output_path+ field + '_m' + morph + '_n' + str(n_nodes) + '_scaler_Field.pkl')

        # -------------------------------
        # 3. Split training and testing sets
        # -------------------------------
        num_sample = Coords_flt.shape[0]
        N_train = int(num_sample * fraction_train)
        train_case = np.random.choice(num_sample, N_train, replace=False)
        test_case = np.setdiff1d(np.arange(num_sample), train_case)

        # Training set
        Coords_train = Coords_scal[train_case, ::].astype(data_type)  # shape: (N_train, num_node, 3)
        Coords_testing = Coords_scal[test_case, ::].astype(data_type)  # shape: (N_test, num_node, 3)

        Geom_train = Geom_scal[train_case, :].astype(data_type) # shape: (N_train, 4)
        Geom_testing = Geom_scal[test_case, :].astype(data_type) # shape: (N_test, 4)

        Field_train = Field_scal[train_case, ::].astype(data_type)  # shape: (N_train, num_node, 1)
        Field_testing = Field_scal[test_case, ::].astype(data_type)  # shape: (N_test, num_node, 1)

        # Print information
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

        def u_L2( y_train , y_pred ):   # L2 error of vertical displacement (u component), also known as mean squared error
            true_vals = inv( y_train , Field_scalers )[:,:,0]
            pred_vals = inv( y_pred , Field_scalers )[:,:,0]
            return np.mean( err_L2( true_vals , pred_vals ) )

        def u_MAE( y_train , y_pred ):  # Mean absolute error (MAE) of vertical displacement (u component)
            true_vals = inv( y_train , Field_scalers )[:,:,0]
            pred_vals = inv( y_pred , Field_scalers )[:,:,0]
            return np.mean( err_MAE( true_vals , pred_vals ) )


        def vm_L2( y_train , y_pred ):  # L2 error of horizontal displacement (v component)
            true_vals = inv( y_train , Field_scalers )[:,:,1]
            pred_vals = inv( y_pred , Field_scalers )[:,:,1]
            return np.mean( err_L2( true_vals , pred_vals ) )

        def vm_MAE( y_train , y_pred ):   # Mean absolute error (MAE) of horizontal displacement (v component)
            true_vals = inv( y_train , Field_scalers )[:,:,1]
            pred_vals = inv( y_pred , Field_scalers )[:,:,1]
            return np.mean( err_MAE( true_vals , pred_vals ) )


        if N_comp == 1:
            metrics = [ u_L2 , u_MAE ]
        else:
            metrics = [ u_L2 , u_MAE , vm_L2 , vm_MAE ]

        model.compile(
            "adam",
            lr=learning_rate,          # Optimizer and learning rate settings, controls gradient descent speed, controls model parameter update step size
            loss=F.mse_loss,  # Must be set this way
            decay=("inverse time", 1, learning_rate/10.),
            metrics=metrics,
        )

        loss_path = output_path

        losshistory, train_state = model.train(iterations=N_epoch, batch_size=batch_size, model_save_path= loss_path + sub)

        loss_name = 'losshistory' + sub + '.npy'

        np.save(loss_path + loss_name,losshistory)

        losshistory_name = 'losshistory' + sub + '.png'
        dde.utils.plot_loss_history(losshistory)                                # Function to plot loss history
        plt.savefig(loss_path + losshistory_name, dpi=300)  # Save image
        plt.close()  # Close image
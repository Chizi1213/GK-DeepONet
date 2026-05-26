import os
os.environ['DDE_BACKEND'] = 'pytorch'  # Set DeepXDE to use PyTorch backend
print(os.environ['DDE_BACKEND'])

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
import math

# ======== Must be called as early as possible after imports! ========
pv.start_xvfb()
# =======================================


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

        #self.regularizer = ["l2", 1e-4]  L2 regularization (weight decay), prevents overfitting, regularization term during model parameter updates


        # Branch networks
        self.geoNet1 = layer_sizes_branch[0]  # branch front part
        self.geoNet2 = layer_sizes_branch[1]  # branch back part

        # Trunk networks
        self.outNet1 = layer_sizes_trunk[0]  # trunk front part
        self.outNet2 = layer_sizes_trunk[1]  # trunk back part

        # Bias
        self.b = nn.Parameter(torch.zeros(1))  # Learnable bias term

        # Output transform (optional)
        self._output_transform = None

    #############Einstein Dot Product##############
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

        # Final activation
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

def generate_gif_for_current_sample(Predict_field_path, field):
    import imageio
    import os

    images_ae, images_pred, images_true = [], [], []

    for j in range(18):
        folder_path = os.path.join(Predict_field_path, f'T{j}')
        if os.path.exists(folder_path):
            try:
                ae_img = imageio.imread(os.path.join(folder_path, f'{field}_Ae.png'))
                pred_img = imageio.imread(os.path.join(folder_path, f'{field}_Pred.png'))
                true_img = imageio.imread(os.path.join(folder_path, f'{field}_True.png'))

                images_ae.append(ae_img)
                images_pred.append(pred_img)
                images_true.append(true_img)
            except Exception as e:
                print(f"Warning: Cannot read images in {folder_path}: {e}")

    # Save GIF
    imageio.mimsave(
        # os.path.join(Predict_field_path, f'{field}_GK-DeepONet_Ae.gif'),
        os.path.join(Predict_field_path, f'{field}_Geom-DeepONet_Ae.gif'),
        images_ae,
        fps=2,
        loop=0  # ✅ Infinite loop
    )

    imageio.mimsave(
        # os.path.join(Predict_field_path, f'{field}_GK-DeepONet.gif'),
        os.path.join(Predict_field_path, f'{field}_Geom-DeepONet.gif'),
        images_pred,
        fps=2,
        loop=0  # ✅ Infinite loop
    )

    imageio.mimsave(
        os.path.join(Predict_field_path, f'{field}_FEM.gif'),
        images_true,
        fps=2,
        loop=0  # ✅ Infinite loop
    )

    print(f"GIF generated: {Predict_field_path}")


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

round = 'r6'                      # r6 is Einstein summation, r5 is Gaussian kernel

num_node = 139372   #dis


ratio_list = [0.05, 0.1, 0.2, 0.5]


for ratio in ratio_list:
    nodes_ratio = ratio
    print(f"\n\nnodes_ratio: {nodes_ratio}")

    for morph in ['100']:

        num_nodes = int(num_node * nodes_ratio)
        print('num_nodes = ', num_nodes)

        sub = field + '_e' +str(N_epoch) + '_m' + morph + '_n' + str(num_nodes) + '_' + round
        print('sub = ', sub )

        # Construct dummy data for building Model (real data not needed)
        geom_dummy = np.random.rand(10, N_Geom).astype(data_type)     # shape: (N, 9)
        coords_dummy = np.random.rand(10, num_nodes, num_dim).astype(data_type)  # shape: (N, XX, 3)
        field_dummy = np.random.rand(10, num_nodes, N_comp).astype(data_type)  # shape: (N, XX, 1)

        x_dummy = (geom_dummy, coords_dummy)
        y_dummy = field_dummy

        data = TripleCartesianProd(x_dummy, y_dummy, x_dummy, y_dummy)

        # Build model
        model = dde.B_Model(data, net)

        output_path = Data_train_morph_ratio + field + '_m' + morph + '_ratio' + str(nodes_ratio) + '_k/'

        # Load scalers
        Geom_scalers = joblib.load( output_path+ field + '_m' + morph + '_n' + str(num_nodes) + '_scaler_Geom.pkl')
        Field_scalers = joblib.load( output_path + field + '_m' + morph + '_n' + str(num_nodes) + '_scaler_Field.pkl')


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
            decay=("inverse time", 1, learning_rate/10.),
            metrics=metrics,
        )

        #######################################Sample Testing#####################################
        Train_data_name = sub + '-' + str(N_epoch) + '.pt'  # PyTorch recommended .pt or .pth extension
        checkpoint_path = output_path + Train_data_name

        model.restore(checkpoint_path)

        Resampledata_path = Data_morph_path + 'Time_Coord_' + field + '_m' + morph + '.npz'
        

        # Define color map (pre-defined)
        colors = [
            [0, 0, 255], [0, 93, 255], [0, 185, 255], [0, 255, 232],
            [0, 255, 139], [0, 255, 46], [46, 255, 0], [139, 255, 0],
            [232, 255, 0], [255, 185, 0], [255, 93, 0], [255, 0, 0]
        ]
        colors = np.array(colors) / 255.0
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        

        sample_index = np.load(Data_morph_path+'selected_indices_m' + morph + '.npy') # (0~X)
        num_sample = len(sample_index)

        

        # All possible sample indices (0 to 99, corresponding to actual IDs 1 to 100)
        all_possible_indices = np.arange(100)  # 0-based index

        test_indices = all_possible_indices


        Resampledata_path_all = Data_morph_path + 'Time_Coord_' + field + '_m100.npz'
        tmp_all = np.load(Resampledata_path_all)
        Coords_flt_all = tmp_all['pos']      # Before scaling
        Field_flt_all = tmp_all['dis']       # Before scaling

        
        Geom_flt_all = np.load(Data_morph_path+'Geom_Time_Load_m100.npy', allow_pickle=True).astype(float) # Geom params

        name_case = f'morph{morph}_ratio{nodes_ratio}'

        # Used to collect average R2 for each sample
        all_sample_avg_r2 = []
        all_sample_avg_mae = []
        all_sample_avg_rmse = []
        sample_metrics_list = []  # Store (sample_id, avg_r2, avg_mae)

        import time

        for i in test_indices:
            iter_start_time = time.time()

            sample = i + 1
            print('\n' + '='*50)
            print(f"Generalization test - Sample ID: {sample} (0-based index: {i})")
            print('='*50)

            start = i * 18
            end = start + 18

            Coords = Coords_flt_all[start:end, :, :]
            Field = Field_flt_all[start:end, :, :]
            Geom = Geom_flt_all[start:end, :]


            print('Geom.shape =', Geom.shape)
            print('Coords.shape =', Coords.shape)


            R2 = np.zeros((len(Geom), 2))
            MAE = np.zeros((len(Geom), 2))
            RMSE = np.zeros((len(Geom), 2))  

            # Predict_field_path = output_path + round + f'/Sample{sample}_' + round +'/'
            Predict_field_path = output_path + round + f'/Sample{sample}_' +'r6/'
            os.makedirs(Predict_field_path, exist_ok=True)

            for j in range(len(Geom)): 
                print('Time_step:' , str(j+1))
                Geom_t = Geom[j:j+1, :].astype(data_type) 
                Geom_t_scal = Geom_scalers.transform( Geom_t )

                
                Coords_t_flt = Coords[j, :,:].astype(data_type) 

                Coords_org = Coords[j, :,:].astype(data_type) 

                normalized_coords, min_coords, scale_factor = normalize_and_translate(Coords_t_flt)
                Coords_t_scal = np.expand_dims(normalized_coords, axis=0)

                x_pred_scal = (Geom_t_scal.astype(data_type), Coords_t_scal.astype(data_type)) 

                y_pred_scal = model.predict(x_pred_scal)                                                      # Test output data
                y_pred_scal = np.squeeze(y_pred_scal, axis=0) 
                y_pred_org = Field_scalers.inverse_transform(y_pred_scal)

                y_true_org = Field[j, :,:].astype(data_type)

                # Calculate mean absolute error (MAE) for each sample
                absolute_error = np.abs(y_pred_org - y_true_org)
                mae = np.mean(absolute_error)
                MAE[j,0] = j
                MAE[j,1] = mae
                print(f"Sample {sample}, Time step {j+1}: MAE = {mae:.4f}")

                # Calculate R2 for each sample
                r2 = r2_score(y_true_org, y_pred_org)
                R2[j,0] = j
                R2[j,1] = r2
                print(f"Sample {sample}, Time step {j+1}: R2 = {r2:.4f}")

                # === New: Calculate RMSE ===
                rmse = np.sqrt(np.mean((y_pred_org - y_true_org) ** 2))
                RMSE[j, 0] = j
                RMSE[j, 1] = rmse
                print(f"Sample {sample}, Time step {j+1}: RMSE = {rmse:.4f}")  

                # ========== Visualization and Saving ==========
                Predict_time_path = Predict_field_path + f'T{j}/'
                os.makedirs(Predict_time_path, exist_ok=True)

                # --- Save CSV ---
                np.savetxt(Predict_time_path + f'{field}_pred_N{j}.csv', y_pred_org, delimiter=',', fmt='%f')
                np.savetxt(Predict_time_path + f'{field}_true_N{j}.csv', y_true_org, delimiter=',', fmt='%f')
                np.savetxt(Predict_time_path + f'{field}_ae_N{j}.csv', absolute_error, delimiter=',', fmt='%f')
                np.savetxt(Predict_time_path + f'{field}_coord_N{j}.csv', Coords_org, delimiter=',', fmt='%f')

                R2_str = f"{r2:.4f}"

                # --- Create PyVista mesh ---
                mesh = pv.PolyData(Coords_org)
                mesh["field_true"] = y_true_org.flatten()
                mesh["field_pred"] = y_pred_org.flatten()
                mesh["field_ae"] = absolute_error.flatten()

                # --- Shared color limits ---
                clim_max, clim_min = 100, 0
                clim_true = [np.percentile(y_true_org, clim_min), np.percentile(y_true_org, clim_max)]
                clim_pred = [np.percentile(y_pred_org, clim_min), np.percentile(y_pred_org, clim_max)]
                clim_ae   = [np.percentile(absolute_error, clim_min), np.percentile(absolute_error, clim_max)]

                # --- Define plot parameters ---
                plots = [
                    ("field_true", clim_true, "FE"),
                    ("field_pred", clim_pred, "Geom-DeepONet"),
                    ("field_ae",   clim_ae,   "Geom-DeepONet_Ae")
                ]

                # --- Plot each image separately ---
                suffix_map = {
                    "FE": "True",
                    "Geom-DeepONet": "Pred",
                    "Geom-DeepONet_Ae": "Ae"
                }


                for scalar_name, clim, title_suffix in plots:
                    plotter = pv.Plotter(off_screen=True)  # Create a new Plotter each time
                    try:

                        plotter.add_mesh(
                            mesh,
                            scalars=scalar_name,
                            clim=clim,
                            cmap=custom_cmap,
                            point_size=5,
                            render_points_as_spheres=True,
                            show_scalar_bar=False  # Disable auto color bar first
                        )

                        # Manually add color bar with larger font
                        plotter.add_scalar_bar(
                            title=None,  # Optional: beautify title
                            n_labels=5,
                            title_font_size=None,      # Color bar title font size
                            label_font_size=30,      # Color bar value label font size
                            width=0.6,               # Color bar width (relative to window)
                            height=0.08,             # Color bar height
                            position_x=0.2,          # Horizontal position (0~1)
                            position_y=0.02          # Vertical position (0~1), near bottom
                        )
                        
                        title = f"{title_suffix}_T{j+1}"


                        plotter.add_title(title)
                        plotter.add_axes()
                        plotter.view_yx()

                        # Get current focal point
                        fp = plotter.camera.focal_point

                        # Calculate offset (based on model size)
                        bounds = mesh.bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
                        x_span = bounds[1] - bounds[0]
                        offset = x_span * 0.07  # Move up by about 10% of height

                        # Move focal point down (-X) to offset the model up (+X) in the view
                        plotter.camera.focal_point = (fp[0] - offset, fp[1], fp[2])

                        plotter.camera.zoom(1.5)

                        # ✅ Key: Use mapped short suffix as file name
                        file_suffix = suffix_map.get(title_suffix, title_suffix)  # Default fallback
                        png_path = Predict_time_path + f"{field}_{file_suffix}.png"
                        plotter.screenshot(png_path)
                        print(f" Saved: {png_path}")

                    except Exception as e:
                        print(f" Error plotting {scalar_name}: {e}")
                        raise
                    finally:
                        plotter.close()  # Ensure closing


            # Record end time of this iteration and calculate duration
            iter_end_time = time.time()
            iter_duration = iter_end_time - iter_start_time
            print(f"completed in: {iter_duration:.4f} seconds")

            # === Average R2 and MAE for each sample ===
            avg_r2 = np.mean(R2[:, 1])
            avg_mae = np.mean(MAE[:, 1])
            avg_rmse = np.mean(RMSE[:, 1])

            all_sample_avg_r2.append(avg_r2)
            all_sample_avg_mae.append(avg_mae)
            all_sample_avg_rmse.append(avg_rmse)

            sample_metrics_list.append((sample, avg_r2, avg_mae, avg_rmse))

            print(f"[Sample {sample}] Average R2 = {avg_r2:.4f}, Average MAE = {avg_mae:.4f}")

            # Save overall evaluation
            np.savetxt(Predict_field_path + f'{field}_R2.csv', R2, delimiter=',', fmt='%f')
            np.savetxt(Predict_field_path + f'{field}_MAE.csv', MAE, delimiter=',', fmt='%f')
            np.savetxt(Predict_field_path + f'{field}_RMSE.csv', RMSE, delimiter=',', fmt='%f')

            generate_gif_for_current_sample(Predict_field_path, field)



        # === Overall average R2 and MAE for all samples ===
        overall_avg_r2 = np.mean(all_sample_avg_r2)
        overall_avg_mae = np.mean(all_sample_avg_mae)
        overall_avg_rmse = np.mean(all_sample_avg_rmse)

        print(f"\n[All test samples] Average R2 = {overall_avg_r2:.4f}")
        print(f"[All test samples] Average MAE = {overall_avg_mae:.4f}")
        print(f"[All test samples] Average RMSE = {overall_avg_rmse:.4f}")

        # === Extract sample IDs and R2 values ===
        sample_metrics_list = np.array(sample_metrics_list, dtype=object)
        samples = np.array([x[0] for x in sample_metrics_list])      # Sample names, e.g., 'Sample1'
        r2_values = np.array([x[1] for x in sample_metrics_list])    # Corresponding R2
        mae_values = np.array([x[2] for x in sample_metrics_list])   # Corresponding MAE (if available)
        rmse_values = np.array([x[3] for x in sample_metrics_list])

        # === Find samples with highest and lowest R2 ===
        max_r2_idx = np.argmax(r2_values)
        min_r2_idx = np.argmin(r2_values)

        max_r2_sample = samples[max_r2_idx]
        max_r2_value = r2_values[max_r2_idx]
        min_r2_sample = samples[min_r2_idx]
        min_r2_value = r2_values[min_r2_idx]

        print(f"[Highest R2 sample] Sample {max_r2_sample}: R2 = {max_r2_value:.4f}")
        print(f"[Lowest R2 sample] Sample {min_r2_sample}: R2 = {min_r2_value:.4f}")

        # === Find key percentile samples (0%, 25%, 50%, 75%, 100%) ===
        percentiles = [0, 25, 50, 75, 100]
        percentile_labels = {
            0: "Worst (0%)",
            25: "25th percentile",
            50: "Median (50%)",
            75: "75th percentile",
            100: "Best (100%)"
        }

        # Calculate theoretical percentile values
        r2_percentile_targets = np.percentile(r2_values, percentiles)

        print("\n[Key percentile sample info]:")
        selected_samples = {}

        for p, target_val in zip(percentiles, r2_percentile_targets):
            # Find the actual sample closest to this percentile value
            idx = np.argmin(np.abs(r2_values - target_val))
            sample_id = samples[idx]
            actual_r2 = r2_values[idx]
            actual_mae = mae_values[idx] if len(mae_values) == len(r2_values) else "N/A"
            actual_rmse = rmse_values[idx]

            selected_samples[p] = {
                "sample": sample_id,
                "r2": actual_r2,
                "mae": actual_mae,
                "rmse": actual_rmse,
                "target": target_val
            }

            print(f"[{percentile_labels[p]}] Sample {sample_id}: R2 = {actual_r2:.4f}, MAE = {actual_mae:.4f}, RMSE = {actual_rmse:.4f}")

        # === Save summary statistics to output_path ===
        summary_path = os.path.join(output_path, f'summary_statistics_{round}.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Total test samples: {len(all_sample_avg_r2)}\n")
            f.write(f"Average R2: {overall_avg_r2:.4f}\n")
            f.write(f"Average MAE: {overall_avg_mae:.4f}\n")
            f.write(f"Average RMSE: {overall_avg_rmse:.4f}\n")
            f.write(f"Highest R2 sample: Sample {max_r2_sample}, R2 = {max_r2_value:.4f}\n")
            f.write(f"Lowest R2 sample: Sample {min_r2_sample}, R2 = {min_r2_value:.4f}\n")
            f.write("\n=== Key Percentile Samples ===\n")
            for p in percentiles:
                info = selected_samples[p]
                label = percentile_labels[p]
                f.write(f"{label}: Sample {info['sample']}, R2 = {info['r2']:.4f}, "
                        f"MAE = {info['mae']:.4f}, RMSE = {info['rmse']:.4f}\n")  # Includes RMSE

        print(f"\n✅ Statistics saved to: {summary_path}")




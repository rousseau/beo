from torchvision import datasets
import torchvision.transforms as transforms

transform = transforms.ToTensor()

train_data = datasets.MNIST(root = 'data', train = True, download = True, transform = transform)
test_data = datasets.MNIST(root = 'data', train = False, download = True, transform = transform)

#%%
import numpy as np
# extract all 3s
digit = 3

# convert to numpy arrays
x_train_all = train_data.data.numpy()
y_train_all = train_data.targets.numpy()
x_test_all = test_data.data.numpy()
y_test_all = test_data.targets.numpy()

x_train = x_train_all[y_train_all == digit, ...]
y_train = y_train_all[y_train_all == digit]
x_test = x_test_all[y_test_all == digit, ...]
y_test = y_test_all[y_test_all == digit]

#%%    
import matplotlib.pyplot as plt

#plt.imshow(x_train_all[0,:,:], cmap="gray")
#plt.show()

#%% split train into train and validation

nb_val = 1000  # keep 1,000 subjects for validation
x_val = x_train[-nb_val:, ...]  # this indexing means "the last nb_val entries" of the zeroth axis
y_val = y_train[-nb_val:]
x_train = x_train[:-nb_val, ...]
y_train = y_train[:-nb_val]

#%% normalization

x_train = x_train.astype('float')/255
x_val = x_val.astype('float')/255
x_test = x_test.astype('float')/255

#%%
nb_vis = 5

# choose nb_vis sample indexes
idx = np.random.choice(x_train.shape[0], nb_vis, replace=False)
example_digits = [f for f in x_train[idx, ...]]
plt.figure()
ax1 = plt.subplot(151)
ax1.imshow(example_digits[0], cmap="gray")
ax2 = plt.subplot(152)
ax2.imshow(example_digits[1], cmap="gray")
ax3 = plt.subplot(153)
ax3.imshow(example_digits[2], cmap="gray")
ax4 = plt.subplot(154)
ax4.imshow(example_digits[3], cmap="gray")
ax5 = plt.subplot(155)
ax5.imshow(example_digits[4], cmap="gray")
plt.show()

#%%

pad_amount = ((0, 0), (2,2), (2,2))

# fix data
x_train = np.pad(x_train, pad_amount, 'constant')
x_val = np.pad(x_val, pad_amount, 'constant')
x_test = np.pad(x_test, pad_amount, 'constant')

# verify
print('shape of training data', x_train.shape)

#%% unet 2d
import torch
import torch.nn as nn 

class Unet(nn.Module):
    def __init__(self, n_channels = 2, n_classes = 2, n_features = 8):
        super(Unet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_features = n_features

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )


        self.dc1 = double_conv(self.n_channels, self.n_features)
        self.dc2 = double_conv(self.n_features, self.n_features*2)
        self.dc3 = double_conv(self.n_features*2, self.n_features*4)
        self.dc4 = double_conv(self.n_features*6, self.n_features*2)
        self.dc5 = double_conv(self.n_features*3, self.n_features)
        self.mp = nn.MaxPool2d(2)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.out = nn.Conv2d(self.n_features, self.n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.dc1(x)

        x2 = self.mp(x1)
        x2 = self.dc2(x2)

        x3 = self.mp(x2)
        x3 = self.dc3(x3)

        x4 = self.up(x3)
        x4 = torch.cat([x4,x2], dim=1)
        x4 = self.dc4(x4)

        x5 = self.up(x4)
        x5 = torch.cat([x5,x1], dim=1)
        x5 = self.dc5(x5)
        return self.out(x5)

#%%
# check size
#tmp = torch.tensor(x_train[0,:,:])
tmp = torch.ones([1,2,32,32])

net = Unet()
outtmp = net.forward(tmp)
#%%
import torch.nn.functional as F
# code from voxelmorph repo
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
#%%
tmpsrc = torch.ones([1,1,32,32])
tmpflow = torch.ones([1,2,32,32])
net = SpatialTransformer(size=(32,32))
outtmp = net.forward(tmpsrc,tmpflow)

#%%
# todo lightning simple : unet fournit le flow, on deforme, on calcule le cout.
# ensuite ajout du diffeo (quelle difference, calcul de l'inverse, etc.)
# gestion du downsampling du flow

import pytorch_lightning as pl

train_losses = []

class morph_model(pl.LightningModule):
  def __init__(self, shape):
    super().__init__()   
    self.shape = shape
    self.unet_model = Unet()
    self.transformer = SpatialTransformer(size=shape)

  def forward(self,source,target):
    #concatenate images for unet
    x = torch.cat([source,target],dim=1)
    flow = self.unet_model(x)
    y_source = self.transformer(source, flow)
    
    return y_source, flow 

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    return optimizer

  def training_step(self, batch, batch_idx):
    source, target = batch

    y_source,flow = self(source,target)
    
    loss = F.mse_loss(target,y_source)
    self.log('train_loss', loss)
    train_losses.append(loss)
    return loss 

#%%
tmpsrc = torch.ones([1,1,32,32])
tmptar = torch.ones([1,1,32,32])
net = morph_model(shape=(32,32))
outtmp = net.forward(tmpsrc,tmptar)

#%%

batch_size = 32
n_training = x_train.shape[0]
source = torch.reshape(torch.Tensor(x_train[:n_training, ...]),(n_training,1,32,32))
#target =  torch.reshape(torch.Tensor(x_train[-n_training:, ...]),(n_training,1,32,32))

#trainset = torch.utils.data.TensorDataset(source, target)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)

from torch.utils.data import Dataset
class CustomDataSet(Dataset):
  def __init__(self, X):
    self.X = X
    self.len = len(self.X)

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    index_source = torch.randint(self.len,(1,))
    index_target = torch.randint(self.len,(1,))

    _source = torch.reshape(self.X[index_source],(1,32,32))
    _target = torch.reshape(self.X[index_target],(1,32,32))
    
    return _source, _target

trainset = CustomDataSet(source)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)


#%%
from pytorch_lightning.loggers import TensorBoardLogger

trainer = pl.Trainer(gpus=1, 
                     max_epochs=25,
                     logger=TensorBoardLogger(save_dir='lightning_logs', default_hp_metric=False, log_graph=True))
trainer.fit(net, trainloader)     

#%%
plt.figure()
plt.plot(train_losses)
plt.title('loss function', size=10)
plt.xlabel('steps', size=10)
plt.ylabel('loss value', size=10)

#%%

def visualization(source_np, target_np, net):
  #convert numpy to torch and reshape
  source = torch.reshape(torch.Tensor(source_np),(1,1,32,32))
  target =  torch.reshape(torch.Tensor(target_np),(1,1,32,32))
  #prediction
  warped, flow = net.forward(source,target)
  #convert to numpy
  warped_np = warped.cpu().detach().numpy()
  flow_np = flow.cpu().detach().numpy()
  #plot
  plt.figure()
  ax1 = plt.subplot(141)
  ax1.imshow(np.reshape(source_np,(32,32)), cmap="gray")
  ax2 = plt.subplot(142)
  ax2.imshow(np.reshape(target_np,(32,32)), cmap="gray")
  ax3 = plt.subplot(143)
  ax3.imshow(np.reshape(warped_np,(32,32)), cmap="gray")
  ax4 = plt.subplot(144)
  ax4.imshow(np.reshape(target_np-warped_np,(32,32)), cmap="gray")
  plt.show()
  
  plt.figure()
  plt.quiver(flow_np[0,0,:,:],flow_np[0,1,:,:])
  plt.show()

#%%
n_source = 0
n_target = 6
visualization(x_train[n_source, ...], x_train[n_target, ...], net)

#%% Generalization on another digit
digit = 5

x_fives = x_train_all[y_train_all == digit, ...]
x_fives = np.pad(x_fives, pad_amount, 'constant')

n_source = 0
n_target = 2
source = torch.reshape(torch.Tensor(x_fives[n_source, ...]),(1,1,32,32))
target =  torch.reshape(torch.Tensor(x_fives[n_target, ...]),(1,1,32,32))

warped, flow = net.forward(source,target)

source_np = source.cpu().detach().numpy()
target_np = target.cpu().detach().numpy()
warped_np = warped.cpu().detach().numpy()
flow_np = flow.cpu().detach().numpy()

plt.figure()
ax1 = plt.subplot(141)
ax1.imshow(np.reshape(source_np,(32,32)), cmap="gray")
ax2 = plt.subplot(142)
ax2.imshow(np.reshape(target_np,(32,32)), cmap="gray")
ax3 = plt.subplot(143)
ax3.imshow(np.reshape(warped_np,(32,32)), cmap="gray")
ax4 = plt.subplot(144)
ax4.imshow(np.reshape(target_np-warped_np,(32,32)), cmap="gray")
plt.show()

#%% Diffeomorphism (SVF)

#from voxelmorph repo
class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class SVF_model(pl.LightningModule):
  def __init__(self, shape, int_steps = 7):
    super().__init__()   
    self.shape = shape
    self.unet_model = Unet()
    self.transformer = SpatialTransformer(size=shape)
    self.int_steps = int_steps #number of integration step (i.e. flow is integrated from velocity fields). 
    self.vecint = VecInt(inshape=shape, nsteps=int_steps)
    
  def forward(self,source,target):
    #concatenate images for unet
    x = torch.cat([source,target],dim=1)
    flow = self.unet_model(x)
    
    if self.int_steps > 0:
      flow = self.vecint(flow)
    
    y_source = self.transformer(source, flow)
    
    return y_source, flow 

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    return optimizer

  def training_step(self, batch, batch_idx):
    source, target = batch

    y_source,flow = self(source,target)
    
    loss = F.mse_loss(target,y_source)
    self.log('train_loss', loss)
    train_losses.append(loss)
    return loss 

#%%
svf_net = SVF_model(shape=(32,32))

svf_trainer = pl.Trainer(gpus=1, 
                     max_epochs=25,
                     logger=TensorBoardLogger(save_dir='lightning_logs', default_hp_metric=False, log_graph=True))
svf_trainer.fit(svf_net, trainloader)     

#%%
n_source = 0
n_target = 6

visualization(x_train[n_source, ...], x_train[n_target, ...], svf_net)

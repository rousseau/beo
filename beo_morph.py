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

plt.imshow(x_train_all[0,:,:], cmap="gray")
plt.show()

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
    def __init__(self, n_channels = 2, n_classes = 2, n_features = 32):
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

class morph_model(pl.LightningModule):
  def __init__(self, unet_model, st_model):
    super().__init__()   
    self.unet_model = unet_model
    self.st_model = st_model

  def forward(self,source,target):
      #concatenate images for unet
      x = torch.cat([source,target],dim=1)
      flow = self.unet_model(x)
      
    return 

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    return optimizer

  def training_step(self, batch, batch_idx):
    x, y = batch
    rx2y,ry2x,rx2x,ry2y,rx2y2x,ry2x2y,idx2x,idy2y = self(x,y)

    loss = {}
    for k in lw.keys(): #compute only loss whose weight is non zero
      if lw[k] > 0:
        if k == 'normv':
          loss[k] = torch.sum( norm_v(feature_net,forward_blocks, x) )
          #loss[k] = loss_sobolev_norm_v(feature_net,forward_blocks, block_x2y_list, n_filters, x)              
        elif k == 'normdv':
          loss[k] = torch.sum( block_lipschitz(block_x2y_list,n_filters) )
        else:
          loss[k] = mse(eval(k),eval(k[-1]))
      else:
        loss[k] = torch.Tensor([0])[0]

    total_loss = 0
    for k in loss.keys():
      if lw[k] > 0:
        total_loss += lw[k] * loss[k]

    training_loss['loss'][-1] += total_loss.item() * x.shape[0]

    for k in loss.keys():
      training_loss[k][-1] += loss[k].item() * x.shape[0]


    #loss = F.mse_loss(rx2y,y) + F.mse_loss(ry2x,x) + F.mse_loss(rx2x,x) + F.mse_loss(ry2y,y)
    #loss+= F.mse_loss(rx2y2x,x) + F.mse_loss(ry2x2y,y) + F.mse_loss(idx2x,x) + F.mse_loss(idy2y,y)
    return total_loss      
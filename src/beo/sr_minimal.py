import torchio as tio
import os
from os.path import expanduser
import glob
import argparse
import torch
import torch.nn as nn 
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime

home = expanduser("~")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Equinus SR')
    parser.add_argument('--data_path', help='Input folder for static HR data', type=str, required = True)
    parser.add_argument('--saving_path', help='Output folder to save results', type=str, required = True)
    parser.add_argument('-e', '--epochs', help='Max epochs', type=int, required=False, default = 250)
    parser.add_argument('-m', '--model', help='Pytorch Lightning model', type=str, required=False)
    parser.add_argument('-p', '--patch_size', help='Patch size', type=int, required=False, default = 88)
    parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 1)
    parser.add_argument('-n', '--n_layers', help='Number of residual layers', type=int, required=False, default = 5)
    parser.add_argument('--overlap_patch_rate', help='Rate for patch overlap [0,1[ for inference', type=float, required=False, default=0.5)
    parser.add_argument('--n_inference', help='Number of subjects during inference', type=int, required=False, default=1)
    parser.add_argument('-s', '--suffix', help='Suffix', type=str, required=False, default='_sr')

    args = parser.parse_args()

    num_epochs = args.epochs
    patch_size = int(args.patch_size)
    batch_size = args.batch_size
    overlap_rate = args.overlap_patch_rate

    suffix = args.suffix
    suffix+= '_e'+str(num_epochs)
    suffix+= '_p'+str(patch_size)
    suffix+= '_b'+str(batch_size)
    #suffix+= '_n'+str(args.n_layers)
    saving_path=os.path.join(args.saving_path,datetime.now().strftime("%d-%m-%Y_%H-%M-%S")+suffix)
    print(saving_path)

    data_path = args.data_path
    all_hr = glob.glob(data_path+'**/*static_3DT1*.nii.gz', recursive=True)
    subjects = []
    for hr_file in all_hr:
        subject = tio.Subject(
            hr=tio.ScalarImage(hr_file),
            lr=tio.ScalarImage(hr_file),
        )
        subjects.append(subject) 
    print('Dataset size:', len(subjects), 'subjects')

    # DATA AUGMENTATION
    normalization = tio.ZNormalization()
    spatial = tio.RandomAffine(scales=0.1,degrees=10, translation=0, p=0.75)
    flip = tio.RandomFlip(axes=('LR',), flip_probability=0.5)

    tocanonical = tio.ToCanonical()

    # resolution hr : 0.26 x 0.26 x 0.5
    # resolution lr : 0.56 x 0.56 x 4
    # resolution claire : 0.41 x 0.41 x 8
    # test 1x1x2 (x4!)
    b1 = tio.Blur(std=(1,0.001,0.001), include='lr') #blur
    d1 = tio.Resample((2,1,1), include='lr')     #downsampling
    u1 = tio.Resample(target='hr', include='lr')     #upsampling

    transforms = [tocanonical, flip, spatial, normalization, b1, d1, u1]
    training_transform = tio.Compose(transforms)
    validation_transform = tio.Compose([tocanonical, normalization, b1, d1, u1])

    # SPLIT DATA
    seed = 42  # for reproducibility
    training_split_ratio = 0.9
    num_subjects = len(subjects)
    num_training_subjects = int(training_split_ratio * num_subjects)
    num_validation_subjects = num_subjects - num_training_subjects

    num_split_subjects = num_training_subjects, num_validation_subjects
    training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects, generator=torch.Generator().manual_seed(seed))

    training_set = tio.SubjectsDataset(
        training_subjects, transform=training_transform)

    validation_set = tio.SubjectsDataset(
        validation_subjects, transform=validation_transform)

    print('Training set:', len(training_set), 'subjects')
    print('Validation set:', len(validation_set), 'subjects')

    #%%
    num_workers = 8
    print('num_workers : '+str(num_workers))
    max_queue_length = 1024
    samples_per_volume = 16

    sampler = tio.data.UniformSampler(patch_size)

    patches_training_set = tio.Queue(
    subjects_dataset=training_set,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume,
    sampler=sampler,
    num_workers=num_workers,
    shuffle_subjects=True,
    shuffle_patches=True,
    )

    patches_validation_set = tio.Queue(
    subjects_dataset=validation_set,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume,
    sampler=sampler,
    num_workers=num_workers,
    shuffle_subjects=False,
    shuffle_patches=False,
    )

    training_loader_patches = torch.utils.data.DataLoader(
    patches_training_set, batch_size=batch_size)

    validation_loader_patches = torch.utils.data.DataLoader(
    patches_validation_set, batch_size=batch_size)

#%%
    class ResNetBlock(torch.nn.Module):
        def __init__(self, in_channels = 32):
            super(ResNetBlock, self).__init__()
            self.in_channels = in_channels
            def double_conv(in_channels):
                return nn.Sequential(
                    nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm3d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm3d(in_channels),
                )
            self.dc = double_conv(self.in_channels)
        
        def forward(self,x):
            z = self.dc(x)
            return x+z

    class ResNet(torch.nn.Module):
        def __init__(self, in_channels = 1, out_channels = 1, n_filters = 32, n_layers = 5):
            super(ResNet, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.n_features = n_filters
            self.n_layers = n_layers

            self.blocks = torch.nn.ModuleList()
            for i in range(n_layers):
                self.blocks.append(ResNetBlock(in_channels = self.n_features))
            
            self.inconv = nn.Conv3d(self.in_channels, self.n_features, kernel_size=3, padding=1)
            self.outconv = nn.Conv3d(self.n_features, self.out_channels, kernel_size=3, padding=1)

        def forward(self,x):
            z = self.inconv(x)
            z = nn.ReLU()(z)
            for i in range(self.n_layers):
                z = self.blocks[i](z)
            z = self.outconv(z)
            return z  

    class Unet(nn.Module):
        def __init__(self, in_channels = 1, out_channels = 1, n_filters = 8):
            super(Unet, self).__init__()

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.n_filters = n_filters

            self.inconv = nn.Conv3d(self.in_channels, self.n_filters, kernel_size=3, padding=1)

            def double_conv(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True),
                )

            self.dc1 = double_conv(self.in_channels, self.n_filters)
            self.dc2 = double_conv(self.n_filters, self.n_filters)
            self.dc3 = double_conv(self.n_filters, self.n_filters)
            self.dc4 = double_conv(self.n_filters, self.n_filters)
            self.dc4out = double_conv(self.n_filters, self.out_channels)

            self.dc5 = double_conv(self.n_filters*2, self.n_filters)
            self.dc5out = double_conv(self.n_filters, self.out_channels)

            self.dc6 = double_conv(self.n_filters*2, self.n_filters)
            self.dc6out = double_conv(self.n_filters, self.out_channels)

            self.dc7 = double_conv(self.n_filters*2, self.n_filters)

            self.ap = nn.AvgPool3d(2)

            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

            self.out = nn.Conv3d(self.n_filters, self.out_channels, kernel_size=1)

        def forward(self, x):
            x1 = self.dc1(x) # p 64, reso native z en 0.5

            x2 = self.ap(x1) # p 32, reso native z en 1
            x2 = self.dc2(x2)

            x3 = self.ap(x2) # p 16, reso native z en 2
            x3 = self.dc3(x3)

            x4 = self.ap(x3) # p 8, reso native z en 4
            x4 = self.dc4(x4)
            x4_out = self.dc4out(self.up(self.up(self.up(x4))))

            x5 = torch.cat([self.up(x4),x3], dim=1)
            x5 = self.dc5(x5)
            x5_out = self.dc5out(self.up(self.up(x5)))

            x6 = torch.cat([self.up(x5),x2], dim=1)
            x6 = self.dc6(x6)
            x6_out = self.dc6out(self.up(x6))

            x7 = torch.cat([self.up(x6),x1], dim=1)
            x7 = self.dc7(x7)

            return (self.out(x7), x6_out, x5_out, x4_out)

#%%
    class Net(pl.LightningModule):
        def __init__(self, in_channels = 1, out_channels = 1, n_filters = 32, n_layers = 5, activation = 'relu'):
            super(Net, self).__init__()

            self.in_channels = in_channels 
            self.out_channels = out_channels
            self.n_filters = n_filters

            self.net = ResNet(in_channels = in_channels, out_channels = out_channels, n_filters = n_filters, n_layers = 10)
            #self.net_8_to_4 = ResNet(in_channels = in_channels, out_channels = out_channels, n_filters = n_filters, n_layers = n_layers)
            #self.net_4_to_2 = ResNet(in_channels = in_channels, out_channels = out_channels, n_filters = n_filters, n_layers = n_layers)
            #self.net_2_to_1 = ResNet(in_channels = in_channels, out_channels = out_channels, n_filters = n_filters, n_layers = n_layers)
            #self.net_1_to_05 = ResNet(in_channels = in_channels, out_channels = out_channels, n_filters = n_filters, n_layers = n_layers)
            #self.net = Unet(in_channels = in_channels, out_channels = out_channels, n_filters = n_filters)

            self.save_hyperparameters()

        def forward(self, x):
            #1, 2, 4, 8
            #x_8_to_4 = self.net_8_to_4(x)
            #x_4_to_2 = self.net_4_to_2(x_8_to_4)
            #x_2_to_1 = self.net_2_to_1(x_4_to_2)
            #x_1_to_05 = self.net_1_to_05(x_2_to_1)
            #return self.net(x)
            #return (x_8_to_4,x_4_to_2,x_2_to_1,x_1_to_05)
            return self.net(x)

        def evaluate_batch(self, batch):
            patches_batch = batch

            hr = patches_batch['hr'][tio.DATA]
            lr = patches_batch['lr'][tio.DATA]

            #(x7,x6,x5,x4) = self(lr)
            rlr = self(lr)

            loss_recon = F.l1_loss(rlr,hr)# + F.l1_loss(x6,hr) + F.l1_loss(x5,hr) + F.l1_loss(x4,hr) # decomp residuelle ou non ?

            return loss_recon
            
        def training_step(self, batch, batch_idx):
            loss = self.evaluate_batch(batch)
            self.log('train_loss', loss)
            return loss

        def validation_step(self, batch, batch_idx):
            loss = self.evaluate_batch(batch)
            self.log('val_loss', loss)
            self.log("hp_metric", loss)
            return loss

        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            return self(batch)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
            return optimizer  

    #%%
    if args.model is not None:
        model = Net.load_from_checkpoint(args.model)
    else:    
        model = Net()

    logger = TensorBoardLogger(save_dir = saving_path)
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=num_epochs,
        logger=logger,
        strategy="ddp_find_unused_parameters_true",
    )
    trainer.fit(model, training_loader_patches, validation_loader_patches)
    trainer.save_checkpoint(saving_path+'/SR.ckpt')
    torch.save(model.state_dict(), saving_path+'/SR_torch.pt')

    #%%
    print('Inference')
    model.eval()

    for i in range(args.n_inference):

        subject = validation_set[i]
        patch_overlap = int(patch_size / 2)  

        grid_sampler = tio.inference.GridSampler(
                        subject,
                        patch_size,
                        patch_overlap,
                    )

        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)

        output_keys = ['rlr']

        aggregators = {}
        for k in output_keys:
            aggregators[k] = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')

        device = torch.device('cuda:0')  
        model.to(device)

        with torch.no_grad():
            for patches_batch in patch_loader:
                lr = patches_batch['lr'][tio.DATA]

                locations = patches_batch[tio.LOCATION]

                lr = lr.to(device)
                rlr = model(lr)

                aggregators['rlr'].add_batch(rlr.cpu(), locations)  

        print('Saving images...')
        for k in output_keys:
            output = aggregators[k].get_output_tensor()
            o = tio.ScalarImage(tensor=output, affine=subject['hr'].affine)
            o.save(saving_path+'/'+k+'_'+str(i)+'.nii.gz')

        subject['hr'].save(saving_path+'/hr_'+str(i)+'.nii.gz')
        subject['lr'].save(saving_path+'/lr_'+str(i)+'.nii.gz')      

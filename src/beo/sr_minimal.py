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

    args = parser.parse_args()

    num_epochs = args.epochs
    patch_size = int(args.patch_size)
    batch_size = args.batch_size
    overlap_rate = args.overlap_patch_rate

    suffix = '_sr_e'+str(num_epochs)
    saving_path=os.path.join(args.saving_path,datetime.now().strftime("%d-%m-%Y_%H-%M-%S")+suffix)
    print(saving_path)

    data_path = args.data_path
    all_hr = glob.glob(data_path+'**/*static_3DT1*.nii.gz', recursive=True)
    subjects = []
    for hr_file in all_hr:
        subject = tio.Subject(
            hr=tio.ScalarImage(hr_file),
            lr8=tio.ScalarImage(hr_file),
            lr4=tio.ScalarImage(hr_file),
            lr2=tio.ScalarImage(hr_file),
            lr1=tio.ScalarImage(hr_file),
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
    b8 = tio.Blur(std=(2,0.001,0.001), include='lr8') #blur
    d8 = tio.Resample((8,0.5,0.5), include='lr8')     #downsampling
    u8 = tio.Resample(target='hr', include='lr8')     #upsampling
    b4 = tio.Blur(std=(1.5,0.001,0.001), include='lr4') #blur
    d4 = tio.Resample((4,0.5,0.5), include='lr4')     #downsampling
    u4 = tio.Resample(target='hr', include='lr4')     #upsampling
    b2 = tio.Blur(std=(1,0.001,0.001), include='lr2') #blur
    d2 = tio.Resample((2,0.5,0.5), include='lr2')     #downsampling
    u2 = tio.Resample(target='hr', include='lr2')     #upsampling
    b1 = tio.Blur(std=(1,0.001,0.001), include='lr1') #blur
    d1 = tio.Resample((1,0.5,0.5), include='lr1')     #downsampling
    u1 = tio.Resample(target='hr', include='lr1')     #upsampling

    transforms = [tocanonical, flip, spatial, normalization, b8, d8, u8, b4, d4, u4, b2, d2, u2, b1, d1, u1]
    training_transform = tio.Compose(transforms)
    validation_transform = tio.Compose([tocanonical, normalization, b8, d8, u8, b4, d4, u4, b2, d2, u2, b1, d1, u1])

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

#%%
    class Net(pl.LightningModule):
        def __init__(self, in_channels = 1, out_channels = 1, n_filters = 32, n_layers = 5, activation = 'relu'):
            super(Net, self).__init__()

            self.in_channels = in_channels 
            self.out_channels = out_channels
            self.n_features = n_filters

            #self.net = ResNet(in_channels = in_channels, out_channels = out_channels, n_filters = n_filters, n_layers = 10)
            self.net_8_to_4 = ResNet(in_channels = in_channels, out_channels = out_channels, n_filters = n_filters, n_layers = n_layers)
            self.net_4_to_2 = ResNet(in_channels = in_channels, out_channels = out_channels, n_filters = n_filters, n_layers = n_layers)
            self.net_2_to_1 = ResNet(in_channels = in_channels, out_channels = out_channels, n_filters = n_filters, n_layers = n_layers)
            self.net_1_to_05 = ResNet(in_channels = in_channels, out_channels = out_channels, n_filters = n_filters, n_layers = n_layers)
            self.save_hyperparameters()

        def forward(self, x):
            #1, 2, 4, 8
            x_8_to_4 = self.net_8_to_4(x)
            x_4_to_2 = self.net_4_to_2(x_8_to_4)
            x_2_to_1 = self.net_2_to_1(x_4_to_2)
            x_1_to_05 = self.net_1_to_05(x_2_to_1)
            #return self.net(x)
            return (x_8_to_4,x_4_to_2,x_2_to_1,x_1_to_05)

        def evaluate_batch(self, batch):
            patches_batch = batch

            hr = patches_batch['hr'][tio.DATA]
            lr8 = patches_batch['lr8'][tio.DATA]
            lr4 = patches_batch['lr4'][tio.DATA]
            lr2 = patches_batch['lr2'][tio.DATA]
            lr1 = patches_batch['lr1'][tio.DATA]

            (x_8_to_4,x_4_to_2,x_2_to_1,x_1_to_05) = self(lr8)

            loss_recon = F.l1_loss(x_1_to_05,hr) + F.l1_loss(x_2_to_1,lr1) + F.l1_loss(x_4_to_2,lr2) + F.l1_loss(x_8_to_4,lr4)

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
        model = Net(n_layers=args.n_layers)

    logger = TensorBoardLogger(save_dir = saving_path)
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=num_epochs,
        logger=logger,
        #strategy="ddp_find_unused_parameters_true"
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
                lr = patches_batch['lr8'][tio.DATA]

                locations = patches_batch[tio.LOCATION]

                lr = lr.to(device)
                (x_8_to_4,x_4_to_2,x_2_to_1,x_1_to_05) = model(lr)
                rlr = x_1_to_05

                aggregators['rlr'].add_batch(rlr.cpu(), locations)  

        print('Saving images...')
        for k in output_keys:
            output = aggregators[k].get_output_tensor()
            o = tio.ScalarImage(tensor=output, affine=subject['hr'].affine)
            o.save(saving_path+'/'+k+'_'+str(i)+'.nii.gz')

        subject['hr'].save(saving_path+'/hr_'+str(i)+'.nii.gz')
        subject['lr'].save(saving_path+'/lr_'+str(i)+'.nii.gz')      

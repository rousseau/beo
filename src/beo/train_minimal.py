import torchio as tio
import os
from os.path import expanduser
import glob
home = expanduser("~")
import itertools
import torch
import torch.nn as nn 
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
from matplotlib import pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Synthesis')
    parser.add_argument('-e', '--epochs', help='Max epochs', type=int, required=False, default = 250)
    parser.add_argument('--saving_path', help='Output folder to save results', type=str, required = True)
    parser.add_argument('--static_path', help='Input folder for static data', type=str, required = True)
    parser.add_argument('--dynamic_path', help='Input folder for dynamic data', type=str, required = True)

    args = parser.parse_args()

    num_epochs = args.epochs

    static_path=args.static_path
    dynamic_path=args.dynamic_path
    saving_path=os.path.join(args.saving_path,datetime.now().strftime("%d-%m-%Y_%H-%M-%S")+'_results')
    print(saving_path)
 
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)

    subjects=[]
    check_subjects=[]
    HR_path=os.path.join(static_path)
    LR_path=os.path.join(dynamic_path)
    subject_names = os.listdir(LR_path)
    forbidden_subjets=['sub_E09','sub_T10','sub_T11','sub_E11'] #'sub_T09','sub_T07'
    for s in subject_names:
        if s not in forbidden_subjets:
            sequences=os.listdir(os.path.join(LR_path,s))
            for seq in sequences:
                volumes=os.listdir(os.path.join(LR_path,s,seq))
                for v in volumes:
                    #HR=glob.glob(os.path.join(HR_path,s+'*.nii.gz'))
                    LR=os.path.join(LR_path,s,seq,v)
                    HR_files=glob.glob(os.path.join(HR_path,s,seq,v.split('.')[0]+'*.nii.gz'))
                    HR_files=[i for i in HR_files if i.find('footmask')==-1]
                    HR_files=[i for i in HR_files if i.find('segment')==-1]
                    if len(HR_files)>0:
                        HR=HR_files[0]
                        file=HR.split('/')[-1].replace('_registration','_footmask_registration')
                        SEG=os.path.join(HR_path,s,seq,file)
                        if s not in check_subjects:
                            check_subjects.append(s)
                        subject=tio.Subject(
                            subject_name=s,
                            LR_image=tio.ScalarImage(LR),
                            HR_image=tio.ScalarImage(HR),
                            label=tio.LabelMap(SEG)
                        )
                        subjects.append(subject)

    dataset = tio.SubjectsDataset(subjects)
    print('Dataset size:', len(check_subjects), 'subjects')

    #print(dataset[0].plot())

    # DATA AUGMENTATION
    flip = tio.RandomFlip(axes=('LR',), flip_probability=1)
    bias = tio.RandomBiasField(coefficients = 0.5, p=0.5)
    noise = tio.RandomNoise(std=0.1, p=0.25)

    normalization = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    spatial = tio.RandomAffine(scales=0.1,degrees=(20,0,0), translation=0, p=0.75)

    transforms = tio.Compose([flip, spatial, bias, normalization, noise])
    #transforms(dataset[0]).plot()

    training_transform = tio.Compose([flip, spatial, bias, normalization, noise])
    validation_transform = tio.Compose([normalization])   

    # SPLIT DATA
    seed = 42  # for reproducibility
    training_split_ratio = 0.8
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
    num_workers = 4
    print('num_workers : '+str(num_workers))
    patch_size = (64,64,1)
    max_queue_length = 1024
    samples_per_volume = 128
    batch_size = 128

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

    print("Nombre de patches de train: " + str(len(training_loader_patches.dataset)))
    print("Nombre de patches de test: " + str(len(validation_loader_patches.dataset)))

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

    class Generator(torch.nn.Module):
        def __init__(self, in_channels = 1, out_channels = 1, n_filters = 32, n_layers = 5):
            super(Generator, self).__init__()
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

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
    
            self.n_features = 16
            #Patch 64*64*1
            self.conv1 = nn.Conv3d(1, self.n_features, kernel_size=3, padding=1)
            self.conv2 = nn.Conv3d(self.n_features, 2*self.n_features, stride=(2,2,1), kernel_size=3, padding=1)
            self.conv3 = nn.Conv3d(2*self.n_features, 4*self.n_features, stride=(2,2,1), kernel_size=3, padding=1)
            self.conv4 = nn.Conv3d(4*self.n_features, 8*self.n_features, stride=(2,2,1), kernel_size=3, padding=1)
            self.conv5 = nn.Conv3d(8*self.n_features
                                , 16*self.n_features, stride=(2,2,1), kernel_size=3, padding=1)
            self.conv6 = nn.Conv3d(16*self.n_features, 1, kernel_size=1, padding=0)

        def forward(self, x):
            x = self.conv1(x) # 64x64x1x16
            x = nn.ReLU()(x)
            x = self.conv2(x) # 32x32x1x32
            x = nn.ReLU()(x)
            x = self.conv3(x) # 16x16x1x64
            x = nn.ReLU()(x)
            x = self.conv4(x) # 8x8x1x128
            x = nn.ReLU()(x)
            x = self.conv5(x) # 4x4x1x256
            x = nn.ReLU()(x)
            x = nn.AvgPool3d(kernel_size=(4,4,1), stride=1)(x)
            x = self.conv6(x)
            return nn.Sigmoid()(x)

    #%%
    class GAN(pl.LightningModule):
        def __init__(self):
            super(GAN, self).__init__()
            self.generator_X = Generator()
            self.discriminator_X = Discriminator()
            self.generator_Y = Generator()
            self.discriminator_Y = Discriminator()
            self.automatic_optimization = False

        def forward(self, x, y):
            fake_y = self.generator_Y(x)
            fake_x = self.generator_X(y)
            return (fake_x, fake_y)
        
        def training_step(self, batch, batch_idx):
            x = batch['LR_image'][tio.DATA]
            y = batch['HR_image'][tio.DATA]        

            optimizer_g, optimizer_dx, optimizer_dy = self.optimizers()

            # train generators
            self.toggle_optimizer(optimizer_g)

            # GAN loss
            fake_y = self.generator_Y(x)
            pred_fake_y = self.discriminator_Y(fake_y)
            loss_GAN_X2Y = F.mse_loss(pred_fake_y, torch.ones_like(pred_fake_y))
            
            fake_x = self.generator_X(y)
            pred_fake_x = self.discriminator_X(fake_x)
            loss_GAN_Y2X = F.mse_loss(pred_fake_x, torch.ones_like(pred_fake_x))

            # Cycle loss   
            recon_x = self.generator_X(fake_y)
            recon_y = self.generator_Y(fake_x)
            loss_cycle_XYX = F.l1_loss(x, recon_x)*10.0 
            loss_cycle_YXY = F.l1_loss(y, recon_y)*10.0 

            # Total loss     
            g_loss = loss_GAN_X2Y + loss_GAN_Y2X + loss_cycle_XYX + loss_cycle_YXY
            self.manual_backward(g_loss)
            optimizer_g.step()
            optimizer_g.zero_grad()
            self.untoggle_optimizer(optimizer_g)  
            self.log("g_loss", g_loss, on_epoch=True)      

            # train discriminator dx
            self.toggle_optimizer(optimizer_dx)

            # Real loss
            pred_real_x = self.discriminator_X(x)
            loss_dx_real = F.mse_loss(pred_real_x, torch.ones_like(pred_real_x))

            # Fake loss
            fake_x = self.generator_X(y)
            pred_fake_x = self.discriminator_X(fake_x)
            loss_dx_fake = F.mse_loss(pred_fake_x, torch.zeros_like(pred_fake_x))

            # Total loss
            d_loss_x = (loss_dx_real + loss_dx_fake)*0.5
            self.manual_backward(d_loss_x)
            optimizer_dx.step()
            optimizer_dx.zero_grad()
            self.untoggle_optimizer(optimizer_dx)
            self.log("d_loss_x", d_loss_x, on_epoch=True)      


            # train discriminator dy
            self.toggle_optimizer(optimizer_dy)

            # Real loss
            pred_real_y = self.discriminator_Y(y)
            loss_dy_real = F.mse_loss(pred_real_y, torch.ones_like(pred_real_y))

            # Fake loss
            fake_y = self.generator_Y(x)
            pred_fake_y = self.discriminator_Y(fake_y)
            loss_dy_fake = F.mse_loss(pred_fake_y, torch.zeros_like(pred_fake_y))

            # Total loss
            d_loss_y = (loss_dy_real + loss_dy_fake)*0.5
            self.manual_backward(d_loss_y)
            optimizer_dy.step()
            optimizer_dy.zero_grad()
            self.untoggle_optimizer(optimizer_dy)
            self.log("d_loss_y", d_loss_y, on_epoch=True)      

        def configure_optimizers(self):

            opt_g = torch.optim.Adam(itertools.chain(self.generator_X.parameters(), self.generator_Y.parameters()), lr=1e-4)
            opt_dx = torch.optim.Adam(self.discriminator_X.parameters(), lr=1e-4)
            opt_dy = torch.optim.Adam(self.discriminator_Y.parameters(), lr=1e-4)

            return [opt_g, opt_dx, opt_dy], []

        def validation_step(self, batch, batch_idx):
            x = batch['LR_image'][tio.DATA]
            y = batch['HR_image'][tio.DATA]   

            fake_x = self.generator_X(y)
            fake_y = self.generator_Y(x)

            recon_x = self.generator_X(fake_y)
            recon_y = self.generator_Y(fake_x)

            plt.figure()
            plt.subplot(2,3,1)
            plt.imshow(x[0,0,:,:].cpu().detach().numpy(), cmap="gray")
            plt.subplot(2,3,2)
            plt.imshow(fake_y[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.subplot(2,3,3)
            plt.imshow(recon_x[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.subplot(2,3,4)
            plt.imshow(y[0,0,:,:].cpu().detach().numpy(), cmap="gray")
            plt.subplot(2,3,5)
            plt.imshow(fake_x[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.subplot(2,3,6)
            plt.imshow(recon_y[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.savefig(os.path.join(saving_path,'validation_epoch-'+str(self.current_epoch)+'.png'))
            plt.close()

    #%%
    model = GAN()

    logger = TensorBoardLogger(save_dir = saving_path)
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=num_epochs,
        logger=logger,
        strategy="ddp_find_unused_parameters_true"
    )
    trainer.fit(model, training_loader_patches, validation_loader_patches)
    trainer.save_checkpoint(saving_path+'/GAN.ckpt')
    torch.save(model.state_dict(), saving_path+'/GAN_torch.pt')

    #%%
    print('Inference')
    model.eval()
    subject = validation_set[0]
    patch_overlap = (32,32,0)

    grid_sampler = tio.inference.GridSampler(
                    subject,
                    patch_size,
                    patch_overlap,
                )

    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)

    output_keys = ['x2y','y2x','x2y2x','y2x2y']

    aggregators = {}
    for k in output_keys:
        aggregators[k] = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')

    device = torch.device('cuda:0')  
    model.to(device)

    with torch.no_grad():
        for patches_batch in patch_loader:
            x = patches_batch['LR_image'][tio.DATA]
            y = patches_batch['HR_image'][tio.DATA] 

            locations = patches_batch[tio.LOCATION]

            x = x.to(device)
            y = y.to(device)

            x2y = model.generator_Y(x)
            y2x = model.generator_X(y)

            x2y2x = model.generator_X(x2y)
            y2x2y = model.generator_Y(y2x)

            aggregators['x2y'].add_batch(x2y.cpu(), locations)
            aggregators['y2x'].add_batch(y2x.cpu(), locations)
            aggregators['x2y2x'].add_batch(x2y2x.cpu(), locations)
            aggregators['y2x2y'].add_batch(y2x2y.cpu(), locations)

    print('Saving images...')
    for k in output_keys:
        output = aggregators[k].get_output_tensor()
        o = tio.ScalarImage(tensor=output, affine=subject['HR_image'].affine)
        o.save(saving_path+'/'+k+'.nii.gz')

    subject['HR_image'].save(saving_path+'/y.nii.gz')
    subject['LR_image'].save(saving_path+'/x.nii.gz')      


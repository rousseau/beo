import argparse
from datetime import datetime
import glob
import torchio as tio
import torch
import torch.nn as nn 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo ArSSR')
    parser.add_argument('--encoder', help='Encoder network (RDN, ResCNN, SRResnet)', type=str, default='RDN')
    parser.add_argument('--decoder_depth', help='Decoder depth', type=int, default=8)
    parser.add_argument('--decoder_width', help='Decoder width', type=int, default=256)
    parser.add_argument('--feature_dim', help='Dimension of the feature vector', type=int, default=128)

    parser.add_argument('-e', '--epochs', help='Max epochs', type=int, required=False, default = 250)
    parser.add_argument('-p', '--patch_size', help='Patch size', type=int, required=False, default = 88)
    parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 1)
    parser.add_argument('--samples_per_volume', help='Sampled patch per volume', type=int, required=False, default = 16)

    parser.add_argument('-s', '--suffix', help='Suffix', type=str, required=False, default='_arssr')

    parser.add_argument('--data_path', help='Input folder for static HR data', type=str, required = True)
    parser.add_argument('--saving_path', help='Output folder to save results', type=str, required = True)

    args = parser.parse_args()

    num_epochs = args.epochs
    patch_size = int(args.patch_size)
    batch_size = args.batch_size
    samples_per_volume = args.samples_per_volume

    suffix = args.suffix
    suffix+= '_e'+str(num_epochs)
    suffix+= '_p'+str(patch_size)
    suffix+= '_b'+str(batch_size)
    suffix+= '_s'+str(samples_per_volume)

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

    b1 = tio.RandomBlur(std=(0.8,1.2,0.25,0.75,0.25,0.75), include='lr') #blur
    d0 = tio.Resample((4,0.5,0.5), include='lr')     #downsampling
    d1 = tio.RandomAnisotropy(axes=0, downsampling=(4,8), include='lr')     #downsampling
    d2 = tio.RandomAnisotropy(axes=1, downsampling=(2,3), include='lr')     #downsampling
    d3 = tio.RandomAnisotropy(axes=2, downsampling=(2,3), include='lr')     #downsampling
    u1 = tio.Resample(target='hr', include='lr')     #upsampling

    transforms = [tocanonical, flip, spatial, normalization, b1, d1, d2, d3, u1]
    training_transform = tio.Compose(transforms)
    validation_transform = tio.Compose([tocanonical, normalization, b1, d0, u1])

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

#%% RDN encoder
# -------------------------------
# RDN encoder network
# <Zhang, Yulun, et al. "Residual dense network for image super-resolution.">
# Here code is modified from: https://github.com/yjn870/RDN-pytorch/blob/master/models.py
# -------------------------------
    class DenseLayer(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(DenseLayer, self).__init__()
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return torch.cat([x, self.relu(self.conv(x))], 1)


    class RDB(nn.Module):
        def __init__(self, in_channels, growth_rate, num_layers):
            super(RDB, self).__init__()
            self.layers = nn.Sequential(
                *[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])
            # local feature fusion
            self.lff = nn.Conv3d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

        def forward(self, x):
            return x + self.lff(self.layers(x))  # local residual learning


    class RDN(nn.Module):
        def __init__(self, feature_dim=128, num_features=64, growth_rate=64, num_blocks=8, num_layers=3):
            super(RDN, self).__init__()
            self.G0 = num_features
            self.G = growth_rate
            self.D = num_blocks
            self.C = num_layers
            # shallow feature extraction
            self.sfe1 = nn.Conv3d(1, num_features, kernel_size=3, padding=3 // 2)
            self.sfe2 = nn.Conv3d(num_features, num_features, kernel_size=3, padding=3 // 2)
            # residual dense blocks
            self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
            for _ in range(self.D - 1):
                self.rdbs.append(RDB(self.G, self.G, self.C))
            # global feature fusion
            self.gff = nn.Sequential(
                nn.Conv3d(self.G * self.D, self.G0, kernel_size=1),
                nn.Conv3d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
            )
            self.output = nn.Conv3d(self.G0, feature_dim, kernel_size=3, padding=3 // 2)

        def forward(self, x):
            sfe1 = self.sfe1(x)
            sfe2 = self.sfe2(sfe1)
            x = sfe2
            local_features = []
            for i in range(self.D):
                x = self.rdbs[i](x)
                local_features.append(x)
            x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
            x = self.output(x)
            return x
    
#%%
    class Net(pl.LightningModule):
        def __init__(self):
            super(Net, self).__init__()

            self.encoder = RDN()

            self.save_hyperparameters()

        def forward(self, x, coords):
            feature_map = self.encoder(x)
            features_vector = ...
            features = ...
            pred = self.decoder(features)
            return pred

        def evaluate_batch(self, batch):
            patches_batch = batch

            hr = patches_batch['hr'][tio.DATA]
            lr = patches_batch['lr'][tio.DATA]
            corner_coords = patches_batch[tio.LOCATION] #Nx6 (corners of patch)

            batch_size = batch.shape[0]
            patch_size = hr.shape[1]
            num_points_per_patch = patch_size ** 3
            point_indices = torch.zeros((batch_size, num_points_per_patch, 3))

            # Iterate over the patches and get the indices of the points in each patch.
            for i in range(batch_size):
                # Get the coordinates of the points in the patch.
                x = torch.arange(start=corner_coords[i,0], end=corner_coords[i,3])
                y = torch.arange(start=corner_coords[i,1], end=corner_coords[i,4])
                z = torch.arange(start=corner_coords[i,2], end=corner_coords[i,5])

                # Create a meshgrid of the coordinates.
                meshgrid = torch.meshgrid(x, y, z)

                # Flatten the meshgrid into a tensor of shape (patch_size ** 3, 3).
                flattened_meshgrid = torch.stack(meshgrid, dim=2).view(-1, 3)

                # Add the batch index to each point coordinate.
                flattened_meshgrid = flattened_meshgrid.unsqueeze(0).repeat(batch_size, 1, 1)

                # Set the point indices in the tensor.
                point_indices[i] = flattened_meshgrid

            return point_indices

            rlr = self(lr)

            loss_recon = 0

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

import torch
import torchvision
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import torchio as tio
import os
import torchvision.transforms.functional as TF

# Fix for Intel Mkl (MacOS)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize dHCP dataset')
    parser.add_argument('-i', '--input', help='Input tsv dhcp file', type=str, required = True)
    parser.add_argument('-b', '--bids_dir', help='BIDS directory', type=str, required = True)
    parser.add_argument('-k', '--keyword', help='Keyword used to select images (like desc-restore_T2w)', type=str, required=False, default='desc-restore_T2w')
    args = parser.parse_args()

    # dHCP dataset
    # fmriresults01.txt : derivatives (fetal and neonatal)
    # image03.txt : raw (fetal and neonatal)
    df = pd.read_csv(args.input, sep='\t')

    # Sorting by age in ascending order
    df = df.sort_values(by=['scan_age'])

    print('Number of subjects :'+str(len(df['subject_id'])))    
    lst_imgs = []
    for index, row in df.iterrows():
        subject = row['subject_id']
        session = row['session_id']
        data_path = os.path.join(args.bids_dir, subject, session, 'anat')
        image_filename = subject+'_'+session+'_'+args.keyword+'.nii.gz'   
        lst_imgs.append(os.path.join(data_path, image_filename))
    print('Number of images :'+str(len(lst_imgs)))

    nb_rows = 10
    nb_cols = 30   
    n_size = 256

    nb_img_batch = nb_cols * nb_rows

    n_batch = len(lst_imgs) // nb_img_batch
    if len(lst_imgs) % nb_img_batch != 0:
        n_batch += 1

    # Torch io transforms for rescaling and resizing.
    rescale = tio.RescaleIntensity(out_min_max=(0, 1))
    resize = tio.CropOrPad(n_size)
    transforms = tio.Compose([rescale, resize])

    for b in range(n_batch):
        grids = []
        print("Batch", b)
        for i in range(nb_img_batch):
            n = b * nb_img_batch + i
            if n >= len(lst_imgs):
                slice = torch.zeros(1, n_size, n_size)
            else:
                print(str(i) + " : " + lst_imgs[b * nb_img_batch + i].split("/")[-1].split('.')[0])
                img = tio.ScalarImage(lst_imgs[n])[tio.DATA]
                img = transforms(img)
                slice = TF.rotate(img[..., int(img.shape[-1] / 2)], 90, expand=True)
            grids.append(slice)
                
        grids = torchvision.utils.make_grid(grids, nrow=nb_cols)
        grids = grids.permute(1,2,0).numpy()
        plt.imshow(grids)
        plt.title("Batch " + str(b))
        plt.show()
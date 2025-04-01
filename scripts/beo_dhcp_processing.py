import pandas as pd
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processing of dHCP fetal dataset')
    parser.add_argument('-i', '--input', help='Input tsv dhcp file', type=str, required = True)
    parser.add_argument('-b', '--bids_dir', help='Source BIDS directory', type=str, required = True)
    parser.add_argument('-o', '--output_bids_dir', help='Ouput new derivatives BIDS directory', type=str, required = True)    
    parser.add_argument('-t', '--template_dir', help='Age-specific dHCP template directory', type=str, required = True)
    args = parser.parse_args()    

    # Read the tsv data file
    df = pd.read_csv(args.input, sep='\t')

    # Check if the output directory exists
    if not os.path.exists(args.output_bids_dir):
        os.makedirs(args.output_bids_dir)

    # Loop over the subjects
    for index, row in df.iterrows():
        subject = row['subject_id']
        session = row['session_id']
        age_at_scan = row['scan_age']
        data_path = os.path.join(args.bids_dir, 'derivatives', subject, session, 'anat')
        image_filename = subject+'_'+session+'_desc-restore_T2w.nii.gz'   

        # Create directory for the subject
        output_subject_dir = os.path.join(args.output_bids_dir, subject, session, 'anat')
        if not os.path.exists(output_subject_dir):
            os.makedirs(output_subject_dir)

        # Remove bias field using N4BiasFieldCorrection
        output_N4_filename = os.path.join(output_subject_dir, os.path.basename(image_filename).replace('.nii.gz','_N4.nii.gz'))    
        cmd_line = 'N4BiasFieldCorrection -d 3 -i '+os.path.join(data_path, image_filename)+' -o '+output_N4_filename
        print(cmd_line)
        #os.system(cmd_line)

        # Rigid registration on age-specific dHCP template
        # Find for the current age_at_scan the closest integer age in the range [21,36]
        template_age = int(age_at_scan)
        if age_at_scan < 21:
            template_age = 21
        if age_at_scan > 36:
            template_age = 36
        
        print(subject, session, age_at_scan, template_age)

        template_image = os.path.join(args.template_dir, 'transformed-t2-t'+str(template_age)+'.00.nii.gz')
        output_filename = os.path.join(output_subject_dir, os.path.basename(output_N4_filename).replace('.nii.gz','_ants'))
        cmd_line = 'antsRegistration -d 3 -m MI['+template_image+','+output_N4_filename+',1,32,Regular,0.25] -t Rigid[0.1] -c [500x250x100,1e-6,10] -s 4x2x1 -f 4x2x1 -u '
        cmd_line+= ' -o ['+output_filename+', '+output_filename+'.nii.gz]'
        print(cmd_line)
        os.system(cmd_line)


import argparse
import SimpleITK as sitk
import os
from Preprocessor import Preprocessor



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classification Training.")
    parser.add_argument('-i',"--image-dir", default=None, required=True,type=str, help="image dir path")
    parser.add_argument('-o',"--output-dir", default=None, type=str, help="path to save outputs")
    parser.add_argument('-b',"--bias-correction", default=True, type=bool, help="enable bias correction")
    # parser.add_argument('-u',"--unsharp-masking", default=False, type=bool, help="enable unsharp masking")
    parser.add_argument('-d',"--denoising", default='bilateral', type=str, help="choose your denoising method")
    parser.add_argument('-c',"--clache", default=False, type=bool)

    # /Volumes/Elements/DATASET/IPMN_images_masks/t1/images
    
    args = parser.parse_args()
    image_dir = args.image_dir
    output_dir = args.output_dir

    proprocessor = Preprocessor()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    for name in os.listdir(image_dir):
        if name.endswith('.nii.gz') and not name.startswith('.'):
            try:
                save_path = os.path.join(output_dir,name)
                print(save_path)
                if os.path.exists(save_path):
                    continue
                image_path = os.path.join(image_dir,name)
                raw_image = sitk.ReadImage(image_path, sitk.sitkFloat32)
                proprocessor.set_image(raw_image)
                # Start preprocessing
                if args.bias_correction is True:
                    print('bias correction applied')
                    proprocessor.n4_bias_correction()
                else:
                    print('bias correction skipped')

                if args.denoising =='None':
                    print('denoising skipped')
                else:
                    match args.denoising:
                        case 'gaussian':
                            print('gaussian denoising applied')
                            proprocessor.bilateral_denoising()
                        case 'bilateral':
                            print('bilateral denoising applied')
                            proprocessor.bilateral_denoising()
                
                if args.clache is True:
                    print('clache applied')
                    proprocessor.clache()
                else:
                    print('clache skipped')

                image_pre = proprocessor.get_image()
                sitk.WriteImage(image_pre, save_path+'.nii.gz')
                print(name,'preprocessing finished' )
            except Exception as e:
                print(f'Error occured when processing {name}: {e}')
        
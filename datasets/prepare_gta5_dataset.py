import os
import glob
from PIL import Image


def load_resized_img(path):
    return Image.open(path).convert('RGB').resize((256, 256))


def process_cityscapes(leftImg8bit_dir, output_dir, phase):
    save_phase = 'test' if phase == 'val' else 'train'
    savedir = os.path.join(output_dir, save_phase)
    os.makedirs(savedir + 'B', exist_ok=True)
    print("Directory structure prepared at %s" % output_dir)

    photo_expr = os.path.join(leftImg8bit_dir, phase) + "/*/*_leftImg8bit.png"
    photo_paths = glob.glob(photo_expr)
    photo_paths = sorted(photo_paths)

    for i, photo_path in enumerate(photo_paths):
        photo = load_resized_img(photo_path)

        # data for cyclegan where the two images are stored at two distinct directories
        savepath = os.path.join(savedir + 'B', "%d_B.jpg" % i)
        photo.save(savepath, format='JPEG', subsampling=0, quality=100)


def process_gta5_images(gta5Image_dir, output_dir):
    savedir = os.path.join(output_dir, 'train')
    os.makedirs(savedir + 'A', exist_ok=True)
    print("Directory structure prepared at %s" % output_dir)

    gta5_expr = os.path.join(gta5Image_dir, '') + "/*/*.png"
    gta5_paths = glob.glob(gta5_expr)
    gta5_paths = sorted(gta5_paths)

    for i, gta5_path in enumerate(gta5_paths):
        figure = load_resized_img(gta5_path)

        savepath = os.path.join(savedir + 'A', "%d_A.jpg" % i)
        figure.save(savepath, format='JPEG', subsampling=0, quality=100)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gta5Image_dir', type=str, required=False,
                        default='./gta5Images',
                        help='Path to the gta5 image directory.')
    parser.add_argument('--leftImg8bit_dir', type=str, required=False,
                        default='./leftImg8bit',
                        help='Path to the Cityscapes image directory.')
    parser.add_argument('--output_dir', type=str, required=False,
                        default='./gta5',
                        help='Directory the output images will be written to.')

    opt = parser.parse_args()

    print('Preparing Cityscapes Dataset for train phase')
    process_cityscapes(opt.leftImg8bit_dir, opt.output_dir, "train")

    print('Preparing GTA5 Dataset for train and val phase')
    process_gta5_images(opt.gta5Image_dir, opt.output_dir)

    print('Done')
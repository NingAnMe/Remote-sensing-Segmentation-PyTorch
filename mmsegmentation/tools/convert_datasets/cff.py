import argparse
import os
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to TrainIds')
    parser.add_argument('cff_path', help='cityscapes data path')
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    x = os.listdir(os.path.join(args.cff_path, 'img_train'))
    y = [i.replace('jpg', 'png') for i in x]
    x_train, x_s, y_train, y_s = train_test_split(x, y, test_size=0.4)
    x_val, x_test, y_val, y_test = train_test_split(x_s, y_s, test_size=0.5)

    for d in ['train', 'val', 'test']:
        img_path = os.path.join(args.cff_path, 'img', d)
        label_path = os.path.join(args.cff_path, 'label', d)
        for d_path in [img_path, label_path]:
            if not os.path.isdir(d_path):
                os.makedirs(d_path)
    for x, y in zip(x_train, y_train):

        os.symlink(os.path.join(args.cff_path, 'img_train', x), os.path.join(args.cff_path, 'img', 'train'))
        os.symlink(os.path.join(args.cff_path, 'lab_train', x), os.path.join(args.cff_path, 'label', 'train'))
    
    for x, y in zip(x_val, y_val):
        os.symlink(os.path.join(args.cff_path, 'img_train', x), os.path.join(args.cff_path, 'img', 'val'))
        os.symlink(os.path.join(args.cff_path, 'lab_train', x), os.path.join(args.cff_path, 'label', 'val'))
    
    for x, y in zip(x_test, y_test):
        os.symlink(os.path.join(args.cff_path, 'img_train', x), os.path.join(args.cff_path, 'img', 'test'))
        os.symlink(os.path.join(args.cff_path, 'lab_train', x), os.path.join(args.cff_path, 'label', 'test'))


if __name__ == '__main__':
    main()

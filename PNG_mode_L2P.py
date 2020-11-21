#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020-11-20 16:11
# @Author  : NingAnMe <ninganme@qq.com>
import os
import argparse
from PIL import Image
import numpy as np


def get_palette(img_png, class_num=0):
    """
    根据类别的不同，按一定的规律生成调色板
    :param img_png:
    :param class_num: 0-256
    :return: 一维列表， 256 * 3 个值
    """
    if not class_num:
        img = Image.open(img_png)
        class_num = len(np.unique(img))
        img.close()

    print('class_num : {}'.format(class_num))

    palette = np.zeros((256, 3), dtype=np.uint8)
    for k in range(0, class_num):
        label = k
        i = 0
        while label:  # 按一定规则移位产生调色板
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))  # >>为二进制右移
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette.flatten().tolist()


def png_L2P(file_in, palette=None, file_out=None):
    """
    将Image L模式的图片转为P模式（伪彩图）的图片
    :param file_in: 输入文件路径
    :param palette: 如果为png文件，则拷贝其palette
    :param file_out: 输出文件路径
    :return:
    """
    if palette:
        if os.path.splitext(palette)[1].lower() == '.png':
            p = Image.open(palette).getpalette()
        else:
            raise ValueError('--palette : {}'.format(palette))
    else:
        p = get_palette(file_in)
    # 增加调色板
    img_in = Image.open(file_in)
    img_p = img_in.convert("P")
    img_p.putpalette(p)

    if not file_out:
        file_out = file_in + '_p.png'
    img_p.save(file_out)
    print('>>> : {}'.format(file_out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--input', '-i', help='转换图像或者转换图像的目录', required=True)
    parser.add_argument('--palette', '-p', help='palette模板或者带有palette模板的png图像', required=False)
    parser.add_argument('--output', '-o', help='输出图像', required=False)
    args = parser.parse_args()

    input_ = args.input
    palette_ = args.palette
    output_ = args.output
    if os.path.isfile(input_):
        png_L2P(input_, palette=palette_, file_out=output_)
    elif os.path.isdir(input_):
        for filename in os.listdir(input_):
            png_L2P(os.path.join(input_, filename), palette=palette_, file_out=output_)
    else:
        args.print_help()

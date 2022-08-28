#!/usr/bin/env python
import argparse
from domain.function import adaptive_instance_normalization
from pathlib import Path
from PIL import Image
import random
import torch
import torchvision.transforms


def input_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(torchvision.transforms.Resize(size))
    if crop != 0:
        transform_list.append(torchvision.transforms.CenterCrop(crop))
    transform_list.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(transform_list)
    return transform

def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def stylize_image(vgg, decoder, original_img, num_styles, label_path, image_size, alpha=1):
    # disable decompression bomb errors
    Image.MAX_IMAGE_PIXELS = None
    skipped_imgs = []

    # collect style files
    extensions = ['png', 'jpeg', 'jpg']

    style_dir = Path(label_path)
    style_dir = style_dir.resolve()


    styles = []
    for ext in extensions:
        styles += list(style_dir.rglob('*.' + ext))
    assert len(styles) > 0, 'No images with specified extensions found in style directory' + style_dir


    styles = sorted(styles)

    style_tf = input_transform(image_size, image_size)
    content = original_img
    content = content.cuda().unsqueeze(0)

    # # actual style transfer as in AdaIN
    # outputs = []
    # for style_path in random.sample(styles, num_styles):
    #     style_img = Image.open(style_path).convert('RGB')
    #     style = style_tf(style_img)
    #     style = style.cuda().unsqueeze(0)
    #     with torch.no_grad():
    #         output = style_transfer(vgg, decoder, content, style,
    #                                 alpha)
    #     outputs.append(output)
    #
    # tmp_img = random.sample(outputs,1)
    # stylize_img = tmp_img[0].squeeze()


    # actual style transfer as in AdaIN  '''S2'''
    style_path = random.sample(styles, 1)
    style_img = Image.open(style_path[0]).convert('RGB')
    style = style_tf(style_img)
    style = style.cuda().unsqueeze(0)
    with torch.no_grad():
        output = style_transfer(vgg, decoder, content, style,
                                alpha)

    return output

if __name__ == '__main__':
    stylize_image()



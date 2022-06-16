#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import json
import os
import os.path as osp

import numpy as np
import PIL.Image
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import labelme

def main():
    labels_file = "/home/imad/Downloads/Datasets/Training_2/new_out/label.txt"
    in_dir = "/home/imad/Downloads/Datasets/Training_2/new_out/RGBs"
    out_dir = "/home/imad/Downloads/Datasets/Training_2/new_out/output"

    if osp.exists(out_dir):
        print('Output directory already exists:',out_dir)
        quit(1)
    os.makedirs(out_dir)
    os.makedirs(osp.join(out_dir, 'JPEGImages'))
    os.makedirs(osp.join(out_dir, 'SegmentationClass'))
    os.makedirs(osp.join(out_dir, 'SegmentationClassPNG'))
    os.makedirs(osp.join(out_dir, 'SegmentationClassVisualization'))
    print('Creating dataset:',out_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(labels_file).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_background_'
        class_names.append(class_name)
    class_names = tuple(class_names)
    print('class_names:', class_names)
    out_class_names_file = osp.join(out_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)

    colormap = labelme.utils.label_colormap(255)

    for label_file in glob.glob(osp.join(in_dir, '*.json')):
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            out_img_file = osp.join(
                out_dir, 'JPEGImages', base + '.jpg')
            out_lbl_file = osp.join(
                out_dir, 'SegmentationClass', base + '.npy')
            out_png_file = osp.join(
                out_dir, 'SegmentationClassPNG', base + '.png')
            out_viz_file = osp.join(
                out_dir, 'SegmentationClassVisualization', base + '.jpg')

            data = json.load(f)

            img_file = osp.join(osp.dirname(label_file), data['imagePath'])
            img = np.asarray(PIL.Image.open(img_file))
            #print(img.shape)
            #img =  cv2.resize(img, (512,512), interpolation=cv2.INTER_NEAREST)
            #print(img.shape)
            PIL.Image.fromarray(img).save(out_img_file)

            lbl = labelme.utils.shapes_to_label(
                img_shape=img.shape,
                shapes=data['shapes'],
                label_name_to_value=class_name_to_id,
            )
            labelme.utils.lblsave(out_png_file, lbl)
            lbl_oualid = lbl.reshape(lbl.shape[0],lbl.shape[1],1)
            lbl_oualid= lbl_oualid.astype(np.float32)
            #lbl_oualid= cv2.resize(lbl_oualid, (512,512), interpolation=cv2.INTER_NEAREST)
            #lbl_oualid = lbl_oualid.reshape(img_mask.shape[0],img_mask.shape[1],1)
            np.save(out_lbl_file, lbl_oualid)

            viz = labelme.utils.draw_label(
                lbl, img, class_names, colormap=colormap)
            PIL.Image.fromarray(viz).save(out_viz_file)


if __name__ == '__main__':
    main()

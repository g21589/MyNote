# -*- coding: utf-8 -*-

import os
import shutil
import random

rseed            = 4321
validation_split = 0.3
input_dir        = os.path.normpath('./Datasets/flower')
output_dir       = os.path.normpath('./Datasets/flower_split_%.2f_%d' % (validation_split, rseed))

random.seed(rseed)

for root, labels, imgs in os.walk(input_dir):
    print('Path:', root)
    print(' |- Dir:', labels)
    print(' |- Img:', imgs[0:3])
    n_imgs = len(imgs)
    if n_imgs > 0:
        random.shuffle(imgs)
        train_size = int(float(n_imgs) * (1 - validation_split))
        train_imgs = imgs[0:train_size]
        val_imgs = imgs[train_size:]
        print(n_imgs, len(train_imgs), len(val_imgs), len(train_imgs) + len(val_imgs))
        
        # TODO: Split def & ref
        
        for img in train_imgs:
            dst_dir = root.replace(input_dir, os.path.join(output_dir, 'Train'))
            if not os.path.isdir(dst_dir):
                os.makedirs(dst_dir)
            src = os.path.join(root, img)
            dst = os.path.join(dst_dir, img)
            shutil.copyfile(src, dst)
            
        for img in val_imgs:
            dst_dir = root.replace(input_dir, os.path.join(output_dir, 'Valid'))
            if not os.path.isdir(dst_dir):
                os.makedirs(dst_dir)
            src = os.path.join(root, img)
            dst = os.path.join(dst_dir, img)
            shutil.copyfile(src, dst)
        
    print('---------------------------------------')
    
print('Finished!!')

import argparse
import glob
import os
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    # path of prediect images , and ground truth images
    parser.add_argument('--pred_path', default="")
    parser.add_argument('--real_path', default="")
    args = parser.parse_args()
    
    pred_path = args.pred_path
    real_path = args.real_path

    # get all images under paths, sorted is to make the order same
    pred_img_lists = sorted(glob.glob(os.path.join(pred_path,"*.png")))
    real_img_lists = sorted(glob.glob(os.path.join(real_path,"*.png")))
    
    #  predicted images and ground truth images should have the same length
    assert len(pred_img_lists)==len(real_img_lists), 'different number of png images for predicted and ground truth'

    # init the l2 distance
    l2_rgb = 0

    
    for i,img in enumerate(pred_img_lists):
        # read images
        pred = cv2.imread(img)
        gt = cv2.imread(real_img_lists[i])
        gt = cv2.resize(gt,(256,256))
    
        
        # cal the l2 distance
        l2 = np.sum((gt-pred)**2)

        l2_rgb += l2
    # print out results
    print(f"L2 distance: {l2/(i+1)}")


if __name__=='__main__':
    main()

        
        
        

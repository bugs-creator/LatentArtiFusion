from image_similarity_measures.evaluate import evaluation
import argparse
import glob
import os
import cv2
import numpy as np


def main():
    # get path as input
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', required=True, help="path to reconstructed images")
    parser.add_argument('--real_path', required=True, help="path to ground truth images")
    args = parser.parse_args()
    
    pred_path = args.pred_path
    real_path = args.real_path

    # get all png images under given paths
    pred_lists = sorted(glob.glob(os.path.join(pred_path,"*.png")))
    gt_lists = sorted(glob.glob(os.path.join(real_path,"*.png")))

    # they should have the same number of images, otherwise, it is considered as wrong path given
    assert len(gt_lists) == len(pred_lists)
    
 
    # init all, use a dict to record

    metrics=["ssim", "psnr","fsim"]
    d = {}
    for m in metrics:
        d[m] = 0
    for i,imgP in enumerate(gt_lists):
        # print(i)
        e = evaluation(org_img_path=imgP, 
           pred_img_path=pred_lists[i], 
           metrics=metrics)
        for m in metrics:
            d[m] += e[m]
            
    # print out results
    for m in metrics:
        print("{} : {}".format(m,d[m]/(i+1)))
        
if __name__ == '__main__':
    main()

    
    
        

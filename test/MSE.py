import argparse
import glob
import os
import cv2
import numpy as np


def main():
    # read path to predicted images, mask and ground truth from paths
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', default="/root/autodl-tmp/histology/predicted/aa_cyclegan/test_latest/images")
    parser.add_argument('--mask_path', default="/root/autodl-tmp/histology/predicted/aa_cyclegan/test_latest/images")
    parser.add_argument('--real_path', default="/root/autodl-tmp/histology/predicted/aa_cyclegan/test_latest/images")
    args = parser.parse_args()
    
    pred_path = args.pred_path
    real_path = args.real_path
    mask_path = args.mask_path

    # get all predicted images, mask and ground truth from paths under the given paths
    pred_img_lists = sorted(glob.glob(os.path.join(pred_path,"*.png")))
    real_img_lists = sorted(glob.glob(os.path.join(real_path,"*.png")))
    mask_img_lists = sorted(glob.glob(os.path.join(mask_path,"*.png")))
    
    # they should have the same length
    assert len(pred_img_lists)==len(real_img_lists), "Different Number of inputs"
    
    mse_rgb = 0


    # loop to get the result
    for i,img in enumerate(pred_img_lists):
        pred = cv2.imread(img)
        gray_pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
        
        gt = cv2.imread(real_img_lists[i])
        gt = cv2.resize(gt,(256,256))
     
        
        mask = cv2.imread(mask_img_lists[i])
        mask = cv2.resize(mask,(256,256))
      
        num_nonzero = np.count_nonzero(mask_gray, axis=0)
        count = 0
        for l in num_nonzero:
            count += l
        assert count!=0
            
        

        
        mse = np.sum((gt-pred)**2)/count
        # print(mask_gray.shape)

    mse_rgb += mse
        
    print("MSE: ",mse_rgb/(i+1))


if __name__=='__main__':
    main()

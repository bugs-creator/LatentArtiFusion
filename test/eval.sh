export RECONST_IMG_PATH='TBC'
export GT_IMG_PATH='./data/histology'
export MASK_PATH='./test_data/histology_mask'

python ./L2.py --pred_path=$RECONST_IMG_PATH --real_path=$GT_IMG_PATH
python ./MSE.py --pred_path=$RECONST_IMG_PATH --real_path=$GT_IMG_PATH --mask_path=$MASK_PATH
python ./cal_imgSim.py --pred_path=$RECONST_IMG_PATH --real_path=$GT_IMG_PATH 

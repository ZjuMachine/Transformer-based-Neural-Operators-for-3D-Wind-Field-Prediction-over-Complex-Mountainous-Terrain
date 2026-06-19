export CUDA_VISIBLE_DEVICES=0

python main_evaluation_windey.py \
--cfd_model=Transolver \
--data_dir /dell_mnt/windey_data/cutted \
--save_dir /dell_mnt/yuzhouz/Transolver/Windey/dataset/windey_precessed_cutted \
--model_name cutted_with_sdf_pp \
--slice_num=64 \
--fold_i=0

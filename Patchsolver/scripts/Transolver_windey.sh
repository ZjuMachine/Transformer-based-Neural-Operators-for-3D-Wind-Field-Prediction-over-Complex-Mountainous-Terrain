export CUDA_VISIBLE_DEVICES=1

# python main.py \
# --cfd_model=Transolver \
# --preprocessed=1 \
# --data_dir /dell_mnt/windey_data/cutted \
# --save_dir /dell_mnt/yuzhouz/Transolver/Windey/dataset/windey_precessed_cutted \
# --model_name cutted_with_sdf_pp \
# --slice_num=64 \
# --batch_size=1


python main.py \
--cfd_model=Transolver \
--preprocessed=0 \
--data_dir Car-Design-ShapeNetCar/dataset/xiaopeng \
--save_dir Car-Design-ShapeNetCar/dataset/xiaopeng_handle \
--model_name cutted_with_sdf_pp \
--slice_num=64 \
--batch_size=1


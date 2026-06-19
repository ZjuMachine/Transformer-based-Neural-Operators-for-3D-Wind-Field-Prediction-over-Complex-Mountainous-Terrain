export CUDA_VISIBLE_DEVICES=1

python evaluation_new.py \
--cfd_model=Transolver \
--preprocessed=1 \
--data_dir ../all_dataset_1/vkt_files \
--save_dir ../all_dataset_1/np_files \
--model_name test0713 \
--slice_num=64 \
--nb_epochs=100 \
--fold_i=0

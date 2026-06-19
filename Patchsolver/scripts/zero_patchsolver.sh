export CUDA_VISIBLE_DEVICES=2

# fold id 10 zero-shot
python evaluation_new.py \
--cfd_model=Patch_solver \
--preprocessed=1 \
--data_dir ../all_dataset_1/vkt_files \
--save_dir ../all_dataset_1/np_files \
--model_name test0723-1 \
--slice_num=16 \
--tesy_mode=2 \
--fold_id=0 \
--test_task_name zero_shot \
--nb_epochs=100 



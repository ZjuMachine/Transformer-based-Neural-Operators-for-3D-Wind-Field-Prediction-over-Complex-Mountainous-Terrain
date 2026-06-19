export CUDA_VISIBLE_DEVICES=0

##############################  0.1% ############################## 

python main.py \
--cfd_model=Patch_solver \
--preprocessed=1 \
--data_dir ../all_dataset_1/vkt_files \
--save_dir ../all_dataset_1/np_files \
--model_name test0713 \
--slice_num=16 \
--batch_size=1 \
--nb_epochs=100 \
--use_sparse \
--sparse_ratio=100


# test
python evaluation_new.py \
--cfd_model=Patch_solver \
--preprocessed=1 \
--data_dir ../all_dataset_1/vkt_files \
--save_dir ../all_dataset_1/np_files \
--model_name test0713 \
--slice_num=16 \
--nb_epochs=100 \
--use_sparse \
--sparse_ratio=100


# zero-shot 
python evaluation_new.py \
--cfd_model=Patch_solver \
--preprocessed=1 \
--data_dir ../all_dataset_1/vkt_files \
--save_dir ../all_dataset_1/np_files \
--model_name test0713 \
--slice_num=16 \
--nb_epochs=100 \
--tesy_mode=2 \
--test_task_name zero_shot \
--fold_i=0 \
--use_sparse \
--sparse_ratio=100


##############################  1% ############################## 

python main.py \
--cfd_model=Patch_solver \
--preprocessed=1 \
--data_dir ../all_dataset_1/vkt_files \
--save_dir ../all_dataset_1/np_files \
--model_name test0713-1000 \
--slice_num=16 \
--batch_size=1 \
--nb_epochs=100 \
--use_sparse \
--sparse_ratio=1000

# test
python evaluation_new.py \
--cfd_model=Patch_solver \
--preprocessed=1 \
--data_dir ../all_dataset_1/vkt_files \
--save_dir ../all_dataset_1/np_files \
--model_name test0713-1000 \
--slice_num=16 \
--nb_epochs=100 \
--use_sparse \
--sparse_ratio=1000


# zero-shot 
python evaluation_new.py \
--cfd_model=Patch_solver \
--preprocessed=1 \
--data_dir ../all_dataset_1/vkt_files \
--save_dir ../all_dataset_1/np_files \
--model_name test0713-1000 \
--slice_num=16 \
--nb_epochs=100 \
--tesy_mode=2 \
--test_task_name zero_shot \
--fold_i=0 \
--use_sparse \
--sparse_ratio=1000

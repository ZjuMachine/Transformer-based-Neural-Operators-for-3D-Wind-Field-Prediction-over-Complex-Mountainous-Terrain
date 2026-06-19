export CUDA_VISIBLE_DEVICES=1

python evaluation_new.py \
--cfd_model=Patch_solver \
--preprocessed=1 \
--data_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_store_2 \
--save_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_np_2 \
--model_name test0723-1 \
--slice_num=16 \
--nb_epochs=100 
export CUDA_VISIBLE_DEVICES=2

python evaluation_new.py \
--cfd_model=Transolver \
--preprocessed=1 \
--data_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_store_2 \
--save_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_np_2 \
--model_name test0713 \
--slice_num=64 \
--nb_epochs=100 \
--tesy_mode=2 \
--test_task_name zero_shot \
--fold_i=0

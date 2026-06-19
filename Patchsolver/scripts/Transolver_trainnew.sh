export CUDA_VISIBLE_DEVICES=2

python main.py \
--cfd_model=Transolver \
--preprocessed=1 \
--data_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_store_2 \
--save_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_np_2 \
--model_name test0713 \
--slice_num=64 \
--batch_size=1 \
--nb_epochs=100 



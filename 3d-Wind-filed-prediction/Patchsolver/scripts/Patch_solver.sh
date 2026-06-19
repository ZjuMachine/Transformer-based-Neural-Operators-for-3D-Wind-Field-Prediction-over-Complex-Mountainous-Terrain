export CUDA_VISIBLE_DEVICES=0

# python main.py \
# --cfd_model=Patch_solver \
# --preprocessed=1 \
# --data_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_store_2 \
# --save_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_np_2 \
# --model_name test0723-1 \
# --slice_num=16 \
# --batch_size=1 \
# --nb_epochs=100 


python main.py \
--cfd_model=Patch_solver \
--preprocessed=1 \
--data_dir ../all_dataset_1/vkt_files \
--save_dir ../all_dataset_1/np_files \
--model_name test0723-normgate \
--slice_num=16 \
--batch_size=1 \
--nb_epochs=100 

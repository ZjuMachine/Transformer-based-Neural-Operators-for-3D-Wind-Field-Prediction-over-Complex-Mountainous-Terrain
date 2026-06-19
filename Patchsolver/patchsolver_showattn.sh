export CUDA_VISIBLE_DEVICES=0

python main_evaluation_showattention.py \
--cfd_model=Patch_solver \
--preprocessed=1 \
--data_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_store_2_vtkmodified \
--save_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_np_2 \
--model_name test0723-1 \
--slice_num=16 \
--nb_epochs=100 \
--attn_vis \
--attn_heights 10 \
--attn_per_height 1 \
--attn_tol 5.0 \
--attn_topk_tokens 8


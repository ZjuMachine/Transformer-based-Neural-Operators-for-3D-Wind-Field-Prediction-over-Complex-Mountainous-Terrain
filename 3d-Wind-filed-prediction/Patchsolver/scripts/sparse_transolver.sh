export CUDA_VISIBLE_DEVICES=0
##############################  100############################## 
# train
# python main.py \
# --cfd_model=Transolver \
# --preprocessed=1 \
# --data_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_store_2 \
# --save_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_np_2 \
# --model_name test0713 \
# --slice_num=64 \
# --batch_size=1 \
# --nb_epochs=100 \
# --use_sparse 


# test
# python evaluation_new.py \
# --cfd_model=Transolver \
# --preprocessed=1 \
# --data_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_store_2 \
# --save_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_np_2 \
# --model_name test0713 \
# --slice_num=64 \
# --nb_epochs=100 \
# --fold_i=0 \
# --use_sparse 


# zero-shot test
# python evaluation_new.py \
# --cfd_model=Transolver \
# --preprocessed=1 \
# --data_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_store_2 \
# --save_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_np_2 \
# --model_name test0713 \
# --slice_num=64 \
# --nb_epochs=100 \
# --tesy_mode=2 \
# --test_task_name zero_shot \
# --fold_i=0 \
# --use_sparse 





##############################  1000############################## 
# train
# python main.py \
# --cfd_model=Transolver \
# --preprocessed=1 \
# --data_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_store_2 \
# --save_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_np_2 \
# --model_name test0713-1000 \
# --slice_num=64 \
# --batch_size=1 \
# --nb_epochs=100 \
# --use_sparse 


# # test
# python evaluation_new.py \
# --cfd_model=Transolver \
# --preprocessed=1 \
# --data_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_store_2 \
# --save_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_np_2 \
# --model_name test0713-1000 \
# --slice_num=64 \
# --nb_epochs=100 \
# --fold_i=0 \
# --use_sparse 


# # zero-shot test
# python evaluation_new.py \
# --cfd_model=Transolver \
# --preprocessed=1 \
# --data_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_store_2 \
# --save_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_np_2 \
# --model_name test0713-1000 \
# --slice_num=64 \
# --nb_epochs=100 \
# --tesy_mode=2 \
# --test_task_name zero_shot \
# --fold_i=0 \
# --use_sparse 







##############################  100--20251205 ############################## 
# train
python main.py \
--cfd_model=Transolver \
--preprocessed=1 \
--data_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_store_2 \
--save_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_np_2 \
--model_name test20251205 \
--slice_num=64 \
--batch_size=1 \
--nb_epochs=100 \
--use_sparse \
--sparse_ratio=100


# test
python evaluation_new.py \
--cfd_model=Transolver \
--preprocessed=1 \
--data_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_store_2 \
--save_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_np_2 \
--model_name test20251205 \
--slice_num=64 \
--nb_epochs=100 \
--fold_i=0 \
--use_sparse \
--sparse_ratio=100


# zero-shot test
python evaluation_new.py \
--cfd_model=Transolver \
--preprocessed=1 \
--data_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_store_2 \
--save_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_np_2 \
--model_name test20251205 \
--slice_num=64 \
--nb_epochs=100 \
--tesy_mode=2 \
--test_task_name zero_shot \
--fold_i=0 \
--use_sparse \
--sparse_ratio=100




##############################  1000--20251205 ############################## 
# # train
# python main.py \
# --cfd_model=Transolver \
# --preprocessed=1 \
# --data_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_store_2 \
# --save_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_np_2 \
# --model_name test20251205-1000 \
# --slice_num=64 \
# --batch_size=1 \
# --nb_epochs=100 \
# --use_sparse \
# --sparse_ratio=1000


# # test
# python evaluation_new.py \
# --cfd_model=Transolver \
# --preprocessed=1 \
# --data_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_store_2 \
# --save_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_np_2 \
# --model_name test20251205-1000 \
# --slice_num=64 \
# --nb_epochs=100 \
# --fold_i=0 \
# --use_sparse \
# --sparse_ratio=1000


# # zero-shot test
# python evaluation_new.py \
# --cfd_model=Transolver \
# --preprocessed=1 \
# --data_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_store_2 \
# --save_dir /mntlvlin20tb/yujzhang/Windy_yj/all_dataset_np_2 \
# --model_name test20251205-1000 \
# --slice_num=64 \
# --nb_epochs=100 \
# --tesy_mode=2 \
# --test_task_name zero_shot \
# --fold_i=0 \
# --use_sparse \
# --sparse_ratio=1000
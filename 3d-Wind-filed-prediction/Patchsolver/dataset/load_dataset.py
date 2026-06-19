import os
from dataset.dataset import get_datalist, get_datalist_windey


def get_samples_windey(root):
    folds = []
    # folds = [f'newcase{i}' for i in range(1, 19)]
    # all data
    for i in range(1, 4):
        # folds.append(f'jiepai-{i}')
        folds.append(f'chatou-{i}')
        # folds.append(f'taojiang-{i}')
    # # folds.pop(1)
    # folds = ['newcase10']
    samples = []
    
    # multi directions
    for fold in folds:
        fold_samples = []
        files = os.listdir(os.path.join(root, fold, 'result'))
        # file就是角度名称
        for file in files:
            path = os.path.join(root, os.path.join(fold, 'result', file, 'VTK'))
            if os.path.isdir(path):
                fold_samples.append(os.path.join(fold, 'result', file, 'VTK'))
        samples.append(fold_samples)
    print(samples)
    # print(len(samples))

    return samples  # [['newcase10/result/0.0/VTK']，['newcase11/result/0.0/VTK']]，need to calculate number of  samples


def get_samples_windey_trainval(args,root):
    folds = []
    # folds = [f'newcase{i}' for i in range(1, 19)]
    # all data
    for i in range(1, 4):
        # folds.append(f'jiepai-{i}')
        folds.append(f'chatou-{i}')
        # folds.append(f'taojiang-{i}')
    # # folds.pop(1)
    # folds = ['newcase10']
    samples = []
    vallst=[]
    
    for fold in folds:
        fold_samples = []
        fold_samples_val=[]
        files = os.listdir(os.path.join(root, fold, 'result'))
  
        for file in files:
            path = os.path.join(root, os.path.join(fold, 'result', file, 'VTK'))
            if file=='0.0':
                if os.path.isdir(path):
                    fold_samples_val.append(os.path.join(fold, 'result', file, 'VTK'))
            else:
                # except 0
                if os.path.isdir(path):
                    fold_samples.append(os.path.join(fold, 'result', file, 'VTK'))
        samples.append(fold_samples)
        vallst.append(fold_samples_val)
    
    tesy_mode=args.tesy_mode 
    if tesy_mode==1:
        pass
    elif tesy_mode==2:
        rootnew='../all_dataset_zero_shot/vkt_files'

        folds_new=[]
        vallst=[]
        for i in range(1, 2):
            folds_new.append(f'chenzhou-1')
            # folds_new.append(f'chenzhou-2')
            # folds_new.append(f'daguping')
            # folds_new.append(f'hengdong')
            # folds_new.append(f'loudi')
        for fold in folds_new:
            fold_samples_val=[]
            files = os.listdir(os.path.join(rootnew, fold, 'result'))
            for file in files:
                path = os.path.join(rootnew, os.path.join(fold, 'result', file, 'VTK'))
                if os.path.isdir(path):
                    fold_samples_val.append(os.path.join(fold, 'result', file, 'VTK'))
            vallst.append(fold_samples_val)
        
    print(samples)
    print(vallst)
    # print(len(samples))

    return samples,vallst  # [['newcase10/result/0.0/VTK']，['newcase11/result/0.0/VTK']]，need to calculate number of  samples

def load_train_val_fold_windey(args, preprocessed):

    # samples = get_samples_windey(args.data_dir)
    samples,val_samples = get_samples_windey_trainval(args,args.data_dir)
    trainlst = []
    vallst=[]

    
    for i in range(len(samples)):
        # if i == args.fold_id:
        #     continue
        trainlst += samples[i]
    for i in range(len(val_samples)):
        # if i == args.fold_id:
        #     continue
        vallst += val_samples[i]


    print(f"trainlst: {trainlst}")
    print(f"vallst: {vallst}")
    print('train data length',len(trainlst))
    print('val data length',len(vallst))

    if preprocessed:
        print("use preprocessed data")
    print("loading data")

    train_dataset, coef_norm = get_datalist_windey(args,args.data_dir, trainlst, norm=True, savedir=args.save_dir,
                                            preprocessed=preprocessed)
    if args.tesy_mode==2:
        rootnew='../all_dataset_zero_shot/vkt_files'
        rootnew_save='../all_dataset_zero_shot/np_files'
        val_dataset = get_datalist_windey(args,rootnew, vallst, coef_norm=coef_norm, savedir=rootnew_save,
                                preprocessed=preprocessed)
    else:
        val_dataset = get_datalist_windey(args,args.data_dir, vallst, coef_norm=coef_norm, savedir=args.save_dir,
                                preprocessed=preprocessed)
    print("load data finish")
    return train_dataset, val_dataset, coef_norm


def load_train_val_fold_file_windey(args, preprocessed):


    samples,val_samples = get_samples_windey_trainval(args,args.data_dir)
    trainlst = []
    vallst=[]

    for i in range(len(samples)):
        trainlst += samples[i]
    for i in range(len(val_samples)):
        vallst += val_samples[i]


    if preprocessed:
        print("use preprocessed data")
    print("loading data")
    train_dataset, coef_norm = get_datalist_windey(args,args.data_dir, trainlst, norm=True, savedir=args.save_dir,
                                            preprocessed=preprocessed)
    if args.tesy_mode==2:
        rootnew='../all_dataset_zero_shot/vkt_files'
        rootnew_save='../all_dataset_zero_shot/np_files'
        val_dataset = get_datalist_windey(args,rootnew, vallst, coef_norm=coef_norm, savedir=rootnew_save,
                                preprocessed=preprocessed)
    else:
        val_dataset = get_datalist_windey(args,args.data_dir, vallst, coef_norm=coef_norm, savedir=args.save_dir,
                                preprocessed=preprocessed)
    print("load data finish")
    return train_dataset, val_dataset, coef_norm, vallst

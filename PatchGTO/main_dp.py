import argparse
# import yaml
import json
from types import SimpleNamespace

import time
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import init
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
import torch
import numpy as np
import os
from tensorboardX import SummaryWriter
from torch.nn import init, DataParallel
import h5py

# load
from utils.base import set_seed
from utils.shapenet_velocity_all import Car_Dataset_v

from PatchAeroGTO import AeroGTO

from utils.train import train, validate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', type=str, help='Path to config file')  # Change the default config file name if needed

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)  # Load JSON instead of YAML
    
    args = SimpleNamespace(**config)
    
    return args


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.MultiheadAttention):
        torch.nn.init.xavier_uniform_(m.in_proj_weight)
        if hasattr(m, 'in_proj_bias') and m.in_proj_bias is not None:
            m.in_proj_bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(m.out_proj.weight)
        if hasattr(m.out_proj, 'bias') and m.out_proj.bias is not None:
            m.out_proj.bias.data.fill_(0.01)

def gather_tensor(tensor):
    """
    Gathers tensors from all processes and reduces them by summing up.
    """
    # Ensure the tensor is on the same device as specified for the operation
    tensor = tensor.to(device)
    # All-reduce: Sum the tensors from all processes
    return tensor


def get_model(args):
    """ 
    "model": {
        "name": "Gamer_MSE_loss",
        "if_init": true,
        "N_block": 8,
        "state_size": 3,
        "state_embedding_dim": 128,
        "att_embedding_dim": 128,
        "n_head": 8,
        "n_token": 256
    },"""
    model = AeroGTO(
        N_block = args.model["N_block"], 
        state_size = args.model["state_size"],  
        state_embedding_dim = args.model["state_embedding_dim"], 
        att_embedding_dim = args.model["att_embedding_dim"],
        n_head = args.model["n_head"],
        n_token = args.model["n_token"]
        ).to(device)
    return model


def main(args):

    model = get_model(args).to(device)
    # model.load_state_dict(torch.load(args.resume_path)["state_dict"])
    
    if args.model["if_init"]:
        model.apply(init_weights)
    
    if args.train["if_multi_gpu"]:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)
    

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())    
    params = sum([np.prod(p.size()) for p in model_parameters])
    

    # load data
    train_dataset = Car_Dataset_v(
        data_path = args.dataset["data_path"],
        mode="train",
        N_target = args.dataset["train"]["N_target"], 
        E_target = args.dataset["train"]["E_target"], 
        adj_num = args.dataset["train"]["adj_num"],
        sample_times = args.dataset["train"]["sample_times"]
        )
    


    test_dataset = Car_Dataset_v( 
        data_path = args.dataset["data_path"],
        mode="test",
        N_target = args.dataset["train"]["N_target"], 
        E_target = args.dataset["train"]["E_target"], 
        adj_num= args.dataset["test"]["adj_num"]
        )

    
    train_dataloader = DataLoader(train_dataset,
                        batch_size=args.dataset["train"]["batchsize"], 
                        shuffle=args.dataset["train"]["shuffle"], 
                        num_workers=args.dataset["train"]["num_workers"],
                        )
    
    test_dataloader = DataLoader(test_dataset, 
                            batch_size=args.dataset["test"]["batchsize"], 
                            shuffle=args.dataset["test"]["shuffle"], 
                            num_workers=args.dataset["test"]["num_workers"],
                            )
    EPOCH = args.train["epoch"]
    warmup_epochs = 5
        
    print("#############")
    print("#params:", params)
    print(f"EPOCH: {EPOCH}")
    print(f"model name: {args.model['name']}")
    
    print(f"No. of train samples: {len(train_dataset)}, No. of test samples: {len(test_dataset)}")
    print(f"No. of train batches: {len(train_dataloader)}, No. of test batches: {len(test_dataloader)}")
    print("#############")
    
    with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
        file.write(f"No. of train samples: {len(train_dataset)}, No. of test samples: {len(test_dataset)}\n")
        file.write(f"No. of train batches: {len(train_dataloader)}, No. of test batches: {len(test_dataloader)}\n")
        file.write(f"Let's use {torch.cuda.device_count()} GPUs!\n")
        file.write(f"{args.name}, #params: {params}\n")
        file.write(f"EPOCH: {EPOCH}\n")
        file.write("*"*16 + "\n")
    
    log_dir = f"{args.save_path}/logs/{args.name}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    real_lr = float(args.train["lr"])
    optim = torch.optim.AdamW(model.parameters(), lr=real_lr)

    scheduler = CosineAnnealingLR(optim, T_max= EPOCH, eta_min = float(args.train["eta_min"]))
    

    sample = train_dataset[0]  
    input_test, _ = sample  
    M_test = input_test["node_pos"].size()[0] 
    print("node_pos:" + str(M_test) + "\n")

    min_loss = 100
    for epoch in range(EPOCH):
        start_time = time.time()
        train_error = train(args, model, train_dataloader, optim, device)
        end_time = time.time()
        

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0] 
        
        training_time = (end_time - start_time)
        current_lr = torch.tensor(current_lr, device=device)
        train_loss = torch.tensor(train_error['loss'], device=device)
        L2_v = torch.tensor(train_error['L2_v'], device=device)
        L2_v_norm = torch.tensor(train_error['L2_v_norm'], device=device)
        MSE_loss = torch.tensor(train_error['MSE_loss'], device=device)
        MSE_loss_norm = torch.tensor(train_error['MSE_loss_norm'], device=device)
        training_time = torch.tensor(training_time, device=device)
        
        writer.add_scalar('lr/lr', current_lr, epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('L2/train_L2_v', L2_v, epoch)
        writer.add_scalar('L2/train_L2_v_norm', L2_v_norm, epoch)
        writer.add_scalar('MSE/train_MSE_loss', MSE_loss, epoch)
        writer.add_scalar('MSE/train_MSE_loss_norm', MSE_loss_norm, epoch)

        
        with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
            file.write(f"Epoch: {epoch + 1}/{EPOCH}, Train Loss: {train_loss:.4f}\n")
            file.write(f"L2_v_norm: {L2_v_norm:.4f}, L2_v: {L2_v:.4f}\n")
            file.write(f"MSE_loss_norm: {MSE_loss_norm:.4f}, MSE_loss: {MSE_loss:.4f}\n")
            file.write(f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}\n")
        
        if (epoch+1) % 1 == 0 or epoch == 0 or (epoch+1) == EPOCH:
            print(f"Epoch: {epoch + 1}/{EPOCH}, Train Loss: {train_loss:.4f}")
            print(f"L2_v_norm: {L2_v_norm:.4f}, L2_v: {L2_v:.4f}")
            print(f"MSE_loss_norm: {MSE_loss_norm:.4f}, MSE_loss: {MSE_loss:.4f}")
            print(f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}")
            print("#################")

            
        # if (epoch+1) % 10 == 0 or epoch == 0 or (epoch+1) == EPOCH:
        if (epoch+1) % 5 == 0 or epoch == 0 or (epoch+1) == EPOCH:
            start_time = time.time() 
            test_error =  validate(args, model, test_dataloader, device=device)
            end_time = time.time()
            training_time1 = (end_time - start_time)
            
            test_L2_v = torch.tensor(test_error['L2_v'], device=device)
            test_L2_v_norm = torch.tensor(test_error['L2_v_norm'], device=device)
            test_MSE_loss = torch.tensor(test_error['MSE_loss'], device=device)
            test_MSE_loss_norm = torch.tensor(test_error['MSE_loss_norm'], device=device)

            test_L2_v = gather_tensor(test_L2_v)
            test_L2_v_norm = gather_tensor(test_L2_v_norm)
            test_MSE_loss = gather_tensor(test_MSE_loss)
            test_MSE_loss_norm = gather_tensor(test_MSE_loss_norm)
            
            print(f"Epoch: {epoch + 1}/{EPOCH}, test_L2_v_norm: {test_L2_v_norm:.4f}, test_L2_v: {test_L2_v:.4f}\n")
            print(f"test_MSE_loss_norm: {test_MSE_loss_norm:.4f}, test_MSE_loss: {test_MSE_loss:.4f}\n")
            print(f"time pre test epoch/s:{training_time1:.2f}")
            print("#################")
            
            writer.add_scalar('L2/test_L2_v', test_L2_v, epoch)
            writer.add_scalar('L2/test_L2_v_norm', test_L2_v_norm, epoch)
            writer.add_scalar('MSE/test_MSE_loss', test_MSE_loss, epoch)
            writer.add_scalar('MSE/test_MSE_loss_norm', test_MSE_loss_norm, epoch)
        
            
            with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
                file.write(f"Epoch: {epoch + 1}/{EPOCH}, test_L2_v_norm: {test_L2_v_norm:.4f}, test_L2_v: {test_L2_v:.4f}\n")
                file.write(f"test_MSE_loss_norm: {test_MSE_loss_norm:.4f}, test_MSE_loss: {test_MSE_loss:.4f}\n")
                file.write(f"time pre test epoch/s:{training_time1:.2f}\n")
        
        # if (epoch+1) % 50 == 0 or epoch == 0 or (epoch+1) == EPOCH:
        if (epoch+1) % 10 == 0 or epoch == 0 or (epoch+1) == EPOCH:
            if args.if_save:
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict() if args.train["if_multi_gpu"] else model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'learning_rate': scheduler.get_last_lr()[0],  
                }
                nn_save_path = os.path.join(args.save_path, "nn")
                os.makedirs(nn_save_path, exist_ok=True)
                torch.save(checkpoint, f"{nn_save_path}/{args.name}_{epoch+1}.nn")
        
        if min_loss > test_L2_v:
            min_loss = test_L2_v
            if args.if_save:
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict() if args.train["if_multi_gpu"] else model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'learning_rate': scheduler.get_last_lr()[0], 
                }
                nn_save_path = os.path.join(args.save_path, "nn")
                os.makedirs(nn_save_path, exist_ok=True)
                torch.save(checkpoint, f"{nn_save_path}/{args.name}_best.nn")


    writer.close()
            
if __name__ == "__main__":
    args = parse_args()
    print(args)
    print(f"args.device:{args.device}")
    
    device = args.device
    if 'cuda' in device:
        assert torch.cuda.is_available()
        if device == 'cuda' and torch.cuda.device_count() > 1 and args.train["if_multi_gpu"]:
            use_multi_gpu = True
            print(f"lets use {torch.cuda.device_count()} gpus!")
        else:
            use_multi_gpu = False
            print(f"lets use 1 gpu!")
    else:
        use_multi_gpu = False
        print(f"lets use cpu!")
    device = torch.device(device)
    print('device:', device)
    
    if not os.path.exists(f"{args.save_path}/record/"):
        os.makedirs(f"{args.save_path}/record/")
    with open(f"{args.save_path}/record/{args.name}_training_log.txt", "w") as file:
        file.write(str(args) + "\n")
        file.write("=="*10 + "\n")
        file.write(f"the begin time is {time.asctime(time.localtime(time.time()))}\n")
        file.write("=="*10 + "\n")
        
    # if args.seed is not None:
    #     print(f"args.seed:{args.seed}")
    #     set_seed(args.seed)
    
    print("#*"*10 + "  begin the main:")
    # # train+val+test
    main(args)

    print("#*"*10 + "  end the main!")
    with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
        file.write("=="*10 + "\n")
        file.write(f"the end time is {time.asctime(time.localtime(time.time()) )}\n")
        file.write("=="*10 + "\n")
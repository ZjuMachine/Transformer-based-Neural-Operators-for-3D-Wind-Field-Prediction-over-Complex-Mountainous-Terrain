# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling use_cross_unet or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import numpy as np
import time, json, os
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader
from tqdm import tqdm


def get_nb_trainable_params(model):
    '''
    Return the number of trainable parameters
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def train(device, model, train_loader, optimizer, scheduler, reg=1):
    model.train()

    criterion_func = nn.MSELoss(reduction='none')
    losses_velo = []
    for i, (cfd_data, geom) in enumerate(train_loader):
        # print(f"case {i}")
        # epsilon = torch.rand(cfd_data.x.shape[0], 1).to(device)
        epsilon = None
        cfd_data = cfd_data.to(device)
        
        geom = geom.to(device)
        optimizer.zero_grad()
        # print('enter')
        out = model((cfd_data, geom, epsilon))
        targets = cfd_data.y

        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(dim=0)
        loss_velo = loss_velo_var.mean()

        loss_velo.backward()

        optimizer.step()
        scheduler.step()

        losses_velo.append(loss_velo.item())
        
        # del (loss_velo_var)
        # del (loss_velo)
        # del (cfd_data)
        # del (geom)
        torch.cuda.empty_cache() #

    return np.mean(losses_velo)


@torch.no_grad()
def test(device, model, test_loader):
    model.eval()

    criterion_func = nn.MSELoss(reduction='none')
    losses_velo = []
    for cfd_data, geom in test_loader:
        # epsilon = torch.rand(cfd_data.x.shape[0], 1).to(device)
        epsilon = None
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        out = model((cfd_data, geom, epsilon))
        targets = cfd_data.y

        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(dim=0)
        loss_velo = loss_velo_var.mean()

        losses_velo.append(loss_velo.item())
        # del (loss_velo_var)
        # del (loss_velo)
        # del (cfd_data)
        # del (geom)
        torch.cuda.empty_cache()

    return np.mean(losses_velo)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(device, train_dataset, val_dataset, Net, hparams, path, reg=1, val_iter=1, coef_norm=[]):
    model = Net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'],
        final_div_factor=1000.,
    )
    start = time.time()

    train_loss, val_loss = 1e5, 1e5
    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    for epoch in pbar_train:
        train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, drop_last=True)
        loss_velo = train(device, model, train_loader, optimizer, lr_scheduler, reg=reg)
        train_loss = loss_velo
        del (train_loader)
        # torch.cuda.empty_cache()

        if val_iter is not None and (epoch == hparams['nb_epochs'] - 1 or epoch % val_iter == 0):
            val_loader = DataLoader(val_dataset, batch_size=1)

            loss_velo = test(device, model, val_loader)
            val_loss = loss_velo
            del (val_loader)
            # torch.cuda.empty_cache()

            # 每次计算验证损失后保存检查点
            checkpoint_path = path + os.sep + f'model_{hparams["model_name"]}_{hparams["nb_epochs"]}_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'hparams': hparams,
            }, checkpoint_path)
            # model.load_state_dict(checkpoint['model_state_dict']) 这样去加载这个保存好的权重即可

            print(f"Checkpoint saved: {checkpoint_path}")

            pbar_train.set_postfix(train_loss=train_loss, val_loss=val_loss)

        else:
            pbar_train.set_postfix(train_loss=train_loss)

    end = time.time()
    time_elapsed = end - start
    params_model = get_nb_trainable_params(model).astype('float')
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
    torch.save(model, path + os.sep + f'model_{hparams["model_name"]}_{hparams["nb_epochs"]}.pth')

    if val_iter is not None:
        with open(path + os.sep + f'log_{hparams["nb_epochs"]}.json', 'a') as f:
            json.dump(
                {
                    'nb_parameters': params_model,
                    'time_elapsed': time_elapsed,
                    'hparams': hparams,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'coef_norm': list(coef_norm),
                }, f, indent=12, cls=NumpyEncoder
            )

    return model

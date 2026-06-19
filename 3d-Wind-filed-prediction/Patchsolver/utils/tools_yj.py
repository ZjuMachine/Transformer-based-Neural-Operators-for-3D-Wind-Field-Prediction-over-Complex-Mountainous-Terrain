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


import os
from typing import List, Tuple

def extract_name_and_angle(path):

    # print(path_list)
    # results = []
    # for path in path_list:
    # print(path)
    path = path.strip()  
    if '/' in path:
        parts = path.split('/')
        if len(parts) >= 3:
            name = parts[0]
            angle = parts[2]
            # results.append((name, angle))
        # print(name, angle)
    return name ,angle 

def get_save_name(name: str, angle: str, replace_dot: bool = False) -> str:

    if replace_dot:
        angle = angle.replace('.', '_')  # 
    return f"{name}_{angle}"
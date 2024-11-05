from os import path as osp
from typing import Callable, List, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures.bbox_3d.cam_box3d import CameraInstance3DBoxes
from mmdet3d.datasets import NuScenesDataset


# BBOX_CLASS_TO_ID = {
#     # Dynamic Classes
#     "Car"                   : 0,
#     "Pedestrian"            : 1,
#     "Bike"                  : 2,
#     "Motorcycle"            : 3,
#     "Golf Cart"             : 4, # Unused
#     "Truck"                 : 5, # Unused
#     "Scooter"               : 6,
#     # Static Classes
#     "Tree"                  : 7,
#     "Traffic Sign"          : 8,
#     "Canopy"                : 9,
#     "Traffic Light"         : 10,
#     "Bike Rack"             : 11,
#     "Bollard"               : 12,
#     "Construction Barrier"  : 13, # Unused
#     "Parking Kiosk"         : 14,
#     "Mailbox"               : 15,
#     "Fire Hydrant"          : 16,
#     # Static Class Mixed
#     "Freestanding Plant"    : 17,
#     "Pole"                  : 18,
#     "Informational Sign"    : 19,
#     "Door"                  : 20,
#     "Fence"                 : 21,
#     "Railing"               : 22,
#     "Cone"                  : 23,
#     "Chair"                 : 24,
#     "Bench"                 : 25,
#     "Table"                 : 26,
#     "Trash Can"             : 27,
#     "Newspaper Dispenser"   : 28,
#     # Static Classes Indoor
#     "Room Label"            : 29,
#     "Stanchion"             : 30,
#     "Sanitizer Dispenser"   : 31,
#     "Condiment Dispenser"   : 32,
#     "Vending Machine"       : 33,
#     "Emergency Aid Kit"     : 34,
#     "Fire Extinguisher"     : 35,
#     "Computer"              : 36,
#     "Television"            : 37, # unused
#     "Other"                 : 38,
#     "Horse"                 : 39,
#     # New Classes
#     "Pickup Truck"          : 40,
#     "Delivery Truck"        : 41,
#     "Service Vehicle"       : 42,
#     "Utility Vehicle"       : 43,
#     "Fire Alarm"            : 44,
#     "ATM"                   : 45,
#     "Cart"                  : 46,
#     "Couch"                 : 47,
#     "Traffic Arm"           : 48,
#     "Wall Sign"             : 49,
#     "Floor Sign"            : 50,
#     "Door Switch"           : 51,
#     "Emergency Phone"       : 52,
#     "Dumpster"              : 53,
#     "Vacuum Cleaner"        : 54, # unused
#     "Segway"                : 55,
#     "Bus"                   : 56,
#     "Skateboard"            : 57,
#     "Water Fountain"        : 58
# }

BBOX_CLASS_TO_ID = {
    "Car"                   : 0,
    "Pedestrian"            : 1,
    "Cyclist"               : 2
}

BBOX_ID_TO_COLOR = [ # BGR
    (140, 51, 147),         #0 Car (Blue)
    (7, 33, 229),           #1 Person (Green)
    (66, 21, 72),           #2 Bike
    (67, 31, 116),          #3 Motorcycle (Orange)
    (159, 137, 254),        #4 Golf Cart (Yellow)
    (52, 32, 130),          #5 Truck (Purple)
    (239, 92, 215),         #6 Scooter (Red)
    (4, 108, 69),           #7 Tree (Gold)
    (160, 129, 2),          #8 Traffic Sign (Dark Red)
    (160, 93, 2),           #9 Canopy (Silver)
    (254, 145, 38),         #10 Traffic Lights (Lime)
    (227, 189, 1),          #11 Bike Rack (Pink)
    (202, 79, 74),          #12 Bollard (Gray)
    (255, 196, 208),        #13 Construction Barrier (Light Orange)
    (166, 240, 4),          #14 Parking Kiosk (Dark Blue)
    (113, 168, 3),          #15 Mailbox (Royal Blue)
    (14, 60, 157),           #16 Fire Hydrant (Red)
    (41, 159, 115),         #17 Freestanding Plant (Green)
    (91, 79, 14),           #18 Pole (Texan Orange)
    (220, 184, 94),         #19 Informational Sign (Yellow)
    (202, 159, 41),         #20 Door (Dark Orange)
    (253, 137, 129),        #21 Fence (Brown)
    (97, 37, 32),           #22 Railing (Orange)
    (91, 31, 39),           #23 Cone (Light Orange)
    (24, 55, 95),           #24 Chair (Turquoise)
    (0, 87, 192),           #25 Bench (Dark Green)
    (31, 70, 142),          #26 Table (Olive)
    (24, 45, 66),           #27 Trash Can (Bright Blue)
    (30, 54, 11),           #28 Newspaper Dispenser (Light Orange)
    (247, 148, 90),         #29 Room Label (Magenta)
    (250, 126, 149),        #30 Stanchion (Light Gray)
    (70, 106, 19),          #31 Sanitizer Dispenser (Turquoise)
    (128, 132, 0),          #32 Condiment Dispenser (Dark Green)
    (152, 163, 0),          #33 Vending Machine (Sky Blue)
    (6, 32, 231),        #34 Emergency Aid Kit (Light Pink)
    (8, 68, 212),          #35 Fire Extinguisher (Light Red)
    (18, 34, 119),          #36 Computer (Dark Green)
    (17, 46, 168),          #37 Television (Black)
    (203, 226, 37),            #38 Other (Orange)
    (255, 83, 0),            #39 Horse (Orange)
    (100, 34, 168),         #40 Pickup Truck (Sky Blue)
    (150, 69, 253),         #41 Delivery Truck (Neon Pink)
    (46, 22, 78),           #42 Service Vehicle (Lime)
    (121, 46, 216),         #43 Utility Vehicle (Green)
    (37, 95, 238),           #44 Fire Alarm (Blue)
    (95, 100, 14),          #45 ATM (Light Pink)
    (25, 97, 119),          #46 Cart (Light Red)
    (18, 113, 225),         #47 Couch (Dark Green)
    (207, 66, 89),          #48 Traffic Arm (Black)
    (215, 80, 2),           #49 Wall Sign (White)
    (161, 125, 16),         #50 Floor Sign (Light Red)
    (82, 46, 22),           #51 Door Switch (Dark Green)
    (28, 42, 65),         #52 Emergency Phone (Light Black)
    (0, 140, 180),          #53 Dumpster (White)
    (0, 73, 207),           #54 Vacuum Cleaner (Dark Gray)
    (120, 94, 242),         #55 Segway
    (35, 28, 79),           #56 Bus
    (56, 30, 178),          #57 Skateboard
    (48, 49, 20)            #58 WAter Fountain 
]


@DATASETS.register_module()
class CodaDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        load_type (str): Type of loading mode. Defaults to 'frame_based'.

            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
                to convert to the FOV-based data type to support image-based
                detector.
            - 'fov_image_based': Only load the instances inside the default
                cam, and need to convert to the FOV-based data type to support
                image-based detector.
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        with_velocity (bool): Whether to include velocity prediction
            into the experiments. Defaults to True.
        use_valid_flag (bool): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
    """
    METAINFO = {
        'classes': tuple(BBOX_CLASS_TO_ID.keys()),
        'version': 'v1.0-trainval',
        'palette': BBOX_ID_TO_COLOR
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 box_type_3d: str = 'LiDAR',
                 load_type: str = 'frame_based',
                 modality: dict = dict(
                     use_camera=False,
                     use_lidar=True,
                 ),
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 with_velocity: bool = False,
                 use_valid_flag: bool = False,
                 **kwargs) -> None:
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            box_type_3d=box_type_3d,
            load_type=load_type,
            modality=modality,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            with_velocity=with_velocity,
            use_valid_flag=use_valid_flag,
            **kwargs)
        
        
    def _filter_with_mask(self, ann_info: dict) -> dict:
        return ann_info
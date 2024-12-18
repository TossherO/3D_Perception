import numpy as np
import torch
import mmcv
import cv2
from mmdet3d.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadImageFromFile
from typing import Optional, Union
import mmengine.fileio as fileio


@TRANSFORMS.register_module()
class LoadImageFromFileCoda(LoadImageFromFile):

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        cam = list(results['images'].keys())[0]
        filename = results['images'][cam]['img_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        lidar2cam = np.array(results['images'][cam]['lidar2cam'])
        cam2img = np.eye(4).astype(np.float64)
        cam2img[:3, :3] = np.array(results['images'][cam]['cam2img'])
        results['lidar2img'] = cam2img @ lidar2cam

        return results
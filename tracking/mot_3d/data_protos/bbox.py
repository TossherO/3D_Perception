""" The defination and basic methods of bbox
"""
import numpy as np
from copy import deepcopy


class BBox:
    def __init__(self, x=None, y=None, z=None, l=None, w=None, h=None, o=None):
        self.x = x      # center x
        self.y = y      # center y
        self.z = z      # center z
        self.l = l      # height
        self.w = w      # width
        self.h = h      # length
        self.o = o      # orientation
        self.s = None   # detection score
    
    def __str__(self):
        return 'x: {}, y: {}, z: {}, length: {}, width: {}, height: {}, heading: {}, score: {}'.format(
            self.x, self.y, self.z, self.l, self.w, self.h, self.o, self.s)
    
    @classmethod
    def bbox2dict(cls, bbox):
        return {
            'center_x': bbox.x, 'center_y': bbox.y, 'center_z': bbox.z,
            'length': bbox.l, 'height': bbox.h, 'width': bbox.w, 'heading': bbox.o}
    
    @classmethod
    def bbox2array(cls, bbox, return_score=True):
        if bbox.s is None or not return_score:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.l, bbox.w, bbox.h, bbox.o])
        else:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.l, bbox.w, bbox.h, bbox.o, bbox.s])

    @classmethod
    def array2bbox(cls, data):
        bbox = BBox()
        bbox.x, bbox.y, bbox.z, bbox.l, bbox.w, bbox.h, bbox.o = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox
    
    @classmethod
    def dict2bbox(cls, data):
        bbox = BBox()
        bbox.x = data['center_x']
        bbox.y = data['center_y']
        bbox.z = data['center_z']
        bbox.l = data['length']
        bbox.w = data['width']
        bbox.h = data['height']
        bbox.o = data['heading']
        if 'score' in data.keys():
            bbox.s = data['score']
        return bbox
    
    @classmethod
    def copy_bbox(cls, bboxa, bboxb):
        bboxa.x = bboxb.x
        bboxa.y = bboxb.y
        bboxa.z = bboxb.z
        bboxa.l = bboxb.l
        bboxa.w = bboxb.w
        bboxa.h = bboxb.h
        bboxa.o = bboxb.o
        bboxa.s = bboxb.s
        return
    
    @classmethod
    def box2corners2d(cls, bbox):
        """ the coordinates for bottom corners
        """
        bottom_center = np.array([bbox.x, bbox.y, bbox.z - bbox.h / 2])
        cos, sin = np.cos(bbox.o), np.sin(bbox.o)
        pc0 = np.array([bbox.x + cos * bbox.l / 2 + sin * bbox.w / 2,
                        bbox.y + sin * bbox.l / 2 - cos * bbox.w / 2,
                        bbox.z - bbox.h / 2])
        pc1 = np.array([bbox.x + cos * bbox.l / 2 - sin * bbox.w / 2,
                        bbox.y + sin * bbox.l / 2 + cos * bbox.w / 2,
                        bbox.z - bbox.h / 2])
        pc2 = 2 * bottom_center - pc0
        pc3 = 2 * bottom_center - pc1
    
        return [pc0.tolist(), pc1.tolist(), pc2.tolist(), pc3.tolist()]
    
    @classmethod
    def box2corners3d(cls, bbox):
        """ the coordinates for bottom corners
        """
        center = np.array([bbox.x, bbox.y, bbox.z])
        bottom_corners = np.array(BBox.box2corners2d(bbox))
        up_corners = 2 * center - bottom_corners
        corners = np.concatenate([up_corners, bottom_corners], axis=0)
        return corners.tolist()
    
    @classmethod
    def motion2bbox(cls, bbox, motion):
        result = deepcopy(bbox)
        result.x += motion[0]
        result.y += motion[1]
        result.z += motion[2]
        result.o += motion[6]
        return result
    
    @classmethod
    def set_bbox_size(cls, bbox, size_array):
        result = deepcopy(bbox)
        result.l, result.w, result.h = size_array
        return result
    
    @classmethod
    def set_bbox_with_states(cls, prev_bbox, state_array):
        prev_array = BBox.bbox2array(prev_bbox)
        prev_array[:4] += state_array[:4]
        prev_array[4:] = state_array[4:]
        bbox = BBox.array2bbox(prev_array)
        return bbox 
    
    @classmethod
    def box_pts2world(cls, ego_matrix, pcs):
        new_pcs = np.concatenate((pcs,
                                  np.ones(pcs.shape[0])[:, np.newaxis]),
                                  axis=1)
        new_pcs = ego_matrix @ new_pcs.T
        new_pcs = new_pcs.T[:, :3]
        return new_pcs
    
    @classmethod
    def edge2yaw(cls, center, edge):
        vec = edge - center
        yaw = np.arccos(vec[0] / np.linalg.norm(vec))
        if vec[1] < 0:
            yaw = -yaw
        return yaw
    
    @classmethod
    def bbox2world(cls, ego_matrix, box):
        # center and corners
        corners = np.array(BBox.box2corners2d(box))
        center = BBox.bbox2array(box)[:3][np.newaxis, :]
        center = BBox.box_pts2world(ego_matrix, center)[0]
        corners = BBox.box_pts2world(ego_matrix, corners)
        # heading
        edge_mid_point = (corners[0] + corners[1]) / 2
        yaw = BBox.edge2yaw(center[:2], edge_mid_point[:2])
        
        result = deepcopy(box)
        result.x, result.y, result.z = center
        result.o = yaw
        return result
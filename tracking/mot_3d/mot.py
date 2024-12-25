import mot_3d.tracklet as tracklet
from .frame_data import FrameData
from .update_info_data import UpdateInfoData
from .association import associate_dets_to_tracks, associate_unmatched_trks
import torch


class MOTModel:
    def __init__(self, configs, class_name):
        self.trackers = list()         # tracker for each single tracklet
        self.frame_count = 0           # record for the frames
        self.count = 0                 # record the obj number to assign ids
        self.time_stamp = None         # the previous time stamp

        general_configs = configs['general']
        class_specific_configs = configs['class_specific'].get(class_name)
        if class_specific_configs != 'None':
            for key in class_specific_configs:
                general_configs[key] = class_specific_configs[key]

        self.match_type = general_configs['match_type']
        self.asso = general_configs['asso']
        self.score_threshold = general_configs['score_threshold']
        self.asso_thres = general_configs['asso_thres'][self.asso]
        self.score_threshold_second_stage = general_configs['score_threshold_second_stage'][self.asso]
        self.asso_thres_second_stage = general_configs['asso_thres_second_stage'][self.asso]
        self.motion_model = general_configs['motion_model']
        self.max_age = general_configs['max_age_since_update']
        self.min_hits = general_configs['min_hits_to_birth']
        self.single_match = general_configs['single_match']
        self.configs = general_configs

    
    def frame_mot(self, input_data: FrameData):
        """ For each frame input, generate the latest mot results
        Args:
            input_data (FrameData): input data, including detection bboxes and ego information
        Returns:
            tracks on this frame: [(bbox0, id0), (bbox1, id1), ...]
        """
        self.frame_count += 1

        # initialize the time stamp on frame 0
        if self.time_stamp is None:
            self.time_stamp = input_data.time_stamp
    
        # filter out low-score detections
        dets = input_data.dets
        det_indexes = [i for i, det in enumerate(dets) if det.s >= self.score_threshold]
        dets = [dets[i] for i in det_indexes]

        # prediction and association
        trk_preds = list()
        for trk in self.trackers:
            trk_preds.append(trk.predict(input_data.time_stamp))
        matched, unmatched_dets, unmatched_trks = associate_dets_to_tracks(dets, trk_preds, 
            self.match_type, self.asso, self.asso_thres)
        for k in range(len(matched)):
            matched[k][0] = det_indexes[matched[k][0]]
        for k in range(len(unmatched_dets)):
            unmatched_dets[k] = det_indexes[unmatched_dets[k]]
        
        # association in second stage
        if self.single_match:
            unmatched_trk_preds = [self.trackers[t].get_state() for t in unmatched_trks]
            update_modes = [0] * len(unmatched_trks)
        else:
            dets = input_data.dets
            det_indexes = [i for i, det in enumerate(dets) if det.s >= self.score_threshold_second_stage]
            dets = [dets[i] for i in det_indexes]
            unmatched_trk_preds = [self.trackers[t].get_state() for t in unmatched_trks]
            update_modes = associate_unmatched_trks(dets, unmatched_trk_preds, self.asso, self.asso_thres_second_stage)

        # update the matched tracks
        for i in range(len(matched)):
            d = matched[i][0]
            trk = self.trackers[matched[i][1]]
            update_info = UpdateInfoData(mode=1, bbox=input_data.dets[d], ego=input_data.ego, 
                        frame_index=self.frame_count, pc=input_data.pc, dets=input_data.dets)
            trk.update(update_info)
        for i in range(len(unmatched_trks)):
            trk = self.trackers[unmatched_trks[i]]
            update_info = UpdateInfoData(mode=update_modes[i], bbox=unmatched_trk_preds[i], ego=input_data.ego, 
                        frame_index=self.frame_count, pc=input_data.pc, dets=input_data.dets)
            trk.update(update_info)

        # create new tracks for unmatched detections
        for index in unmatched_dets:
            track = tracklet.Tracklet(self.configs, self.count, input_data.dets[index], input_data.det_types[index], 
                self.frame_count, time_stamp=input_data.time_stamp)
            self.trackers.append(track)
            self.count += 1
        
        # remove dead tracks
        track_num = len(self.trackers)
        for index, trk in enumerate(reversed(self.trackers)):
            if trk.death(self.frame_count):
                self.trackers.pop(track_num - 1 - index)
        
        # output the results
        result = list()
        for trk in self.trackers:
            state_string = trk.state_string(self.frame_count)
            result.append((trk.get_state(), trk.id, state_string, trk.det_type))
        
        # wrap up and update the information about the mot trackers
        self.time_stamp = input_data.time_stamp
        for trk in self.trackers:
            trk.sync_time_stamp(self.time_stamp)

        return result
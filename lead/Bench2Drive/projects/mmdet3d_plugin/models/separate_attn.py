import warnings
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import linear
from torch.nn.init import xavier_uniform_, constant_

from mmcv.utils import deprecated_api_warning
from mmcv.runner import auto_fp16
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.drop import build_dropout
import torch.utils.checkpoint as cp
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    NORM_LAYERS,
    PLUGIN_LAYERS,
    FEEDFORWARD_NETWORK,
)

from .blocks import linear_relu_ln


@ATTENTION.register_module()
class SeparateAttention(nn.Module):
    def __init__(self,
                 attn=None,
                 embed_dims=256,
                 query_select=None,
                 separate_list=None,
                 decouple_list=None,
                 **kwargs):
        super(SeparateAttention, self).__init__()
        self.query_select = query_select
        self.separate_list = separate_list
        self.decouple_list = decouple_list
        assert separate_list is not None
        assert decouple_list is not None
        assert len(separate_list) == len(decouple_list)

        if isinstance(attn, dict):
            attns = [build_from_cfg(attn, ATTENTION) for _ in range(len(separate_list))]
        elif isinstance(attn, list):
            attns = [build_from_cfg(attn_, ATTENTION) for attn_ in attn]
        else:
            raise NotImplementedError

        self.attns = nn.Sequential(*attns)

    def forward(self,
                query, key=None, value=None,
                query_pos=None, key_pos=None,
                attn_mask=None,
                num_anchor_cumsum=None,
                num_temp_anchor_cumsum=None,
                fc_before=None, fc_after=None,
                **kwargs):
        output_instance = query.clone()
        if key is None:
            for sep, separate in enumerate(self.separate_list):
                sep_query = []
                sep_value = []
                sep_query_pos = []
                num_query_list = []
                for modality in separate:
                    index = self.query_select.index(modality)
                    start, end = num_anchor_cumsum[index], num_anchor_cumsum[index+1]
                    sep_query.append(query[:, start:end])
                    num_query_list.append(end - start)
                    if value is not None:
                        sep_value.append(value[:, start:end])
                    if query_pos is not None:
                        sep_query_pos.append(query_pos[:, start:end])
                sep_query = torch.cat(sep_query, dim=1)
                sep_value = torch.cat(sep_value, dim=1) if len(sep_value) else None
                sep_query_pos = torch.cat(sep_query_pos, dim=1) if len(sep_query_pos) else None

                sep_attn_mask = self.get_separate_attn_mask(attn_mask, separate, num_anchor_cumsum)

                if self.decouple_list[sep]:
                    sep_query = torch.cat([sep_query, sep_query_pos], dim=-1)
                    sep_query_pos = None

                    if sep_value is not None:
                        sep_value = fc_before(sep_value)

                    output = fc_after(self.attns[sep](query=sep_query, value=sep_value, query_pos=sep_query_pos, attn_mask=sep_attn_mask))
                else:
                    output = self.attns[sep](query=sep_query, value=sep_value, query_pos=sep_query_pos, attn_mask=sep_attn_mask)

                # update query
                num_query_cumsum = np.cumsum([0] + num_query_list)
                for index1, modality in enumerate(separate):
                    index2 = self.query_select.index(modality)
                    start1, end1 = num_query_cumsum[index1], num_query_cumsum[index1+1]
                    start2, end2 = num_anchor_cumsum[index2], num_anchor_cumsum[index2+1]
                    output_instance[:, start2:end2] = output[:, start1:end1]

        else:
            for sep, separate in enumerate(self.separate_list):
                sep_query = []
                sep_key = []
                sep_value = []
                sep_query_pos = []
                sep_key_pos = []
                num_query_list = []
                for modality in separate:
                    index = self.query_select.index(modality)
                    start1, end1 = num_anchor_cumsum[index], num_anchor_cumsum[index+1]
                    start2, end2 = num_temp_anchor_cumsum[index], num_temp_anchor_cumsum[index+1]
                    sep_query.append(query[:, start1:end1])
                    sep_key.append(key[:, start2:end2])
                    num_query_list.append(end1 - start1)

                    if value is not None:
                        sep_value.append(value[:, start2:end2])
                    if query_pos is not None:
                        sep_query_pos.append(query_pos[:, start1:end1])
                    if key_pos is not None:
                        sep_key_pos.append(key_pos[:, start2:end2])
                sep_query = torch.cat(sep_query, dim=1)
                sep_key = torch.cat(sep_key, dim=1)
                sep_value = torch.cat(sep_value, dim=1) if len(sep_value) else None
                sep_query_pos = torch.cat(sep_query_pos, dim=1) if len(sep_query_pos) else None
                sep_key_pos = torch.cat(sep_key_pos, dim=1) if len(sep_key_pos) else None

                # for none key / value
                if sep_key.size(1) == 0:
                    sep_key = None
                    sep_key_pos = None
                if sep_value.size(1) == 0:
                    sep_value = None

                sep_attn_mask = self.get_separate_attn_mask(attn_mask, separate, num_anchor_cumsum, num_temp_anchor_cumsum)

                if self.decouple_list[sep]:
                    sep_query = torch.cat([sep_query, sep_query_pos], dim=-1)
                    sep_key = torch.cat([sep_key, sep_key_pos], dim=-1)
                    sep_query_pos, sep_key_pos = None, None

                    if sep_value is not None:
                        sep_value = fc_before(sep_value)

                    output = fc_after(self.attns[sep](query=sep_query, key=sep_key, value=sep_value,
                                                      query_pos=sep_query_pos, key_pos=sep_key_pos, attn_mask=sep_attn_mask))
                else:
                    output = self.attns[sep](query=sep_query, key=sep_key, value=sep_value,
                                             query_pos=sep_query_pos, key_pos=sep_key_pos, attn_mask=sep_attn_mask)

                # update query
                num_query_cumsum = np.cumsum([0] + num_query_list)
                for index1, modality in enumerate(separate):
                    index2 = self.query_select.index(modality)
                    start1, end1 = num_query_cumsum[index1], num_query_cumsum[index1+1]
                    start2, end2 = num_anchor_cumsum[index2], num_anchor_cumsum[index2+1]
                    output_instance[:, start2:end2] = output[:, start1:end1]

        return output_instance


    def get_separate_attn_mask(self, attn_mask, separate, num_anchor_cumsum, num_temp_anchor_cumsum=None):
        if attn_mask is None:
            return None

        sep_attn_mask2 = []
        for modality1 in separate:
            m_index1 = self.query_select.index(modality1)
            m_start1, m_end1 = num_anchor_cumsum[m_index1], num_anchor_cumsum[m_index1 + 1]
            sep_attn_mask1 = []
            for modality2 in separate:
                m_index2 = self.query_select.index(modality2)
                if num_temp_anchor_cumsum is None:
                    m_start2, m_end2 = num_anchor_cumsum[m_index2], num_anchor_cumsum[m_index2 + 1]
                else:
                    m_start2, m_end2 = num_temp_anchor_cumsum[m_index2], num_temp_anchor_cumsum[m_index2 + 1]

                sep_attn_mask1.append(attn_mask[m_start1:m_end1, m_start2:m_end2])
            sep_attn_mask2.append(torch.cat(sep_attn_mask1, dim=1))
        sup_attn_mask = torch.cat(sep_attn_mask2, dim=0)

        if (sup_attn_mask == 0).all().item() == True:
            sup_attn_mask = None  # for flash-attn

        return sup_attn_mask


@ATTENTION.register_module()
class TemporalSeparateAttention(nn.Module):
    def __init__(self,
                 attn=None,
                 embed_dims=256,
                 query_select=None,
                 query_list=None,
                 key_list=None,
                 decouple_list=None,
                 use_updated_query=False,
                 **kwargs):
        super(TemporalSeparateAttention, self).__init__()
        self.query_select = query_select
        self.query_list = query_list
        self.key_list = key_list
        self.decouple_list = decouple_list
        self.use_updated_query = use_updated_query
        assert query_list is not None
        assert decouple_list is not None
        assert len(query_list) == len(key_list) == len(decouple_list)

        if isinstance(attn, dict):
            attns = [build_from_cfg(attn, ATTENTION) for _ in range(len(query_list))]
        elif isinstance(attn, list):
            attns = [build_from_cfg(attn_, ATTENTION) for attn_ in attn]
        else:
            raise NotImplementedError

        self.attns = nn.Sequential(*attns)

    def forward(self,
                query, key=None, value=None,
                query_pos=None, key_pos=None,
                attn_mask=None,
                num_anchor_cumsum=None,
                num_temp_anchor_cumsum=None,
                fc_before=None, fc_after=None,
                **kwargs):
        output_instance = query.clone()

        if key is None or num_temp_anchor_cumsum is None:
            key = query
            key_pos = query_pos
            num_temp_anchor_cumsum = num_anchor_cumsum

        for sep in range(len(self.decouple_list)):
            sep_query = []
            sep_key = []
            sep_value = []
            sep_query_pos = []
            sep_key_pos = []
            num_query_list = []

            sub_query_list = self.query_list[sep]
            sub_key_list = self.key_list[sep]

            for query_modality in sub_query_list:
                query_index = self.query_select.index(query_modality)
                start1, end1 = num_anchor_cumsum[query_index], num_anchor_cumsum[query_index+1]
                if self.use_updated_query:
                    sep_query.append(output_instance[:, start1:end1])
                else:
                    sep_query.append(query[:, start1:end1])
                num_query_list.append(end1 - start1)

                if query_pos is not None:
                    sep_query_pos.append(query_pos[:, start1:end1])

            for key_modality in sub_key_list:
                key_index = self.query_select.index(key_modality)
                start2, end2 = num_temp_anchor_cumsum[key_index], num_temp_anchor_cumsum[key_index+1]
                sep_key.append(key[:, start2:end2])

                if key_pos is not None:
                    sep_key_pos.append(key_pos[:, start2:end2])
                if value is not None:
                    sep_value.append(value[:, start2:end2])

            sep_query = torch.cat(sep_query, dim=1)
            sep_key = torch.cat(sep_key, dim=1)
            sep_value = torch.cat(sep_value, dim=1) if len(sep_value) else None
            sep_query_pos = torch.cat(sep_query_pos, dim=1) if len(sep_query_pos) else None
            sep_key_pos = torch.cat(sep_key_pos, dim=1) if len(sep_key_pos) else None

            # for none key / value
            if sep_key.size(1) == 0:
                sep_key = None
                sep_key_pos = None
            if sep_value is not None and sep_value.size(1) == 0:
                sep_value = None

            sep_attn_mask = self.get_separate_attn_mask(attn_mask, sub_query_list, sub_key_list,
                                                        num_anchor_cumsum, num_temp_anchor_cumsum)

            if self.decouple_list[sep]:
                sep_query = torch.cat([sep_query, sep_query_pos], dim=-1)
                sep_query_pos = None
                if sep_key is not None:
                    sep_key = torch.cat([sep_key, sep_key_pos], dim=-1)
                    sep_key_pos = None
                if sep_value is not None:
                    sep_value = fc_before(sep_value)

                output = fc_after(self.attns[sep](query=sep_query, key=sep_key, value=sep_value,
                                                  query_pos=sep_query_pos, key_pos=sep_key_pos, attn_mask=sep_attn_mask))
            else:
                output = self.attns[sep](query=sep_query, key=sep_key, value=sep_value,
                                         query_pos=sep_query_pos, key_pos=sep_key_pos, attn_mask=sep_attn_mask)

            # update query
            num_query_cumsum = np.cumsum([0] + num_query_list)
            for index1, modality in enumerate(sub_query_list):
                index2 = self.query_select.index(modality)
                start1, end1 = num_query_cumsum[index1], num_query_cumsum[index1+1]
                start2, end2 = num_anchor_cumsum[index2], num_anchor_cumsum[index2+1]
                output_instance[:, start2:end2] = output[:, start1:end1]

        return output_instance


    def get_separate_attn_mask(self, attn_mask, query_list, key_list, num_anchor_cumsum, num_temp_anchor_cumsum=None):
        if attn_mask is None:
            return None

        sep_attn_mask2 = []
        for modality1 in query_list:
            m_index1 = self.query_select.index(modality1)
            m_start1, m_end1 = num_anchor_cumsum[m_index1], num_anchor_cumsum[m_index1 + 1]
            sep_attn_mask1 = []
            for modality2 in key_list:
                m_index2 = self.query_select.index(modality2)
                if num_temp_anchor_cumsum is None:
                    m_start2, m_end2 = num_anchor_cumsum[m_index2], num_anchor_cumsum[m_index2 + 1]
                else:
                    m_start2, m_end2 = num_temp_anchor_cumsum[m_index2], num_temp_anchor_cumsum[m_index2 + 1]

                sep_attn_mask1.append(attn_mask[m_start1:m_end1, m_start2:m_end2])
            sep_attn_mask2.append(torch.cat(sep_attn_mask1, dim=1))
        sup_attn_mask = torch.cat(sep_attn_mask2, dim=0)

        if (sup_attn_mask == 0).all().item() == True:
            sup_attn_mask = None  # for flash-attn

        return sup_attn_mask


@ATTENTION.register_module()
class InteractiveAttention(nn.Module):
    def __init__(self,
                 attn=None,
                 embed_dims=256,
                 query_select=None,
                 query_list=None,
                 key_list=None,
                 decouple_list=None,
                 with_distance_attn_mask=False,
                 with_velocity_attn_mask=False,
                 attn_mask_ban_list=None,
                 attn_mask_cancel_list=None,
                 **kwargs):
        super(InteractiveAttention, self).__init__()
        self.query_select = query_select
        self.query_list = query_list
        self.key_list = key_list
        self.decouple_list = decouple_list

        self.with_distance_attn_mask = with_distance_attn_mask
        self.with_velocity_attn_mask = with_velocity_attn_mask

        self.attn_mask_ban_list = attn_mask_ban_list
        self.attn_mask_cancel_list = attn_mask_cancel_list

        assert query_list is not None
        assert decouple_list is not None
        assert len(query_list) == len(key_list) == len(decouple_list)

        if isinstance(attn, dict):
            attns = [build_from_cfg(attn, ATTENTION) for _ in range(len(query_list))]
        elif isinstance(attn, list):
            attns = [build_from_cfg(attn_, ATTENTION) for attn_ in attn]
        else:
            raise NotImplementedError

        self.attns = nn.Sequential(*attns)

    def forward(self,
                query, key=None, value=None,
                query_pos=None, key_pos=None,
                attn_mask=None,
                num_anchor_cumsum=None,
                num_temp_anchor_cumsum=None,
                fc_before=None, fc_after=None,
                distance_tau=None,
                velocity_tau=None,
                **kwargs):
        output_instance = query.clone()

        key = query
        key_pos = query_pos

        num_query_cumsum = num_anchor_cumsum
        num_key_cumsum = num_anchor_cumsum

        for sep in range(len(self.decouple_list)):
            sep_query = []
            sep_key = []
            sep_value = []
            sep_query_pos = []
            sep_key_pos = []
            num_query_list = []

            sub_query_list = self.query_list[sep]
            sub_key_list = self.key_list[sep]

            for query_modality in sub_query_list:
                query_index = self.query_select.index(query_modality)
                start1, end1 = num_query_cumsum[query_index], num_query_cumsum[query_index+1]
                sep_query.append(query[:, start1:end1])
                num_query_list.append(end1 - start1)

                if query_pos is not None:
                    sep_query_pos.append(query_pos[:, start1:end1])

            for key_modality in sub_key_list:
                key_index = self.query_select.index(key_modality)
                start2, end2 = num_key_cumsum[key_index], num_key_cumsum[key_index+1]
                sep_key.append(key[:, start2:end2])

                if key_pos is not None:
                    sep_key_pos.append(key_pos[:, start2:end2])
                if value is not None:
                    sep_value.append(value[:, start2:end2])

            sep_query = torch.cat(sep_query, dim=1)
            sep_key = torch.cat(sep_key, dim=1)
            sep_value = torch.cat(sep_value, dim=1) if len(sep_value) else None
            sep_query_pos = torch.cat(sep_query_pos, dim=1) if len(sep_query_pos) else None
            sep_key_pos = torch.cat(sep_key_pos, dim=1) if len(sep_key_pos) else None

            # for none key / value
            if sep_key.size(1) == 0:
                sep_key = None
                sep_key_pos = None
            if sep_value is not None and sep_value.size(1) == 0:
                sep_value = None

            sep_attn_mask = self.get_separate_attn_mask(attn_mask, sub_query_list, sub_key_list, num_query_cumsum, num_key_cumsum)

            if self.with_distance_attn_mask:
                dist_attn_mask = self.get_distance_attn_mask(sep_query, sub_query_list, sub_key_list, distance_tau, kwargs)
                sep_attn_mask = sep_attn_mask + dist_attn_mask if sep_attn_mask is not None else dist_attn_mask

            if self.with_velocity_attn_mask:
                velo_attn_mask = self.get_velocity_attn_mask(sep_query, sub_query_list, sub_key_list, velocity_tau, kwargs)
                sep_attn_mask = sep_attn_mask + velo_attn_mask if sep_attn_mask is not None else velo_attn_mask

            if self.attn_mask_ban_list is not None:
                sep_attn_mask = self.ban_attn_mask(sep_attn_mask, sub_query_list, sub_key_list, num_query_cumsum, num_key_cumsum)

            if self.attn_mask_cancel_list is not None:
                sep_attn_mask = self.cancel_attn_mask(sep_attn_mask, sub_query_list, sub_key_list, num_query_cumsum, num_key_cumsum)

            if self.decouple_list[sep]:
                sep_query = torch.cat([sep_query, sep_query_pos], dim=-1)
                sep_query_pos = None
                if sep_key is not None:
                    sep_key = torch.cat([sep_key, sep_key_pos], dim=-1)
                    sep_key_pos = None
                if sep_value is not None:
                    sep_value = fc_before(sep_value)

                output = fc_after(self.attns[sep](query=sep_query, key=sep_key, value=sep_value,
                                                  query_pos=sep_query_pos, key_pos=sep_key_pos, attn_mask=sep_attn_mask))
            else:
                output = self.attns[sep](query=sep_query, key=sep_key, value=sep_value,
                                         query_pos=sep_query_pos, key_pos=sep_key_pos, attn_mask=sep_attn_mask)

            # update query
            num_query_cumsum = np.cumsum([0] + num_query_list)
            for index1, modality in enumerate(sub_query_list):
                index2 = self.query_select.index(modality)
                start1, end1 = num_query_cumsum[index1], num_query_cumsum[index1+1]
                start2, end2 = num_anchor_cumsum[index2], num_anchor_cumsum[index2+1]
                output_instance[:, start2:end2] = output[:, start1:end1]

        return output_instance


    def get_separate_attn_mask(self, attn_mask, query_list, key_list, num_query_cumsum, num_key_cumsum):
        if attn_mask is None:
            return None

        sep_attn_mask2 = []
        for query in query_list:
            q_index = self.query_select.index(query)
            q_start, q_end = num_query_cumsum[q_index], num_query_cumsum[q_index + 1]
            sep_attn_mask1 = []
            for key in key_list:
                k_index = self.query_select.index(key)
                k_start, k_end = num_key_cumsum[k_index], num_key_cumsum[k_index + 1]

                sep_attn_mask0 = attn_mask[q_start:q_end, k_start:k_end]
                sep_attn_mask1.append(sep_attn_mask0)

            sep_attn_mask2.append(torch.cat(sep_attn_mask1, dim=1))
        sup_attn_mask = torch.cat(sep_attn_mask2, dim=0)

        if (sup_attn_mask == 0).all().item() == True:
            sup_attn_mask = None  # for flash-attn

        return sup_attn_mask


    def get_distance_attn_mask(self, sep_query, sep_query_list, sep_key_list, distance_tau, kwargs):
        bs = sep_query.shape[0]

        det_points = None
        map_points = None
        plan_points = None
        ego_points = None

        points_list = list(set(sep_query_list + sep_key_list))
        for points_type in points_list:
            if points_type == 'ego':
                ego_points = sep_query.new_zeros((bs, 1, 2)) # [bs, 1, 2] point-level
            if points_type == 'det':
                det_points = kwargs['det_anchor'][..., :2]   # [bs, n, 2] point-level
            if points_type == 'map':
                map_points = kwargs['map_anchor'].reshape(bs, kwargs['map_anchor'].size(1), -1, 2) # [bs, n, 20, 2] instance-level
            if points_type == 'plan':
                plan_points = kwargs['plan_anchor'].reshape(bs, kwargs['plan_anchor'].size(1), -1, 2) # [bs, n, 6, 2] instance-level

        level_dict = {
            "ego": "point",
            "det": "point",
            "map": "instance",
            "plan": "instance",
        }

        def get_dist(query_type, key_type):
            if query_type == 'ego':
                query_points = ego_points
            elif query_type == 'det':
                query_points = det_points
            elif query_type == 'map':
                query_points = map_points
            elif query_type == 'plan':
                query_points = plan_points
            else:
                raise NotImplementedError
            query_level = level_dict[query_type]

            if key_type == 'ego':
                key_points = ego_points
            elif key_type == 'det':
                key_points = det_points
            elif key_type == 'map':
                key_points = map_points
            elif key_type == 'plan':
                key_points = plan_points
            else:
                raise NotImplementedError
            key_level = level_dict[key_type]

            # point-point
            if query_level == "point" and key_level == "point":
                dist = torch.norm(query_points[:, :, None] - key_points[:, None], dim=-1)

            # point-instance
            elif query_level == "point" and key_level == "instance":
                dist = torch.norm(query_points[:, :, None, None, :] - key_points[:, None], dim=-1)
                dist = torch.min(dist, dim=-1).values

            # instance-point
            elif query_level == "instance" and key_level == "point":
                dist = torch.norm(query_points[:, :, None] - key_points[:, None, :, None], dim=-1)
                dist = torch.min(dist, dim=-1).values

            # instance-instance
            elif query_level == "instance" and key_level == "instance":
                dist = torch.norm(query_points[:, :, None, :, None] - key_points[:, None, :, None], dim=-1)
                dist = torch.min(dist.flatten(-2), dim=-1).values

            return dist

        all_query2key_dist_list = []
        for query_type in sep_query_list:
            query2key_dist_list = []
            for key_type in sep_key_list:
                query_key_dist = get_dist(query_type, key_type)
                query2key_dist_list.append(query_key_dist)
            all_query2key_dist = torch.cat(query2key_dist_list, dim=-1)
            all_query2key_dist_list.append(all_query2key_dist)
        distance = torch.cat(all_query2key_dist_list, dim=-2)

        tau = distance_tau(sep_query)
        tau = tau.permute(0, 2, 1)

        attn_mask = -distance[:, None, :, :] * tau[..., None]
        attn_mask = attn_mask.flatten(0, 1)

        return attn_mask


    def get_velocity_attn_mask(self, sep_query, sep_query_list, sep_key_list, velocity_tau, kwargs):
        bs = sep_query.shape[0]

        ego_vels = None
        det_vels = None
        map_vels = None
        plan_vels = None

        points_list = list(set(sep_query_list + sep_key_list))
        for points_type in points_list:
            if points_type == 'ego':
                ego_vels = sep_query.new_zeros((bs, 1, 1))
            if points_type == 'det':
                det_vels = torch.norm(kwargs['det_anchor'][..., 8:10], dim=-1, keepdim=True)
            if points_type == 'map':
                map_vels = sep_query.new_zeros((bs, kwargs['map_anchor'].size(1), 1))
            if points_type == 'plan':
                plan_vels = sep_query.new_zeros((bs, kwargs['plan_anchor'].size(1), 1))

        def get_vel(query_type, key_type):
            if query_type == 'ego':
                query_vels = ego_vels
            elif query_type == 'det':
                query_vels = det_vels
            elif query_type == 'map':
                query_vels = map_vels
            elif query_type == 'plan':
                query_vels = plan_vels
            else:
                raise NotImplementedError

            if key_type == 'ego':
                key_vels = ego_vels
            elif key_type == 'det':
                key_vels = det_vels
            elif key_type == 'map':
                key_vels = map_vels
            elif key_type == 'plan':
                key_vels = plan_vels
            else:
                raise NotImplementedError

            vels = (query_vels[:, :, None] - key_vels[:, None]).squeeze(-1)

            return vels

        all_query2key_vel_list = []
        for query_type in sep_query_list:
            query2key_vel_list = []
            for key_type in sep_key_list:
                query_key_vel = get_vel(query_type, key_type)
                query2key_vel_list.append(query_key_vel)
            all_query2key_vel = torch.cat(query2key_vel_list, dim=-1)
            all_query2key_vel_list.append(all_query2key_vel)
        velocity = torch.cat(all_query2key_vel_list, dim=-2)
        velocity = velocity - velocity.max()

        tau = velocity_tau(sep_query)
        tau = tau.permute(0, 2, 1)

        attn_mask = velocity[:, None, :, :] * tau[..., None]
        attn_mask = attn_mask.flatten(0, 1)

        return attn_mask


    def ban_attn_mask(self, attn_mask, query_list, key_list, num_query_cumsum, num_key_cumsum):
        if attn_mask is None:
            num_query = 0
            num_key = 0
            for query in query_list:
                q_index = self.query_select.index(query)
                q_start, q_end = num_query_cumsum[q_index], num_query_cumsum[q_index + 1]
                num_query = num_query + q_end - q_start
            for key in key_list:
                k_index = self.query_select.index(key)
                k_start, k_end = num_key_cumsum[k_index], num_key_cumsum[k_index + 1]
                num_key = num_key + k_end - k_start
            attn_mask = torch.zeros((num_query, num_key), dtype=torch.float32, device='cuda')

        for query in query_list:
            q_index = self.query_select.index(query)
            q_start, q_end = num_query_cumsum[q_index], num_query_cumsum[q_index + 1]
            for key in key_list:
                k_index = self.query_select.index(key)
                k_start, k_end = num_key_cumsum[k_index], num_key_cumsum[k_index + 1]

                if (query, key) in self.attn_mask_ban_list:
                    if len(attn_mask.shape) == 2:
                        ban_attn_mask = attn_mask[q_start:q_end, k_start:k_end]
                        attn_mask[q_start:q_end, k_start:k_end] = torch.ones_like(ban_attn_mask).fill_(float("-inf"))
                    elif len(attn_mask.shape) == 3:
                        ban_attn_mask = attn_mask[:, q_start:q_end, k_start:k_end]
                        attn_mask[:, q_start:q_end, k_start:k_end] = torch.ones_like(ban_attn_mask).fill_(float("-inf"))
                    else:
                        raise NotImplementedError
        return attn_mask


    def cancel_attn_mask(self, attn_mask, query_list, key_list, num_query_cumsum, num_key_cumsum):
        if attn_mask is None:
            num_query = 0
            num_key = 0
            for query in query_list:
                q_index = self.query_select.index(query)
                q_start, q_end = num_query_cumsum[q_index], num_query_cumsum[q_index + 1]
                num_query = num_query + q_end - q_start
            for key in key_list:
                k_index = self.query_select.index(key)
                k_start, k_end = num_key_cumsum[k_index], num_key_cumsum[k_index + 1]
                num_key = num_key + k_end - k_start
            attn_mask = torch.zeros((num_query, num_key), dtype=torch.float32, device='cuda')

        for query in query_list:
            q_index = self.query_select.index(query)
            q_start, q_end = num_query_cumsum[q_index], num_query_cumsum[q_index + 1]
            for key in key_list:
                k_index = self.query_select.index(key)
                k_start, k_end = num_key_cumsum[k_index], num_key_cumsum[k_index + 1]

                if (query, key) in self.attn_mask_cancel_list:
                    if len(attn_mask.shape) == 2:
                        ban_attn_mask = attn_mask[q_start:q_end, k_start:k_end]
                        attn_mask[q_start:q_end, k_start:k_end] = torch.ones_like(ban_attn_mask).fill_(0.)
                    elif len(attn_mask.shape) == 3:
                        ban_attn_mask = attn_mask[:, q_start:q_end, k_start:k_end]
                        attn_mask[:, q_start:q_end, k_start:k_end] = torch.ones_like(ban_attn_mask).fill_(0.)
                    else:
                        raise NotImplementedError
        return attn_mask
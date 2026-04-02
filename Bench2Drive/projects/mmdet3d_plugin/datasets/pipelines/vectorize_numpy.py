import numpy as np
from shapely.geometry import LineString
from typing import List, Tuple, Union, Dict
from IPython import embed

class VectorizeMapNumpy(object):
    """Generate vectoized map and put into `semantic_mask` key.
    Concretely, shapely geometry objects are converted into sample points (ndarray).
    We use args `sample_num`, `sample_dist`, to specify sampling method.

    Args:
        normalize (bool): whether to normalize points to range (0, 1).
        coords_dim (int): dimension of point coordinates.
        sample_num (int): number of points to interpolate from a polyline. Set to -1 to ignore.
        sample_dist (float): interpolate distance. Set to -1 to ignore.
    """

    def __init__(self, 
                 coords_dim: int,
                 sample_num: int=-1, 
                 sample_dist: float=-1, 
                 permute: bool=False
        ):
        self.coords_dim = coords_dim
        self.sample_num = sample_num
        self.sample_dist = sample_dist
        self.permute = permute

        if sample_dist > 0:
            assert sample_num < 0
            self.sample_fn = self.interp_fixed_dist
        elif sample_num > 0:
            assert sample_dist < 0
            self.sample_fn = self.interp_fixed_num

    def interp_fixed_num(self, line: LineString):
        ''' Interpolate a line to fixed number of points.
        
        Args:
            line (LineString): line
        
        Returns:
            points (array): interpolated points, shape (N, 2)
        '''

        distances = np.linspace(0, line.length, self.sample_num)
        sampled_points = np.array([list(line.interpolate(distance).coords) 
            for distance in distances]).squeeze()

        return sampled_points

    def interp_fixed_dist(self, line: LineString):
        ''' Interpolate a line at fixed interval.
        
        Args:
            line (LineString): line
        
        Returns:
            points (array): interpolated points, shape (N, 2)
        '''

        distances = list(np.arange(self.sample_dist, line.length, self.sample_dist))
        # make sure to sample at least two points when sample_dist > line.length
        distances = [0,] + distances + [line.length,] 
        
        sampled_points = np.array([list(line.interpolate(distance).coords)
                                for distance in distances]).squeeze()
        
        return sampled_points
    
    def get_vectorized_lines(self, lanes):
        ''' Vectorize map elements. Iterate over the input dict and apply the 
        specified sample funcion.
        
        Args:
            line (np.ndarray): line
        
        Returns:
            vectors (array): dict of vectorized map elements.
        '''

        new_lanes = []
        for lane in lanes:
            line = LineString(lane)

            # _line = line.simplify(0.2, preserve_topology=True)
            # _line = np.array(_line.coords)
            # line = line
            
            line = self.sample_fn(line)
            line = line[:self.sample_num, :self.coords_dim]

            if self.permute:
                line = self.permute_line(line)
            line = line.astype(np.float32)
            new_lanes.append(line)
        return new_lanes
    
    def permute_line(self, line, padding=1e5):
        '''
        (num_pts, coords_dim) -> (num_permute, num_pts, coords_dim)
        where num_permute = 2 * (num_pts - 1)
        '''
        is_closed = np.allclose(line[0], line[-1], atol=1e-3)
        num_points = len(line)
        permute_num = num_points - 1
        permute_lines_list = []
        if is_closed:
            pts_to_permute = line[:-1, :] # throw away replicate start end pts
            for shift_i in range(permute_num):
                permute_lines_list.append(np.roll(pts_to_permute, shift_i, axis=0))
            flip_pts_to_permute = np.flip(pts_to_permute, axis=0)
            for shift_i in range(permute_num):
                permute_lines_list.append(np.roll(flip_pts_to_permute, shift_i, axis=0))
        else:
            permute_lines_list.append(line)
            permute_lines_list.append(np.flip(line, axis=0))

        permute_lines_array = np.stack(permute_lines_list, axis=0)

        if is_closed:
            tmp = np.zeros((permute_num * 2, num_points, self.coords_dim))
            tmp[:, :-1, :] = permute_lines_array
            tmp[:, -1, :] = permute_lines_array[:, 0, :] # add replicate start end pts
            permute_lines_array = tmp

        else:
            # padding
            padding = np.full([permute_num * 2 - 2, num_points, self.coords_dim], padding)
            permute_lines_array = np.concatenate((permute_lines_array, padding), axis=0)
        
        return permute_lines_array
    
    def __call__(self, lanes):
        # lanes: [np.ndarray: nx3, np.ndarray: mx3]
        new_lanes = self.get_vectorized_lines(lanes)
        return new_lanes

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'sample_num={self.sample_num}), '
        repr_str += f'sample_dist={self.sample_dist}), ' 
        repr_str += f'coords_dim={self.coords_dim})'

        return repr_str
import os
import numpy as np


def calculate_vertival_angle(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    horizontal_distance = np.sqrt(x ** 2 + y ** 2)
    vertical_angle = np.arctan2(z, horizontal_distance)
    return np.degrees(vertical_angle)

def assign_line_number(points, num_lines=128, min_angle=-22.5, max_angle=22.5):
    vertical_angle = calculate_vertival_angle(points)
    angle_step = (max_angle - min_angle) / (num_lines - 1)
    line_number = np.floor((vertical_angle - min_angle) / angle_step).astype(int)
    return line_number

def downsample_pcd(points, original_line_number=128, target_line_number=16):
    line_number = assign_line_number(points, num_lines=original_line_number)
    sorted_indices = np.argsort(line_number)
    sorted_points = points[sorted_indices]
    sorted_line_number = line_number[sorted_indices]
    
    step = original_line_number // target_line_number
    selected_line_number = np.arange(0, original_line_number, step)
    mask = np.isin(sorted_line_number, selected_line_number)
    downsampled_points = sorted_points[mask]
    return downsampled_points


if __name__ == '__main__':
    
    root_path = './data/CODA'
    sequences = list(range(0, 21))

    for seq in sequences:
        pcd_path = os.path.join(root_path, '3d_comp', 'os1', str(seq))
        pcd_files = os.listdir(pcd_path)
        downsampled_pcd_path = os.path.join(root_path, '3d_comp_downsample', 'os1', str(seq))
        os.makedirs(downsampled_pcd_path, exist_ok=True)
        
        for pcd_file in pcd_files:
            pcd_file_path = os.path.join(pcd_path, pcd_file)
            points = np.fromfile(pcd_file_path, dtype=np.float32).reshape(-1, 4)
            downsampled_points = downsample_pcd(points)
            downsampled_pcd_file_path = os.path.join(downsampled_pcd_path, pcd_file)
            downsampled_points.tofile(downsampled_pcd_file_path)
            
        print(f'{seq} done')
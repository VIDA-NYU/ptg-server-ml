import time
import json

import numpy as np
import open3d as o3d
import pyarrow as pa
import pyarrow.parquet as pq


def convert_json_to_parquet(input_path: str, output_path: str=''):
    import tqdm
    print('opening json...')
    with open(input_path) as f:
        pc_data = json.load(f)
    print('opened json...')
    output_path = output_path or input_path.rsplit('.', 1)[0] + '.parquet'

    writer = None
    try:
        for data in tqdm.tqdm(pc_data):
            x, y, z = np.array(data['xyz_world']).T
            r, g, b = np.array(data['color']).T
            table = pa.table({
                "x": x, "y": y, "z": z, "r": r, "g": g, "b": b,
                "timestamp": [int(data['timestamp'].split('-')[0])] * len(data['xyz_world']),
            })
            if writer is None:  # because we need access to the schema before opening
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return output_path

def load_parquet(input_path):
    data = pq.read_table(input_path)
    xyz = np.stack([
        data.column('x').to_numpy(),
        data.column('y').to_numpy(),
        data.column('z').to_numpy(),
    ], axis=1)
    color = np.stack([
        data.column('r').to_numpy(),
        data.column('g').to_numpy(),
        data.column('b').to_numpy(),
    ], 1) / 255.
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(xyz)
    pcl.colors = o3d.utility.Vector3dVector(color)
    return pcl, #data.column('timestamp').to_numpy()


def save_point_cloud(pcl: o3d.geometry.PointCloud, output_path: str):
    points = np.asarray(pcl.points).tolist()
    colors = np.asarray(pcl.colors).tolist()
    normals = []

    with open(output_path, 'w') as f:
        f.write(json.dumps({
            'xyz_world': points, 
            'colors': colors, 
            'normals': normals 
        }))


def run(input_path: str, output_path: str, voxel_size=0.025, draw=False):
    pcl, = load_parquet(input_path)
    downsampled = pcl.voxel_down_sample(voxel_size)
    save_point_cloud(downsampled, output_path)
    if draw:
        o3d.visualization.draw([downsampled])


if __name__ == "__main__":
    import fire
    fire.Fire(main)

    ## Example: 
    ## python voxelization2.py ./data/pointcloud/2023.03.15-20.36.42-pointcloud.json ./outputs/voxelizations/2023.03.15-20.36.42-voxelized-pointcloud.json

from plyfile import PlyData
import numpy as np


def read_vertices(plydata):
    #access vertex data
    vertices = plydata['vertex']
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']

    #read the first 5 vertices
    print("Vertices (x,y,z)")
    for i in range(min(5, len(x))):
        print(x[i], y[i], z[i])


def modify_vertices(plydata, modified_path):
    vertices = plydata['vertex']
    vertex_data = np.array([vertices['x'], vertices['y'], vertices['z']]).T

    #shift the first 5 points
    vertex_data[:vertex_data.shape[0]//2, 0] += 1  # Scale x by 2
    vertex_data[:vertex_data.shape[0]//2, 1] += 1  # Scale y by 2
    vertex_data[:vertex_data.shape[0]//2, 2] += 1  # Scale z by 2

    print(vertex_data.shape)

    # Update vertex data in-place
    vertices['x'] = vertex_data[:, 0]
    vertices['y'] = vertex_data[:, 1]
    vertices['z'] = vertex_data[:, 2]

    #save the data
    with open(modified_path, 'wb') as f:
        print("writing new data")
        PlyData(plydata.elements).write(f)


if __name__ == "__main__":
    file_path = r"point_cloud2.ply"
    modified_path = 'modified_file.ply'

    with open(file_path, 'rb') as f:
        plydata = PlyData.read(f)

    read_vertices(plydata)

    modify_vertices(plydata, modified_path)

    with open(modified_path, 'rb') as f:
        plydata2 = PlyData.read(f)

    read_vertices(plydata2)


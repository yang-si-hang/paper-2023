"""
Generate a triangular mesh file in .vtk file with given meshed data
"""

import numpy as np
import vtk
from scipy.spatial import Delaunay


def mesh_data(shape, seed_size):
    L = shape[0]
    W = shape[1]
    # If the shape can be divided by seed_size, the remainder is 1, otherwise 0
    LN_remain = int(1) if np.mod(L, seed_size) < 1.e-8 else int(0)  # 1e-8 due to the precision problem
    WN_remain = int(1) if np.mod(W, seed_size) < 1.e-8 else int(0)
    LN = int(np.ceil(L / seed_size)) + LN_remain
    WN = int(np.ceil(W / seed_size)) + WN_remain

    xx, yy = np.meshgrid(np.linspace(0, L, LN), np.linspace(0, W, WN))
    xx_pad = xx.flatten()
    yy_pad = yy.flatten()
    node = np.array([xx_pad, yy_pad]).T

    tri = Delaunay(node)

    element = tri.simplices
    # element += 1

    edge_set = set()
    for simplices in element:
        for i in range(3):
            edge_tmp = tuple(sorted([i, (i + 1) % 3]))
            edge_set.add(edge_tmp)

    edge = np.array(list(edge_set))

    data = {'v': node, 'e': edge, 'f': element}

    return data


def main():
    data = mesh_data(shape=[0.1, 0.1], seed_size=0.1)
    nodes = data['v']
    triangles = data['f']

    points = vtk.vtkPoints()
    for node in nodes:
        points.InsertNextPoint(node[0], 0., node[1])

    cells = vtk.vtkCellArray()
    for triangle in triangles:
        # polygon = vtk.vtkTriangle()
        # # polygon.GetPointIds().SetNumberOfIds(3)
        # polygon.GetPointIds().SetId(0, triangle[0])
        # polygon.GetPointIds().SetId(1, triangle[1])
        # polygon.GetPointIds().SetId(2, triangle[2])
        # cells.InsertNextCell(polygon)
        cells.InsertNextCell(3)
        cells.InsertCellPoint(triangle[0])
        cells.InsertCellPoint(triangle[1])
        cells.InsertCellPoint(triangle[2])

    # vtk.vtkUnstructuredGrid

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName('trian.vtk')
    writer.SetInputData(polydata)
    writer.Write()


if __name__ == '__main__':
    main()
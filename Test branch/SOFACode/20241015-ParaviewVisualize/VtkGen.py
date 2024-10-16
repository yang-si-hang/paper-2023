import vtk

def create_triangle_mesh_vtk(output_filename):
    # 创建一个多边形数据对象
    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()

    # 添加四边形的顶点
    points.InsertNextPoint(0.0, 0.0, 0.0)
    points.InsertNextPoint(1.0, 0.0, 0.0)
    points.InsertNextPoint(1.0, 1.0, 0.0)
    points.InsertNextPoint(0.0, 1.0, 0.0)

    # 定义第一个三角形 (0, 1, 2)
    triangle1 = vtk.vtkTriangle()
    triangle1.GetPointIds().SetId(0, 0)
    triangle1.GetPointIds().SetId(1, 1)
    triangle1.GetPointIds().SetId(2, 2)
    triangles.InsertNextCell(triangle1)

    # 定义第二个三角形 (0, 2, 3)
    triangle2 = vtk.vtkTriangle()
    triangle2.GetPointIds().SetId(0, 0)
    triangle2.GetPointIds().SetId(1, 2)
    triangle2.GetPointIds().SetId(2, 3)
    triangles.InsertNextCell(triangle2)

    # 创建多边形数据并设置点和三角形
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetPolys(triangles)

    # 创建一个多边形数据的写入器
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(poly_data)
    writer.Write()

    print(f"VTK 文件已保存为: {output_filename}")

if __name__ == "__main__":
    output_filename = "triangle_mesh.vtk"
    create_triangle_mesh_vtk(output_filename)
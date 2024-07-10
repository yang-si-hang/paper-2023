import vtk

# 读取.msh文件
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName("Mesh/liver.msh")
reader.Update()

# 设置颜色映射
mapper = vtk.vtkDataSetMapper()
mapper.SetInputData(reader.GetOutput())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

# 窗口设置
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

renderer.AddActor(actor)
renderer.SetBackground(1, 1, 1)  # 设置背景颜色为白色

render_window.Render()
render_window_interactor.Start()

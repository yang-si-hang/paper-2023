This file want to describe all code files.

### _PDStrain.py
An **example file** that implement the Projective Dynamics method with strain & volume constraint.

### _PDStrainGrasp.py

### _ControlSimulation.py
Control an edge node to deform the soft object in PD simulation, which make a marker
point on soft object move to a desired position.
The controller is based on the *grad function solver* or *DiffPD techonolgy*.

### ControlMultiPoints.py
Minimize the loss with multi grasping points by DiffPD.

### _PDManipulability.py
Get the manipulability from various contact points to specific feature point.

### _PDStrain3D.py
Simulate the deformation of soft 3D cube object with linear elastic material in PD framework.

### GenMesh.py
生成三维几何体的四面体网格,并保存为.msh文件
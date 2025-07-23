#### demo.py

1. 如何确定 A 矩阵的各个元素？
   是求解 $X_f\cdot X_g^{-1}=A\cdot q$ 中 $A$ 的表达式，并且需要打平，即转换为向量的形式。
   对于二维的场景，$q\in \mathbb{R}^{3}$，对于三维的场景，$q\in \mathbb{R}^4$，相应地，$A$ 的维度也需要调整。
2. 如何确定 Bp 矩阵的各个元素？
   首先要做的是进行 Local Solver，即确定额外变量（Auxiliary variable）
   对于 Strain constraint，未变形的约束就是只有旋转，没有变形，那么 $\mathrm F=\mathrm {U\Sigma V}$ ，$\Sigma=\mathrm I$ 时，是未变形状态，所以 $\mathrm Bp=\mathrm {UV}$
   对于 Area constraint，未变形的约束就是面积不变，即 $\prod \Sigma_i=1$，可以按照 Liu 的 Projective Dynamics 原文附录中的迭代公式求解
3. 为什么程序需要从 Strain 和 Area 两个方面来求解？
   因为对于原有的 Hooke 定律，通过 Lame 参数有：

$$
C=\lambda I \text{tr}(\epsilon)+2\mu\epsilon
$$

其中，可以看到第一部分是 Area constraint，而第二部分是 Strain constraint。
所以，对于线弹性材料，可以将能量函数拆解开。而对于其他材料模型，可以尝试直接从能量函数推导而来。

##  Code File description

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
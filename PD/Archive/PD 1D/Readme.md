### Projective Dynamics对一维柔软绳的变形模拟

约束公式参考了`soler2018cosserat`, 文章中的错误很多, 最明显的是关于J的具体形式, 转动惯量和转动矩混用; Stretch & Bend Weight计算方法也不对; 

`PD1D.py`存在的问题: element & Shear Constraint中的un*不是四元数形式

`PD1D_re_2.py`的问题: 

`PD1D_re_3.py`的问题: 只将两端节点的位置引入,舍去中间的节点,没有给Element Orientation施加单位化的约束,使得每次求解后的Orientation都不是标准的,在施加位移后,节点间的距离会缩小???

`PD1D_re_4.py`的问题: 收敛步数变长,而且有变形"波动传递"的现象,Finite Difference方法难以准确应用在这个模型上,参考`DataAnalyse\20240824-PD 1D grad\FiniteDiffResult.ipynb`中的结果

`TestCoeerateLocalSolver.ipynb`: 查看Local Solve中,迭代结果随迭代次数的变化

`TestGradientCosserat.ipynb`: 验证DiffPD中梯度是否正确

`DiffPD1D.py`: 基于`PD1D.py`的反向模型

`DiffPD1D_re_1.py`: 修改了求平均Quaternion的方法

`DiffPD1D_re_3.py`: Stretch constraint中求Bp的方法是正确的,之前的文件都不是四元数形式

`DiffPD1D_re_4.py`: 三维空间下的变形,Constraint Form和Weight Value与`soler2018cosserat`完全相同

`DiffPD1D_re_5.py`: 修改了Bend Constraint的形式,用两个Element Orientation之差取代与Average Orientation之差
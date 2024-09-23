"""
用于生成Gmsh2.2版本的网格文件，可以用Gmsh软件打开查看
肝脏的颜色为[0.82, 0.15, 0.08]
"""
import numpy as np


def read_elements_from_msh(file_path):
    # 读取 .msh 文件中的四面体元素。
    nodes = []
    elements = []
    in_nodes_section = False
    in_elements_section = False

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('$NOD'):
                in_nodes_section = True
                continue
            elif line.startswith('$ENDNOD'):
                in_nodes_section = False
                continue
            elif line.startswith('$ELM'):
                in_elements_section = True
                continue
            elif line.startswith('$ENDELM'):
                in_elements_section = False
                continue

            if in_nodes_section:
                # 解析节点部分
                parts = line.strip().split()
                if len(parts) == 1:
                    # 这是节点的总数行，跳过
                    continue
                node_id = int(parts[0])
                x, y, z = map(float, parts[1:])
                nodes.append((node_id, x, y, z))

            if in_elements_section:
                # 解析四面体元素部分
                parts = line.strip().split()
                if len(parts) == 1:
                    # 这是单元的总数行，跳过
                    continue
                element_id = int(parts[0])
                node_ids = list(map(int, parts[-4:]))
                elements.append((element_id, node_ids))

    return nodes, elements


def construct_msh(points, cells, file_path):
    # 创建msh格式的数据
    msh_data = "$NOD\n{num_nodes}\n".format(num_nodes=len(points))

    for i, point in enumerate(points):
        # msh_data += "{index} {x:.5f} {y:.5f} {z:.5f}\n".format(index=i+1, x=point[0], y=point[1], z=point[2])
        msh_data += f'{i + 1} {point[0]:.5f} {point[1]:.5f} {point[2]:.5f}\n'

    msh_data += "$ENDNOD\n$ELM\n{num_elements}\n".format(num_elements=len(cells))

    for i, cell in enumerate(cells):
        msh_data += f'{i + 1} 4 1 1 4 {int(cell[0]):d} {int(cell[1]):d} {int(cell[2]):d} {int(cell[3]):d}\n'

    msh_data += "$ENDELM\n"

    # 将数据写入文件
    with open(file_path, 'w') as f:
        f.write(msh_data)

    print(f"Mesh data has been written to {file_path}")


if __name__ == '__main__':
    msh_file = 'liver.msh'
    _, elements = read_elements_from_msh(msh_file)

    node_pos = np.loadtxt('node_end_pos2.csv', delimiter=',')
    nodes = []
    cells = []
    for idx, row in enumerate(node_pos):
        nodes.append((row[0], row[1], row[2]))

    for idx, ele in elements:
        cells.append((ele[0], ele[1], ele[2], ele[3]))

    construct_msh(nodes, cells, 'liver_new2.msh')
import numpy as np


def read_msh_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    nodes = []
    cells = []
    is_node_section = False
    is_cell_section = False

    for line in lines:
        if line.strip() == "$NOD":
            is_node_section = True
            continue
        if line.strip() == "$ENDNOD":
            is_node_section = False
            continue
        if line.strip() == "$ELM":
            is_cell_section = True
            continue
        if line.strip() == "$ENDELM":
            is_cell_section = False
            continue

        if is_node_section:
            parts = line.strip().split()
            if len(parts) == 4:
                index, x, y, z = parts
                nodes.append([float(x), float(y), float(z)])

        if is_cell_section:
            parts = line.strip().split()
            if len(parts) > 4:
                index = parts[0]
                cell_nodes = parts[5:]
                cells.append([int(node) for node in cell_nodes])

    nodes_array = np.array(nodes)
    cells_array = np.array(cells)

    return nodes_array, cells_array


# 使用示例文件路径
file_path = "Mesh/liver.msh"
nodes_array, cells_array = read_msh_file(file_path)

# 打印节点和单元信息
print("节点信息（节点矩阵）：")
print(nodes_array)
print("单元信息（单元矩阵）：")
print(cells_array)

# 保存到文件（可选）
np.savetxt("nodes_array.csv", nodes_array, fmt='%f', delimiter=',')
np.savetxt("cells_array.csv", cells_array, fmt='%d', delimiter=',')

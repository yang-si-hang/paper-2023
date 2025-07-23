import os, sys
import numpy as np
import yaml
from shapely.geometry import LineString, Polygon

script_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(script_dir, '..', ".."))
sys.path.append(root_path)
from Utilize.GenMsh import write_mshv2_tri
os.chdir(script_dir)  # 修改当前工作目录

def read_yaml(file_path:str):
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    nodes = np.array(data["node"])
    edges = np.array(data["edge"])
    triangles = np.array(data["ele"])
    node_mass = np.array(data["node_mass"])

    return nodes, edges, triangles, node_mass


def cut_mesh(nodes, triangles, cut_start, cut_end, tol=1e-8):
    """
    Given a set of nodes and triangles, remove triangles that are cut by the line 
    connecting cut_start and cut_end. Nodes are not modified.
    
    Parameters:
        nodes (np.ndarray): Array of shape (n_nodes, 2) containing node positions.
        triangles (np.ndarray): Array of shape (n_triangles, 3) with triangle connectivity.
        cut_start (tuple or list): (x, y) coordinates of the cut start point.
        cut_end (tuple or list): (x, y) coordinates of the cut end point.
        tol (float): Tolerance for intersection length.
        
    Returns:
        np.ndarray: A new array of triangle connectivity (subset of the original) that 
                    do not have a nontrivial intersection with the cut line.
    """
    cut_line = LineString([cut_start, cut_end])
    new_triangles = []

    for tri in triangles:
        # Build the triangle polygon from the three nodes.
        coords = nodes[tri]
        triangle_poly = Polygon(coords)

        # Find the intersection between the triangle and the cut line.
        intersection = triangle_poly.intersection(cut_line)
        
        # Decide whether to remove this triangle:
        # We want to remove triangles where the cut line passes through,
        # meaning the intersection has a dimension (i.e. is a line rather than just a point).
        remove = False
        if not intersection.is_empty:
            # For line-like intersections, remove the triangle.
            if intersection.geom_type in ['LineString', 'MultiLineString']:
                # Optionally, you can check the length to ignore very small intersections.
                if intersection.length > tol:
                    remove = True
            elif intersection.geom_type == 'GeometryCollection':
                # In a collection, if any geometry is line-like, mark for removal.
                for geom in intersection.geoms:
                    if geom.geom_type in ['LineString', 'MultiLineString'] and geom.length > tol:
                        remove = True
                        break

        # Keep the triangle if it is not affected by the cut.
        if not remove:
            new_triangles.append(tri)
        else:
            print(f"Triangle {tri} removed due to cut line intersection.")
    
    return np.array(new_triangles)


if __name__ == "__main__":
    with open("shape_cut2.yaml", "r") as f:
        data = yaml.safe_load(f)
    # update_tri = cut_mesh(np.array(data["node"]), np.array(data["ele"]), [0.0684, 0.141], [0.0684, 0.14-0.0091])
    # update_tri = cut_mesh(np.array(data["node"]), np.array(data["ele"]), [0.0684, 0.131], [0.074, 0.125])
    # update_tri = cut_mesh(np.array(data["node"]), np.array(data["ele"]), [0.074, 0.125], [0.074, 0.117])
    # update_tri = cut_mesh(np.array(data["node"]), np.array(data["ele"]), [0.051, 0.131], [0.051, 0.121])
    # update_tri = cut_mesh(np.array(data["node"]), np.array(data["ele"]), [0.051, 0.121], [0.047, 0.106])
    update_tri = cut_mesh(np.array(data["node"]), np.array(data["ele"]), [0.047, 0.106], [0.051, 0.098])
    data["ele"] = update_tri.tolist()
    np.savetxt("cut_ele.csv", np.array(data["ele"]), fmt="%d", delimiter=",")
    with open("shape_cut4.yaml", "w") as f:
        yaml.dump(data, f)
    print("Mesh updated successfully.")

    write_mshv2_tri("shape_cut4.msh", np.array(data["node"]), np.array(data["ele"]))
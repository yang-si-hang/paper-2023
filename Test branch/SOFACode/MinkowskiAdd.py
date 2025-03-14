"""用于计算两个凸多边形的Minkowski和的函数, 以及计算Minkowski和在给定方向上的投影的函数
created at 2025-03-07 by hsy
"""
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'


def min_index(poly):
    idx = 0
    for i in range(1, len(poly)):
        if poly[i][0] < poly[idx][0] or (poly[i][0] == poly[idx][0] and poly[i][1] < poly[idx][1]):
            idx = i
    return idx


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def intersect_ray_segment(d, p, q, eps=1e-10):
    """Find the intersection of a ray starting at the origin in the direction d with the segment between p and q.
    
    Args:
        d (np.array): Direction vector of the ray (should be nonzero).
        p (np.array): First endpoint of the segment.
        q (np.array): Second endpoint of the segment.
        eps (float, optional): Tolerance for detecting parallelism. Defaults to 1e-10.
    
    Returns:
        tuple or None: (t, point) if intersection exists (with t >= 0 and u in [0,1]),
            None otherwise.
    """
    v = q - p
    # Compute 2D cross product (scalar)
    cross_d_v = np.cross(d, v)
    if np.abs(cross_d_v) < eps:
        # The ray and segment are parallel or collinear
        return None

    # Using cross product formulas:
    t = np.cross(p, v) / cross_d_v
    u = np.cross(p, d) / cross_d_v
    
    if t >= 0 and 0 <= u <= 1:
        return t, p + u * v
    return None


def generate_polygon(J, theta_start_deg, n_points, rect_width=0.1):
    """Generate a convex polygon by transforming points on a unit semicircle.
    
    Args:
        J (np.ndarray): Jacobian (linear transformation) matrix of shape (2,2).
        theta_start_deg (float): Starting angle in degrees for the semicircle.
            The semicircle is taken as the arc from theta_start_deg to theta_start_deg + 180.
        n_points (int): Number of discretization points along the arc.
         
    Returns:
        np.ndarray: Points array of shape (n_points, 2) containing the vertices
            (in order) of the transformed semicircle (semi-ellipse).
    """
    # Generate angles uniformly in degrees from start to start+180 (inclusive)
    angles_deg = np.linspace(theta_start_deg, theta_start_deg + 180, n_points)
    angles_rad = np.deg2rad(angles_deg)
    
    # These points lie on the arc from theta_start to theta_start+180
    arc_points = np.column_stack((np.cos(angles_rad), np.sin(angles_rad)))
    # Close the semicircle by including the center (this gives a convex polygon).
    semicircle = np.vstack((arc_points, np.array([0, 0])))

    # Generate the rectangle polygon.
    # The chord (straight edge) of the semicircle is given by the endpoints:
    theta0 = np.deg2rad(theta_start_deg)
    theta1 = np.deg2rad(theta_start_deg + 180)
    p0 = np.array([np.cos(theta0), np.sin(theta0)])
    p1 = np.array([np.cos(theta1), np.sin(theta1)])

    # Compute the chord vector and its midpoint.
    chord_vec = p1 - p0
    chord_mid = (p0 + p1) / 2
    # The semicircular arc’s “middle” (the apex of the arc) lies at angle theta_start_deg + 90.
    arc_mid = np.array([np.cos(np.deg2rad(theta_start_deg + 90)), np.sin(np.deg2rad(theta_start_deg + 90))])
    
    # Get a unit normal to the chord.
    n = np.array([-chord_vec[1], chord_vec[0]])
    n = n / np.linalg.norm(n)
    # Flip the normal if it points toward the semicircular arc.
    if np.dot(n, arc_mid - chord_mid) > 0:
        n = -n

    # Define the rectangle so that one of its long sides is the chord (from p0 to p1)
    # and it extends in the n direction by rect_width.
    rectangle = np.array([
        p0,
        p1,
        p1 + rect_width * n,
        p0 + rect_width * n
    ])

    # Apply the Jacobian transformation to both polygons.
    T_semicircle = semicircle @ J.T
    T_rectangle = rectangle @ J.T

    poly_points = np.row_stack((T_semicircle[:-1], T_rectangle[2:4]))
    
    return poly_points


def minkowski_sum(poly1, poly2):
    # plot the manipulability set of the two polygons
    # poly1_plot = np.vstack([poly1, poly1[0]])
    # poly2_plot = np.vstack([poly2, poly2[0]])
    # plt.figure(figsize=(8, 6))
    # plt.scatter(poly1[:,0], poly1[:,1], color='blue', label='Manipulability Set 1')
    # plt.scatter(poly2[:,0], poly2[:,1], color='green', label='Manipulability Set 2')
    # plt.plot(poly1_plot[:, 0], poly1_plot[:, 1], 'b-', lw=2, label='Manipulability Set 1')
    # plt.plot(poly2_plot[:, 0], poly2_plot[:, 1], 'g-', lw=2, label='Manipulability Set 2')
    # plt.fill(poly1_plot[:, 0], poly1_plot[:, 1], 'b', alpha=0.3)
    # plt.fill(poly2_plot[:, 0], poly2_plot[:, 1], 'g', alpha=0.3)
    # plt.xlim(-0.2, 0.2)
    # plt.ylim(-0.2, 0.2)
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.legend()
    # # plt.axis('equal')
    # plt.grid(True)
    # plt.show()
    
    n1 = len(poly1)
    n2 = len(poly2)

    # Find the vertex with the smallest x (and then y) for each polygon.
    i = min_index(poly1)
    j = min_index(poly2)

    # Initialize the Minkowski sum with the sum of the two minimal points.
    res = [poly1[i] + poly2[j]]

    # Merge the edge vectors of poly1 and poly2.
    # The loop runs exactly (n1 + n2) times.
    for _ in range(n1 + n2):
        next_i = (i + 1) % n1
        next_j = (j + 1) % n2
        edge1 = poly1[next_i] - poly1[i]
        edge2 = poly2[next_j] - poly2[j]
        # Choose the edge with the larger polar angle using cross product.
        if cross(edge1, edge2) >= 0:
            res.append(res[-1] + edge1)
            i = next_i
        else:
            res.append(res[-1] + edge2)
            j = next_j

    # Remove the last point if it is the same as the first.
    if np.allclose(res[0], res[-1]):
        res.pop()

    sum_points = np.array(res)

    hull = ConvexHull(sum_points)
    hull_points = sum_points[hull.vertices]
    hull_points = np.vstack([hull_points, hull_points[0]])

    # plt.figure(figsize=(8, 6))
    # plt.scatter(sum_points[:,0], sum_points[:,1], color='blue', label='Sum Points')
    # plt.plot(hull_points[:, 0], hull_points[:, 1], 'r-', lw=2, label='Convex Hull')
    
    # plt.title("Minkowski Sum: Sum Points and Convex Hull")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return sum_points


def project_polygon(poly, direction):
    """Project the polygon in a given unit direction and return the largest scalar value.
    
    Args:
        poly: A numpy array of shape (n,2) containing vertices of the convex polygon.
        direction: A numpy array of shape (2,) representing the vector direction.
         
    Returns:
        float: The maximum dot product of the polygon's vertices with the direction.
    """
    direction_unit = direction / np.linalg.norm(direction)

    # Compute the convex hull
    hull = ConvexHull(poly)
    hull_points = poly[hull.vertices]

    intersections = []
    for i in range(len(hull_points)):
        p = hull_points[i]
        q = hull_points[(i + 1) % len(hull_points)]
        result = intersect_ray_segment(direction_unit, p, q)
        if result is not None:
            t, inter_point = result
            intersections.append(inter_point)

    intersections = np.array(intersections)

    # # Plotting the convex hull, ray, and intersection points.
    # plt.figure(figsize=(6, 6))
    # # Draw the convex hull polygon
    # plt.plot(np.append(hull_points[:, 0], hull_points[0, 0]), 
    #         np.append(hull_points[:, 1], hull_points[0, 1]), 'b-', label='Convex Hull')
    # # Draw all points
    # plt.scatter(poly[:, 0], poly[:, 1], c='gray', alpha=0.5, label='Points')
    # # Draw the ray (extend it for visualization)
    # ray_length = 3
    # plt.plot([0, ray_length * direction[0]], [0, ray_length * direction[1]], 'r-', label='Ray')
    # # Draw the intersection points, if any.
    # if intersections.size:
    #     plt.scatter(intersections[:, 0], intersections[:, 1], c='green', s=100, label='Intersection(s)')
    # # Mark the origin
    # plt.scatter(0, 0, c='black', label='Origin')

    # plt.legend()
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Intersection of Ray and Convex Hull')
    # plt.gca().set_aspect('equal')
    # plt.show()

    return np.linalg.norm(intersections, axis=1).max()


def main():
    # ----- Step 1: Define parameters and create two semi-ellipses -----
    # Parameters for the first semi-ellipse:
    J1 = np.array([[0.15323373, 0.07527931],
                   [0.00805035, 0.06296659]])
    theta1_start = 90
    n_points1 = 20                          # Number of discretization points
    
    # Parameters for the second semi-ellipse:
    J2 = np.array([[0.11386716, -0.02124374],
                   [0.06940202,  0.05995755]])
    theta2_start = 0
    n_points2 = 20
    
    # Generate the two convex polygons (each approximates a semi-ellipse).
    poly1 = generate_polygon(J1, theta1_start, n_points1)
    poly2 = generate_polygon(J2, theta2_start, n_points2)
    
    # ----- Step 2: The polygons have been discretized evenly in degrees. -----
    # (Note: The polygons are defined by their vertices along the arc of the unit semicircle,
    #  transformed by the Jacobian matrix.)
    
    # ----- Step 3: Compute the Minkowski sum of the two polygons -----
    minkowski_poly = minkowski_sum(poly1, poly2)
    
    # ----- Step 4: Project the Minkowski sum polygon in a given direction -----
    # Define a unit direction vector (for example, choose an arbitrary direction).
    direction = np.array([0.6, 0.8])
    # Normalize the direction (just to be safe)
    direction = direction / np.linalg.norm(direction)
    
    max_proj = project_polygon(minkowski_poly, direction)
    
    # Print the result (largest scalar value)
    print("The largest scalar projection value is:", max_proj)


if __name__ == "__main__":
    main()

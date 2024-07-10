"""
获得相机的初始参数,以及软体的坐标系
棋盘格摆放的位置需要与软体坐标系的原点只有X方向上的偏移，参考'Data/zed_chess_preset.png'
"""

import numpy as np
from CameraChess import get_camera_intrinsic, get_border


def init_param():
    """
    获得软组织相对于相机的变换矩阵
    """
    # 棋盘格的大小和每个棋盘格的格子宽度（单位：米）
    chessboard_size, square_size = (11, 8), 0.015
    intrisic_matrix, dist, image_chess = get_camera_intrinsic()

    *_, transformation_matrix = get_border(chessboard_size, square_size, image_chess, intrisic_matrix, dist)

    # 将棋盘格坐标系转换到软体坐标系
    transformation_chess_soft = np.array([[1., 0., 0., -0.025],
                                          [0., 1., 0., 0.],
                                          [0., 0., 1., 0.003],
                                          [0., 0., 0., 1.]])

    transformation_soft = transformation_matrix @ transformation_chess_soft

    return transformation_soft, intrisic_matrix


if __name__ == '__main__':
    trans_soft, intrinsic = init_param()
    np.savez('data/camera_param.npz', matrix1=trans_soft, matrix2=intrinsic)
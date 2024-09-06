"""
记录标记球的位置
created at 2024-09-02 by hsy
"""

import os
import time
import asyncio
import xml.etree.ElementTree as ET
import pkg_resources
import qtm_rt
import numpy as np
import matplotlib.pyplot as plt


POINTS_NUM = 4
points_pos = np.zeros((POINTS_NUM, 3))

QTM_FILE = pkg_resources.resource_filename("qtm_rt", "data/Demo.qtm")
qualysis_ip = '192.168.253.1'
qualysis_password = ''


def on_packet(packet):
    header, markers = packet.get_3d_markers_no_label()
    if header.marker_count != POINTS_NUM:
        print(f'Not enough markers. Only {header.marker_count} markers detected.')
        return

    # print('----------------------------------')
    for idx, marker in enumerate(markers):
        points_pos[idx, :] = np.array([marker.x, marker.y, marker.z])
        # print(f"Marker: {marker.id}. Position: ({marker.x}, {marker.y}, {marker.z})")
    # print(points_pos)


async def receive_qualysis(connection):
    captured_data = []

    # Define the callback to capture data
    def on_packet2(packet):
        nonlocal captured_data
        header, markers = packet.get_3d_markers_no_label()
        if header.marker_count != POINTS_NUM:
            print(f'Not enough markers. Only {header.marker_count} markers detected.')
            return

        markers_pos = []
        for idx, marker in enumerate(markers):
            # 转换到单位米
            markers_pos.append([marker.x/1000., marker.y/1000., marker.z/1000.])
            # captured_data.append([marker.x/1000., marker.y/1000., marker.z/1000.])
            # print(f"Marker: {marker.id}. Position: ({marker.x/1000.}, {marker.y/1000.}, {marker.z/1000.})")

        captured_data = markers_pos

    await connection.stream_frames(components=["3dnolabels"], on_packet=on_packet2)

    return captured_data


async def main():
    connection = await qtm_rt.connect(qualysis_ip)
    if connection is None:
        print("Failed to connect")
        return

    # Take control of qtm, context manager will automatically release control after scope end
    # async with qtm_rt.TakeControl(connection, "password"):
    async with qtm_rt.TakeControl(connection, qualysis_password):
        realtime = True
        if realtime:
            # Start new realtime
            await connection.new()
        else:
            # Load qtm file
            await connection.load(QTM_FILE)
            # start rtfromfile
            await connection.start(rtfromfile=True)

    try:
        while True:
            # await connection.stream_frames(components=["3dnolabels"], on_packet=on_packet)
            # print(points_pos)
            captured_data = await receive_qualysis(connection)
            time.sleep(0.1)
            print('----------------------------')
            print(captured_data)
    except KeyboardInterrupt:
        print("Stopping the data stream...")

    # 停止数据流
    await connection.stream_frames_stop()


if __name__ == "__main__":
    asyncio.run(main())
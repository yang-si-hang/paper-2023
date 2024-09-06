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

file_name = os.path.dirname(__file__) + '/data/qualisys_rigids_record.csv'
qualysis_ip = '192.168.253.1'
qualysis_password = ''

QTM_FILE = pkg_resources.resource_filename("qtm_rt", "data/Demo.qtm")

def create_body_index(xml_string):
    """ Extract a name to index dictionary from 6dof settings xml """
    xml = ET.fromstring(xml_string)
    # xml = ET.fromstring(xml_string)

    body_to_index = {}
    for index, body in enumerate(xml.findall("*/Body/Name")):
        body_to_index[body.text.strip()] = index

    return body_to_index


def body_enabled_count(xml_string):
    xml = ET.fromstring(xml_string)
    return sum(enabled.text == "true" for enabled in xml.findall("*/Body/Enabled"))


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


async def data_plot(points):
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], 'ro')
    plt.show()

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
            await connection.stream_frames(components=["3dnolabels"], on_packet=on_packet)
            print(points_pos)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopping the data stream...")

    # 停止数据流
    await connection.stream_frames_stop()


if __name__ == "__main__":
    asyncio.run(main())
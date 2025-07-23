"""用于将 .pvsm(paraview) 文件中的数据路径更改
created at 2025-02-26 by deepseek
"""
import os
import xml.etree.ElementTree as ET
import re
# 设置工作目录为当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)  # 修改当前工作目录

def modify_pvsm(input_path, output_path):
    tree = ET.parse(input_path)
    root = tree.getroot()

    # 查找所有FileName属性的Property元素
    for prop in root.findall(".//Property[@name='FileName']"):
        # 遍历每个Element子节点
        for elem in prop.findall("Element"):
            old_value = elem.get('value')
            # 替换目录部分
            new_value = old_value.replace(
                "/data/", 
                "/DataAnalyse/20250226-Gallbladder Explosure/"
            )
            # 替换文件名中的数字为四位格式
            new_value = re.sub(
                r'_(\d+)\.vtu$',
                lambda x: f"_{int(x.group(1)):04d}.vtu", 
                new_value
            )
            elem.set('value', new_value)

    # 保存修改后的XML
    tree.write(output_path, encoding='UTF-8', xml_declaration=True)

if __name__ == '__main__':
    modify_pvsm('20250225-contact-101-change.pvsm', 'modified.pvsm')
"""调用`ContactOptimal`进行接触优化
created at 2025-03-10 by hsy
"""
import os
import itertools
import yaml
import numpy as np
import subprocess


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)  # 修改当前工作目录

def main():
    contact_angle = {11:90, 22:90, 33:90, 44:90, 55:90, 66:90, 77:90, 88:90, 99:90,
                    21:-90, 32:-90, 43:-90, 54:-90, 65:-90, 76:-90, 87:-90, 98:-90, 109:-90,
                    111:0, 112:0, 113:0, 114:0, 115:0, 116:0, 117:0, 118:0, 119:0}

    # marker = [103, 104]
    # marker = [92, 104]
    marker = [83, 93]

    manipulability_dict = {}

    contact_all = list(contact_angle.keys())
    contact_combs = list(itertools.combinations(contact_all, 2))
    for i, contact in enumerate(contact_combs):
        print(f"{i}-th Contact combiantion: {contact} ---------------------------")
        cmd = ["python", "ContactOptimal.py", "--contact"] + [str(x) for x in contact] + ["--marker"] + [str(x) for x in marker]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        full_output = result.stdout
        # print("Full output:", full_output)
        final_result_line = None
        for line in full_output.splitlines():
            if line.startswith("Manipulability:"):
                final_result_line = line
                break

        if final_result_line is not None:
            # Extract the numeric value after the marker.
            try:
                output_str = final_result_line.split("Manipulability:")[1].strip()
                output_float = float(output_str)
                manipulability_dict[tuple(contact)] = output_float
                print("Output (as float):", output_float)
            except ValueError:
                print("Failed to convert final result to float.")
        else:
            print("Final result marker not found in output.")

    # Save the manipulability dictionary to a YAML file
    with open("manipulability.yaml", 'w') as file:
        yaml.dump(manipulability_dict, file)


if __name__ == "__main__":
    main()
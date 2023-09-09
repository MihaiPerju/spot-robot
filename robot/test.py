import mujoco
from mujoco import viewer


# opening the scene
xml_file_path = 'scene.xml'

with open(xml_file_path, 'r') as file:
    xml = file.read()

spot = mujoco.MjModel.from_xml_string(xml)
spot_data = mujoco.MjData(spot)

viewer.launch_passive(spot)
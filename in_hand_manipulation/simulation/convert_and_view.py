import mujoco
import mujoco.viewer
import time

print("1. Converting URDF to MuJoCo XML...")
model = mujoco.MjModel.from_xml_path("gripper_only.urdf")
mujoco.mj_saveLastXML("gripper_model.xml", model)
print("Saved gripper_model.xml successfully!")

data = mujoco.MjData(model)

print("2. Launching interactive viewer...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Simulation running! Use mouse/trackpad to interact.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)


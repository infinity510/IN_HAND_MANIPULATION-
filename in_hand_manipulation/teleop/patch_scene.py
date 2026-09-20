import re

with open('../simulation/scene_cube.xml', 'r') as f:
    content = f.read()

new_bodies = """    <!-- Object 1: Cylinder -->
    <body name="obj_1" pos="10 0 0.1">
      <freejoint name="obj_1_joint"/>
      <geom name="obj_1_geom" type="cylinder" size="0.035 0.07" mass="0.5" rgba="1 0 0 1" friction="1 0.005 0.0001" contype="0" conaffinity="1" solimp="0.99 0.99 0.001" solref="0.002 1"/>
    </body>
    <!-- Object 2: Capsule (substitute for cone) -->
    <body name="obj_2" pos="11 0 0.1">
      <freejoint name="obj_2_joint"/>
      <geom name="obj_2_geom" type="capsule" size="0.035 0.07" mass="0.5" rgba="0 1 0 1" friction="1 0.005 0.0001" contype="0" conaffinity="1" solimp="0.99 0.99 0.001" solref="0.002 1"/>
    </body>
    <!-- Object 3: Cube -->
    <body name="obj_3" pos="12 0 0.1">
      <freejoint name="obj_3_joint"/>
      <geom name="obj_3_geom" type="box" size="0.035 0.035 0.035" mass="0.5" rgba="0 0 1 1" friction="1 0.005 0.0001" contype="0" conaffinity="1" solimp="0.99 0.99 0.001" solref="0.002 1"/>
    </body>
    <!-- Object 4: Cuboid -->
    <body name="obj_4" pos="13 0 0.1">
      <freejoint name="obj_4_joint"/>
      <geom name="obj_4_geom" type="box" size="0.035 0.035 0.07" mass="0.5" rgba="1 1 0 1" friction="1 0.005 0.0001" contype="0" conaffinity="1" solimp="0.99 0.99 0.001" solref="0.002 1"/>
    </body>
    <!-- Object 5: Sphere -->
    <body name="obj_5" pos="14 0 0.1">
      <freejoint name="obj_5_joint"/>
      <geom name="obj_5_geom" type="sphere" size="0.04" mass="0.5" rgba="1 0 1 1" friction="1 0.005 0.0001" contype="0" conaffinity="1" solimp="0.99 0.99 0.001" solref="0.002 1"/>
    </body>"""

pattern = r'<body name="cube" pos="0 0 0.1">.*?</body>'
content = re.sub(pattern, new_bodies, content, flags=re.DOTALL)

with open('../simulation/scene_cube.xml', 'w') as f:
    f.write(content)

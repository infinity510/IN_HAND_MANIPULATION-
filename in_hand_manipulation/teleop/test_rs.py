import pyrealsense2 as rs
import time

ctx = rs.context()
devices = ctx.query_devices()
if len(devices) > 0:
    print("Resetting camera...")
    devices[0].hardware_reset()
    time.sleep(3) # Wait for camera to come back

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("Starting pipeline...")
pipeline.start(config)

try:
    print("Waiting for frames...")
    frames = pipeline.wait_for_frames()
    print("Got frames!")
finally:
    pipeline.stop()

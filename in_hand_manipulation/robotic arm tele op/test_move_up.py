import time
from piper_sdk import C_PiperInterface_V2

def main():
    print("Connecting to Piper-X on can0...")
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    
    # 1. Reset any previous errors
    piper.MotionCtrl_1(0x02, 0, 0)
    time.sleep(0.5)
    
    print("Enabling arm...")
    while not piper.EnablePiper():
        time.sleep(0.01)
    print("Arm enabled!")

    # Current safe home position based on where the arm actually is right now
    # X=287.8, Y=17.9, Z=287.6, RX=-90.4, RY=9.9, RZ=-92.5
    pos_home = [287.8, 17.9, 287.6, -90.4, 9.9, -92.5]
    # Position 2: Move UP by 30mm (Z + 30)
    pos_up   = [287.8, 17.9, 317.6, -90.4, 9.9, -92.5]

    FACTOR = 1000

    print(f"\nPosition 1 (Home): X={pos_home[0]}mm Y={pos_home[1]}mm Z={pos_home[2]}mm")
    print(f"Position 2 (Up):   X={pos_up[0]}mm Y={pos_up[1]}mm Z={pos_up[2]}mm")
    print("\nStreaming commands... Watch the arm move between two heights.\n")

    count = 0
    position = pos_home

    try:
        while True:
            count += 1
            if count == 1:
                position = pos_home
                print(">>> Moving to HOME position")
            elif count == 200:
                position = pos_up
                print(">>> Moving UP (+30mm)")
            elif count == 400:
                position = pos_home
                print(">>> Moving back to HOME")
                count = 0

            X  = round(position[0] * FACTOR)
            Y  = round(position[1] * FACTOR)
            Z  = round(position[2] * FACTOR)
            RX = round(position[3] * FACTOR)
            RY = round(position[4] * FACTOR)
            RZ = round(position[5] * FACTOR)

            piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
            piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)

            print(f"  SDK: X={X} Y={Y} Z={Z} RX={RX} RY={RY} RZ={RZ}   ", end="\r")
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        print("Disabling arm...")
        try:
            piper.DisablePiper()
        except:
            pass
        print("Done.")

if __name__ == "__main__":
    main()

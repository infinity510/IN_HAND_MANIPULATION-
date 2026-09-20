import time
from piper_sdk import C_PiperInterface_V2

def main():
    print("Connecting to Piper-X on can0...")
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    
    print("Sending reset/recovery command...")
    # MotionCtrl_1(0x02, 0, 0) is the recovery command from the SDK demo
    piper.MotionCtrl_1(0x02, 0, 0)
    time.sleep(1)
    
    print("Checking status after reset...")
    print(piper.GetArmStatus())
    print("\nAttempting to Enable arm...")
    success = piper.EnablePiper()
    print(f"EnablePiper returned: {success}")
    
    if success:
        print("Success! Arm is enabled. You can now run the other scripts.")
    else:
        print("Still failed. You may need to manually move the arm to the center and restart it.")

if __name__ == "__main__":
    main()


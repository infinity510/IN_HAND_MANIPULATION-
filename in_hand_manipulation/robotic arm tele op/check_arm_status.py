import time
from piper_sdk import C_PiperInterface_V2

def main():
    print("Connecting to Piper-X on can0...")
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    
    print("Checking arm status...")
    # Fetch status for a few frames
    for i in range(10):
        print(piper.GetArmStatus())
        time.sleep(0.1)
        
    print("\nAttempting to Enable arm...")
    success = piper.EnablePiper()
    print(f"EnablePiper returned: {success}")

if __name__ == "__main__":
    main()


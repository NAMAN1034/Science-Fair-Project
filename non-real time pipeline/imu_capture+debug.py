import serial
import pandas as pd
import time
import sys

# CONFIGURATION
PORT = "/dev/cu.usbmodem187570801" 
BAUD = 115200                       
DURATION = 30                        
FILENAME = "imu_capture.csv"

data = []

print(f"Connecting to {PORT}...")

try:
    # Using 'with' automatically handles opening and closing safely
    with serial.Serial(PORT, BAUD, timeout=2) as ser:
        print("Device found. Waiting 3s for Arduino to stabilize...")
        # Give the Mac and Arduino a long time to finish the 'reboot handshake'
        time.sleep(3)
        ser.reset_input_buffer()
        
        print(f"Recording for {DURATION} seconds... Start motion now!")
        start_time = time.time()

        while (time.time() - start_time) < DURATION:
            try:
                # Direct read is more stable on Mac than checking in_waiting
                line_bytes = ser.readline()
                if not line_bytes:
                    continue
                
                line = line_bytes.decode('utf-8', errors='ignore').strip()
                
                if line:
                    parts = line.split(',')
                    # Convert to floats, skipping any empty strings
                    nums = [float(p.strip()) for p in parts if p.strip()]
                    
                    if len(nums) >= 6:
                        # Append system time if Arduino only sends 6 values
                        if len(nums) == 6:
                            nums.append(time.time())
                        data.append(nums[:7])
                        
                        if len(data) % 100 == 0:
                            print(f"Captured {len(data)} samples...", end='\r')
            
            except (UnicodeDecodeError, ValueError):
                # Skip partial lines or "Initializing..." text
                continue
            except Exception as e:
                print(f"\nRead error: {e}")
                break # Exit loop if the hardware actually disconnects

except serial.SerialException as e:
    print(f"\nFATAL ERROR: {e}")
    print("TIP: Try unplugging/replugging the USB cable and closing Arduino IDE.")
    sys.exit()

# SAVE DATA
if data:
    df = pd.DataFrame(data, columns=["ax", "ay", "az", "gx", "gy", "gz", "time"])
    df.to_csv(FILENAME, index=False)
    print(f"\n\nSUCCESS: Saved {len(data)} rows to {FILENAME}")
    print(f"Sample Rate: {len(data)/DURATION:.2f} Hz")
else:
    print("\nNo data was captured. If this persists, check your Arduino code's Serial.print format.")
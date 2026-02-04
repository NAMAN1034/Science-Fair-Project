import serial
import time
import csv
import sys

# CONFIG 
PORT = "/dev/tty.usbmodem187570801"
BAUD = 115200
DURATION_SEC = 30
OUTPUT_FILE = "emg_raw_capture.csv"

# CONNECT 
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # Arduino reset time
except Exception as e:
    print("SERIAL CONNECTION FAILED")
    print(e)
    sys.exit(1)

print(f"Connected to {PORT}")
print("⏺Recording for", DURATION_SEC, "seconds...")
print("👉 START TREMOR MOTION NOW")

# READ 
lines = []
start_time = time.time()

while time.time() - start_time < DURATION_SEC:
    line = ser.readline().decode("utf-8", errors="ignore").strip()

    if line:
        lines.append(line)

        # Print first few lines for sanity
        if len(lines) <= 5:
            print("DATA:", line)

ser.close()

# VALIDATION
if len(lines) < 5:
    print("ERROR: No data was captured.")
    print("Check:")
    print("• Serial Monitor is CLOSED")
    print("• Arduino is running")
    print("• Correct PORT")
    sys.exit(1)

# SAVE CSV
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    for row in lines:
        writer.writerow(row.split(","))

print(f"Saved {len(lines)} samples to {OUTPUT_FILE}")

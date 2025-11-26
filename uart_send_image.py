#!/usr/bin/env python3
import struct
import sys
import time
import serial
from pathlib import Path

PKT_START = 0xA5
PKT_CMD_IMAGE = 0x01
PKT_STOP = 0x5A

# Choose serial port and baud; edit as needed
DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 115200

# Read an image header that defines: static const uint32_t img_X_Y[100] = {...};
# We'll parse it very simply, extracting 100 hex values.
def read_header_uint32_list(header_path: Path):
    text = header_path.read_text()
    vals = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("0x"):
            parts = [p.strip() for p in line.split(",") if p.strip()]
            for p in parts:
                if p.startswith("0x"):
                    vals.append(int(p, 16))
    if len(vals) < 100:
        raise ValueError(f"Expected at least 100 values, got {len(vals)}")
    return vals[:100]


def send_image_bits(ser: serial.Serial, words):
    # Frame: start, cmd, len_lo, len_hi, 100 words (little-endian), stop
    ser.write(bytes([PKT_START, PKT_CMD_IMAGE, 100 & 0xFF, (100 >> 8) & 0xFF]))
    for u in words:
        ser.write(struct.pack('<I', u))
    ser.write(bytes([PKT_STOP]))
    ser.flush()


def main():
    port = DEFAULT_PORT
    baud = DEFAULT_BAUD
    header = None
    if len(sys.argv) >= 2:
        header = Path(sys.argv[1])
    else:
        print("Usage: uart_send_image.py <path/to/img_header.h> [port] [baud]", file=sys.stderr)
        sys.exit(1)
    if len(sys.argv) >= 3:
        port = sys.argv[2]
    if len(sys.argv) >= 4:
        baud = int(sys.argv[3])

    words = read_header_uint32_list(header)
    print(f"Loaded {len(words)} pixels from {header}")

    ser = serial.Serial(port=port, baudrate=baud, timeout=5)
    time.sleep(0.2)
    send_image_bits(ser, words)
    # Read one byte result
    resp = ser.read(1)
    if len(resp) == 0:
        print("No response from MB (timeout)", file=sys.stderr)
        sys.exit(2)
    pred = resp[0]
    print(f"Predicted class: {pred}")
    ser.close()

if __name__ == '__main__':
    main()

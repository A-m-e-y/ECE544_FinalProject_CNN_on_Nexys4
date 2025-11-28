#!/usr/bin/env python3
import struct
import sys
import time
import serial
from pathlib import Path
import re

PKT_START = 0xA5
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
    # Simplified frame: START (0xA5), 100 words (400 bytes LE), STOP (0x5A)
    ser.write(bytes([PKT_START]))
    for u in words:
        ser.write(struct.pack('<I', u))
    ser.write(bytes([PKT_STOP]))
    ser.flush()


def flush_input(ser: serial.Serial, quiet_duration=0.3):
    """Drain any existing bytes (boot logs) until we see quiet for quiet_duration seconds."""
    ser.timeout = 0.05
    last_rx = time.time()
    while time.time() - last_rx < quiet_duration:
        data = ser.read(256)
        if data:
            last_rx = time.time()
    # After draining, clear buffer pointer
    ser.reset_input_buffer()


def read_logs_and_pred(ser: serial.Serial, overall_timeout=15.0):
    """Stream xil_printf output live to stdout and capture predicted class.
    Returns (pred_from_line, raw_pred_byte, full_captured_text).
    """
    ser.timeout = 0.05
    start = time.time()
    buf_chars = []
    pred_line_val = None
    raw_byte = None
    saw_pred_line = False
    # We'll continue a short window after pred line to grab the raw byte
    post_pred_deadline = None
    while time.time() - start < overall_timeout:
        chunk = ser.read(128)
        if chunk:
            for b in chunk:
                if 32 <= b <= 126 or b in (10, 13):  # printable or newline
                    ch = chr(b)
                    buf_chars.append(ch)
                    # Echo immediately for user visibility
                    sys.stdout.write(ch)
                    sys.stdout.flush()
                else:
                    # candidate raw pred byte
                    if saw_pred_line and raw_byte is None:
                        raw_byte = b
                # Check for prediction line in current joined text
            joined = ''.join(buf_chars)
            if not saw_pred_line and 'PREDICTED CLASS:' in joined:
                # robust parse using regex; take last match
                matches = list(re.finditer(r"PREDICTED\s+CLASS:\s*(\d+)", joined))
                if matches:
                    pred_line_val = int(matches[-1].group(1))
                    saw_pred_line = True
                    post_pred_deadline = time.time() + 1.0  # wait up to 1s for raw byte
        else:
            if saw_pred_line:
                # If waiting for raw byte and deadline passed, break
                if post_pred_deadline and time.time() > post_pred_deadline:
                    break
    return pred_line_val, raw_byte, ''.join(buf_chars)


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

    ser = serial.Serial(port=port, baudrate=baud, timeout=0.1)
    time.sleep(0.2)
    print("Flushing stale boot logs (if any)...")
    flush_input(ser)
    print("Sending image frame (A5 ... data ... 5A)...")
    send_image_bits(ser, words)
    print("Streaming logs (live):")
    pred_line_val, pred_raw, logs = read_logs_and_pred(ser)
    print("\n--- End of log stream ---")
    if pred_line_val is not None:
        print(f"Parsed line prediction: {pred_line_val}")
    else:
        print("Did not parse PREDICTED CLASS line", file=sys.stderr)
    if pred_raw is not None:
        print(f"Raw prediction byte (post-line): {pred_raw}")
    else:
        print("No separate raw byte captured (class likely printable or not sent)")
    final_pred = pred_line_val if pred_line_val is not None else pred_raw
    if final_pred is not None:
        print(f"Final predicted class: {final_pred}")
    else:
        print("Prediction not obtained", file=sys.stderr)
    ser.close()

if __name__ == '__main__':
    main()

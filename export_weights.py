#!/usr/bin/env python3
"""
Export trained reduced CNN weights from a binary file to a C header (weights.h).
Assumes little-endian float32 sequence in the following order:
- conv1 weights: (Cin=1, kH=3, kW=3, Cout=8) -> K1=9, length K1*C1 = 9*8
- conv1 bias: length 8
- conv2 weights: (Cin=8, kH=3, kW=3, Cout=16) -> K2=72, length 72*16
- conv2 bias: length 16
- conv3 weights: (Cin=16, kH=3, kW=3, Cout=32) -> K3=144, length 144*32
- conv3 bias: length 32
- classifier weights: (C3=32, NUM_CLASSES=10) length 32*10
- classifier bias: length 10

Weights stored column-stacked per out_channel: flatten (in_channel, kH, kW) for each out_channel.
This matches matmul A(1,K) * B(K,Cout).
"""
import struct
import sys
from pathlib import Path

C1, C2, C3 = 8, 16, 32
NUM_CLASSES = 10
K1 = 1*3*3
K2 = C1*3*3
K3 = C2*3*3

layout = [
    ("WEIGHTS_CONV1_BITS", K1*C1),
    ("BIAS_CONV1_BITS", C1),
    ("WEIGHTS_CONV2_BITS", K2*C2),
    ("BIAS_CONV2_BITS", C2),
    ("WEIGHTS_CONV3_BITS", K3*C3),
    ("BIAS_CONV3_BITS", C3),
    ("WEIGHTS_CLS_BITS", C3*NUM_CLASSES),
    ("BIAS_CLS_BITS", NUM_CLASSES),
]

def read_floats(bin_path: Path):
    data = bin_path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError("Binary file size is not a multiple of 4 bytes")
    return list(struct.unpack("<{}f".format(len(data)//4), data))

def to_u32_bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]

def write_header(floats, out_path: Path):
    # Verify total length
    total_needed = sum(length for _, length in layout)
    if len(floats) < total_needed:
        raise ValueError(f"Insufficient floats in bin: have {len(floats)}, need {total_needed}")
    # Emit header
    with out_path.open("w") as fh:
        fh.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
        fh.write("#include <stdint.h>\n\n")
        fh.write("#define C1 8\n#define C2 16\n#define C3 32\n#define NUM_CLASSES 10\n\n")
        offset = 0
        for name, length in layout:
            fh.write(f"const uint32_t {name}[{length}] = \n")
            fh.write("{\n")
            for i in range(length):
                bits = to_u32_bits(floats[offset + i])
                fh.write(f"  0x{bits:08X},")
                if (i+1) % 8 == 0:
                    fh.write("\n")
            fh.write("\n};\n\n")
            offset += length
        fh.write("#endif // WEIGHTS_H\n")

def main():
    bin_path = Path("trained_reduced_cnn.bin")
    out_path = Path("weights.h")
    if len(sys.argv) >= 2:
        bin_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2])
    floats = read_floats(bin_path)
    write_header(floats, out_path)
    print(f"Wrote {out_path} from {bin_path}")

if __name__ == "__main__":
    main()

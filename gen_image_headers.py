#!/usr/bin/env python3
"""Generate C header files for dataset images.

Scans ../Dataset/Dataset_10x10/<class>/*.jpg and produces:
  generated_images/<class>/img_<class>_<index>.h  (uint32_t array with IEEE-754 float bits)
  dataset_arrays.h aggregating all samples.

Each pixel converted to grayscale (already grayscale), normalized to [0,1] float32,
then represented as 0xXXXXXXXX (hex bits) in row-major order (height-major first).

Re-run whenever dataset updates (adding classes 1..9).
"""
import os, sys, struct
from PIL import Image

IMG_SIZE = 10
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'Dataset_10x10'))
OUT_DIR = os.path.join(os.path.dirname(__file__), 'generated_images')

def float_to_hex_bits(f: float) -> str:
    # IEEE754 float32 little-endian to uint32_t
    b = struct.pack('>f', f)  # big-endian to get consistent bit pattern
    u = struct.unpack('>I', b)[0]
    return f"0x{u:08X}"

def process_image(path: str):
    img = Image.open(path).convert('L').resize((IMG_SIZE, IMG_SIZE))
    arr = list(img.getdata())  # length IMG_SIZE*IMG_SIZE
    hex_vals = []
    for y in range(IMG_SIZE):
        for x in range(IMG_SIZE):
            v = arr[y*IMG_SIZE + x] / 255.0
            hex_vals.append(float_to_hex_bits(v))
    return hex_vals

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_samples = []  # (header_guard, rel_header_path, label, symbol_name)
    for cls in range(10):
        cls_dir = os.path.join(ROOT, str(cls))
        if not os.path.isdir(cls_dir):
            break  # assume contiguous 0..N-1
        out_cls = os.path.join(OUT_DIR, str(cls))
        os.makedirs(out_cls, exist_ok=True)
        images = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith('.jpg') or f.lower().endswith('.png')])
        for idx, fname in enumerate(images):
            src_path = os.path.join(cls_dir, fname)
            hex_vals = process_image(src_path)
            symbol = f"img_{cls}_{idx}"
            guard = f"IMG_{cls}_{idx}_H"
            header_path = os.path.join(out_cls, f"{symbol}.h")
            with open(header_path, 'w') as hf:
                header_intro = (
                    "#ifndef {g}\n#define {g}\n#include <stdint.h>\n"
                    "static const uint32_t {sym}[{sz}] = {{\n".format(
                        g=guard, sym=symbol, sz=IMG_SIZE*IMG_SIZE)
                )
                hf.write(header_intro)
                # format values 10 per line
                for i, hv in enumerate(hex_vals):
                    sep = ',' if i < len(hex_vals)-1 else ''
                    if i % 10 == 0:
                        hf.write("    ")
                    hf.write(hv + sep + ' ')
                    if i % 10 == 9:
                        hf.write("\n")
                hf.write("}};\n#endif // {g}\n".format(g=guard))
            rel_header = os.path.relpath(header_path, os.path.dirname(__file__))
            all_samples.append((guard, rel_header, cls, symbol))
    # Write aggregator header
    dataset_header = os.path.join(os.path.dirname(__file__), 'dataset_arrays.h')
    with open(dataset_header, 'w') as dh:
        dh.write("#ifndef DATASET_ARRAYS_H\n#define DATASET_ARRAYS_H\n#include <stdint.h>\n\n")
        for _, rel, _, _ in all_samples:
            dh.write(f"#include \"{rel}\"\n")
        dh.write("\ntypedef struct { const uint32_t *data; int label; } ImageSample;\n")
        dh.write(f"static const int DATASET_IMG_SIZE = {IMG_SIZE};\n")
        dh.write(f"static const int DATASET_NUM_SAMPLES = {len(all_samples)};\n")
        max_label = max([s[2] for s in all_samples]) if all_samples else 0
        dh.write(f"static const int DATASET_NUM_CLASSES = {max_label+1};\n")
        dh.write("static const ImageSample training_samples[] = {\n")
        for _, _, lbl, sym in all_samples:
            dh.write(f"    {{ {sym}, {lbl} }},\n")
        dh.write("};\n\n#endif // DATASET_ARRAYS_H\n")
    print(f"Generated {len(all_samples)} samples across {max_label+1} classes.")

if __name__ == '__main__':
    main()

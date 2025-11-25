# C Implementation (Standalone Reduced CNN)

This folder contains a pure C inference-only implementation of a reduced CNN architecture that avoids large dense layers to keep on-chip memory requirements modest.

## Architecture
- conv1: 1 -> 8 channels, 3x3, stride=1, padding=1, ReLU
- conv2: 8 -> 16 channels, 3x3, stride=1, padding=1, ReLU
- conv3: 16 -> 32 channels, 3x3, stride=1, padding=1, ReLU
- conv_cls: 32 -> 10 channels, 1x1 convolution (logits per spatial position)
- Global Average Pooling over 10x10 -> 10 logits
- Softmax -> probabilities

Largest kernel footprint (conv3): in_channels * 3 * 3 = 16 * 9 = 144 elements per output channel.
This means if later mapped to your accelerator via im2col, worst-case K=144, M=100 windows (10x10), N=32 (or 10 for classifier). So suitable IP parameter choices:
- MAX_M = 100
- MAX_K = 144
- MAX_N = 32

## Files
- `reduced_cnn.h` / `reduced_cnn.c`: Core network and utilities (random init, load/save, forward).
- `main.c`: Simple test harness creating a dummy input and printing probabilities.
- `Makefile`: Local build script (does not alter parent project).

## Build & Run
```bash
cd c_impl
make
./reduced_cnn_demo
```

Save weights (binary dump of struct):
```bash
./reduced_cnn_demo save
```
Load previously saved weights:
```bash
./reduced_cnn_demo load
```

## Weight Binary Layout
The binary produced by `save` is a raw dump of the `ReducedCNNWeights` struct in this order:
1. conv1_w (8*1*3*3 floats)
2. conv1_b (8)
3. conv2_w (16*8*3*3)
4. conv2_b (16)
5. conv3_w (32*16*3*3)
6. conv3_b (32)
7. convcls_w (10*32*1*1)
8. convcls_b (10)

Total floats: 8*9 + 16*8*9 + 32*16*9 + 10*32 + biases (8+16+32+10) = 72 + 1152 + 4608 + 320 + 66 = 6218 floats (~24.9 KB @ 4 bytes each).

## Next Integration Steps
Later you can:
1. Replace direct conv loops with im2col + accelerator matmul calls for each layer.
2. Add UART image ingestion and use the same forward routine (swap out input buffer).
3. Optionally fuse bias + ReLU to reduce passes.

## Notes
- This does NOT modify existing Python or top-level Makefile.
- Pure C, no external deps beyond libm (implicitly linked).
- Deterministic random init (xorshift) for reproducibility.

## Potential Extensions
- Add simple fixed-point mode for FPGA resource reduction.
- Introduce optional 2x2 max-pooling after conv2 or conv3 to further shrink activation size.
- Export weights from original Python model by converting to this reduced architecture (requires a mapping script).

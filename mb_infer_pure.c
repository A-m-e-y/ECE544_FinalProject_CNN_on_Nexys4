#include <stdint.h>
#include <stdio.h>

// Include generated weights and a sample image header.
// After running export_weights.py, ensure weights.h is present.
#include "weights.h"
// Pick any generated image header; you can change this include to test another.
#include "generated_images/3/img_3_26.h"

// Network dimensions (fixed, keep small)
#define IMG_SIZE 10
#define C1 8
#define C2 16
#define C3 32
#define NUM_CLASSES 10

// Fast bit->float and float->bit helpers (avoid lib overhead)
static inline float bits_to_float(uint32_t u)
{
    union
    {
        uint32_t u;
        float f;
    } cvt;
    cvt.u = u;
    return cvt.f;
}
static inline uint32_t float_to_bits(float f)
{
    union
    {
        uint32_t u;
        float f;
    } cvt;
    cvt.f = f;
    return cvt.u;
}

// Relu via sign bit zeroing
static inline uint32_t relu_bits(uint32_t x) { return (x & 0x80000000u) ? 0u : x; }

// Accumulate dot: A(1,K) * B(K,N) -> C(1,N)
static void matmul_row(const uint32_t *A_bits, const uint32_t *B_bits, int K, int N, uint32_t *C_bits)
{
    // Convert A row once to float for speed (still small K)
    float acc;
    for (int n = 0; n < N; ++n)
    {
        acc = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            float a = bits_to_float(A_bits[k]);
            // B layout from saved struct is oc-major contiguous (columns), index as [n*K + k]
            float b = bits_to_float(B_bits[n * K + k]);
            acc += a * b;
        }
        C_bits[n] = float_to_bits(acc);
    }
}

// Conv im2col for one position (h,w), zero-padding, Cin x 3x3 -> K
static void im2col_row_3x3(const uint32_t *img_bits, int Cin, int H, int W, int h, int w, uint32_t *A_bits)
{
    int idx = 0;
    for (int c = 0; c < Cin; ++c)
    {
        for (int kh = -1; kh <= 1; ++kh)
        {
            int ih = h + kh;
            for (int kw = -1; kw <= 1; ++kw)
            {
                int iw = w + kw;
                uint32_t v = 0u; // zero-padding
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                {
                    // Only Cin=1 uses img_bits; higher layers use activation buffers
                    v = img_bits[c * H * W + ih * W + iw];
                }
                A_bits[idx++] = v;
            }
        }
    }
}

// Conv forward: Cin->Cout, K=Cin*3*3, for all positions HxW, ReLU
static void conv_forward(const uint32_t *input_bits, int Cin, int Cout, int H, int W,
                         const uint32_t *W_bits /* K*Cout */, const uint32_t *bias_bits /* Cout */,
                         uint32_t *out_bits)
{
    const int K = Cin * 9;
    // Live debug print: dimensions of the matmul performed per position
    // Each spatial position computes (1xK) * (KxCout)
    printf("[matmul] Conv: M=%d, K=%d, N=%d (Cin=%d, Cout=%d, H=%d, W=%d)\n", H * W, K, Cout, Cin, Cout, H, W);
    uint32_t A_row[9 * 32]; // max K=144 -> 9*32; small stack buffer
    for (int h = 0; h < H; ++h)
    {
        for (int w = 0; w < W; ++w)
        {
            // Build A row
            // We'll store outputs channel-major: [oc][h][w]
            // im2col
            // For layers > 1, input_bits already has Cin channels
            int idx = 0;
            for (int c = 0; c < Cin; ++c)
            {
                for (int kh = -1; kh <= 1; ++kh)
                {
                    int ih = h + kh;
                    for (int kw = -1; kw <= 1; ++kw)
                    {
                        int iw = w + kw;
                        uint32_t v = 0u;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        {
                            v = input_bits[c * H * W + ih * W + iw];
                        }
                        A_row[idx++] = v;
                    }
                }
            }
            // Matmul 1xK * KxCout
            uint32_t tmp[32];
            matmul_row(A_row, W_bits, K, Cout, tmp);
            // Add bias and ReLU
            for (int oc = 0; oc < Cout; ++oc)
            {
                float y = bits_to_float(tmp[oc]) + bits_to_float(bias_bits[oc]);
                out_bits[(oc * H + h) * W + w] = relu_bits(float_to_bits(y));
            }
        }
    }
}

// GAP + classifier: average over HxW then fc to NUM_CLASSES
static void classifier_forward(const uint32_t *act_bits /* channel-major [c][h][w] */, int H, int W,
                               const uint32_t *W_bits /* C3*NUM_CLASSES */, const uint32_t *bias_bits /* NUM_CLASSES */,
                               int *pred)
{
    // Live debug print: GAP -> (1xC3) * (C3xNUM_CLASSES)
    printf("[matmul] Classifier: M=1, K=%d, N=%d (after GAP)\n", C3, NUM_CLASSES);
    // Compute channel averages (GAP)
    float gap[C3];
    for (int c = 0; c < C3; ++c)
        gap[c] = 0.0f;
    for (int c = 0; c < C3; ++c)
    {
        int base = c * H * W;
        for (int h = 0; h < H; ++h)
        {
            for (int w = 0; w < W; ++w)
            {
                gap[c] += bits_to_float(act_bits[base + h * W + w]);
            }
        }
    }
    float invHW = 1.0f / (float)(H * W);
    for (int c = 0; c < C3; ++c)
        gap[c] *= invHW;

    // FC: gap(1xC3) * W(C3x10) + bias
    float logits[NUM_CLASSES];
    for (int n = 0; n < NUM_CLASSES; ++n)
    {
        float acc = 0.0f;
        for (int k = 0; k < C3; ++k)
        {
            // convcls_w is oc-major contiguous per class: index as [n*C3 + k]
            float w = bits_to_float(W_bits[n * C3 + k]);
            acc += gap[k] * w;
        }
        logits[n] = acc + bits_to_float(bias_bits[n]);
    }

    // Argmax
    int best = 0;
    float bestv = logits[0];
    for (int n = 1; n < NUM_CLASSES; ++n)
    {
        if (logits[n] > bestv)
        {
            bestv = logits[n];
            best = n;
        }
    }
    *pred = best;
}

int main(void)
{
    // Input image bits from header (grayscale, 10x10, single channel)
    const uint32_t *img_bits = img_3_26; // name provided by included header

    // Buffers: keep minimal sizes
    // act1: 10x10x8, act2: 10x10x16, act3: 10x10x32
    static uint32_t act1[IMG_SIZE * IMG_SIZE * C1];
    static uint32_t act2[IMG_SIZE * IMG_SIZE * C2];
    static uint32_t act3[IMG_SIZE * IMG_SIZE * C3];

    // Live prints are emitted inside conv_forward() and classifier_forward().

    // Conv1
    conv_forward(img_bits, 1, C1, IMG_SIZE, IMG_SIZE, WEIGHTS_CONV1_BITS, BIAS_CONV1_BITS, act1);
    // Conv2
    conv_forward(act1, C1, C2, IMG_SIZE, IMG_SIZE, WEIGHTS_CONV2_BITS, BIAS_CONV2_BITS, act2);
    // Conv3
    conv_forward(act2, C2, C3, IMG_SIZE, IMG_SIZE, WEIGHTS_CONV3_BITS, BIAS_CONV3_BITS, act3);

    // Classifier
    int pred = -1;
    classifier_forward(act3, IMG_SIZE, IMG_SIZE, WEIGHTS_CLS_BITS, BIAS_CLS_BITS, &pred);

    // Print only final class to keep output small
    printf("Predicted class: %d\n", pred);
    return 0;
}

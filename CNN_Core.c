#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include "platform.h"
#include "xparameters.h"
#include "AXI_MatrixMulEngine.h"
#include "weights.h" // const uint32_t WEIGHTS_* and BIAS_* arrays

#include "CNN_Core.h"


// Bit/float helpers
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

// Optimized: preload B once per layer, then only stream A rows.
static void preload_B_layer(const uint32_t *W_bits, int K, int N)
{
    static float Bf[144 * 32]; // max K*N
    // Hardware driver expects B in row-major [k][n] -> index k*N + n
    // Our weights are stored oc-major contiguous: W_bits[n*K + k]
    for (int k = 0; k < K; ++k)
    {
        for (int n = 0; n < N; ++n)
        {
            Bf[k * N + n] = bits_to_float(W_bits[n * K + k]);
        }
    }
    // accel_load_B(Bf, (uint32_t)K, (uint32_t)N);
    matrixmul_write_matrix_b(MATRIXMUL_BASEADDR, Bf, (uint32_t)K, (uint32_t)N);
}

// Single row matmul with current preloaded B: we still need to write A fresh.
static void matmul_row_preloadedB(const uint32_t *A_bits, int K, int N, uint32_t *C_bits)
{
    static float Af[144];
    static float Cf[32];
    for (int k = 0; k < K; ++k)
        Af[k] = bits_to_float(A_bits[k]);
    // accel_load_A(Af, 1, (uint32_t)K);
    matrixmul_write_matrix_a(MATRIXMUL_BASEADDR, Af, 1, (uint32_t)K);
    // accel_start_wait();
    matrixmul_start(MATRIXMUL_BASEADDR);
    matrixmul_wait_done(MATRIXMUL_BASEADDR);
    // accel_read_C(Cf, 1, (uint32_t)N);
    matrixmul_read_matrix_c(MATRIXMUL_BASEADDR, Cf, 1, (uint32_t)N);
    for (int n = 0; n < N; ++n)
        C_bits[n] = float_to_bits(Cf[n]);
}

// Convolution layer using IP (channel-major activations) 3x3 pad=1
static void conv_layer_ip(const uint32_t *in_bits, int Cin, int Cout, int H, int W,
                          const uint32_t *W_bits, const uint32_t *bias_bits, uint32_t *out_bits)
{
    const int K = Cin * 9;
    xil_printf("[conv] Cin=%d Cout=%d K=%d H=%d W=%d\n", Cin, Cout, K, H, W);
    // accel_set_dims(1, (uint32_t)K, (uint32_t)Cout);
    matrixmul_set_dimensions(MATRIXMUL_BASEADDR, 1, (uint32_t)K, (uint32_t)Cout);
    preload_B_layer(W_bits, K, Cout);
    uint32_t A_row[144];
    uint32_t tmp[32];
    for (int h = 0; h < H; ++h)
    {
        for (int w = 0; w < W; ++w)
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
                        uint32_t v = 0u;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            v = in_bits[(c * H + ih) * W + iw];
                        A_row[idx++] = v;
                    }
                }
            }
            matmul_row_preloadedB(A_row, K, Cout, tmp);
            for (int oc = 0; oc < Cout; ++oc)
            {
                float y = bits_to_float(tmp[oc]) + bits_to_float(bias_bits[oc]);
                uint32_t yb = float_to_bits(y);
                out_bits[(oc * H + h) * W + w] = (yb & 0x80000000u) ? 0u : yb;
            }
        }
        if ((h & 3) == 0)
            xil_printf("  row %d/%d done\n", h, H - 1); // periodic progress
    }
    // Print a few sample activations for verification
    xil_printf("[conv] sample out ch0 pix(0,0)=0x%08lx pix(5,5)=0x%08lx\n", (unsigned long)out_bits[0 * H * W + 0], (unsigned long)out_bits[0 * H * W + 5 * W + 5]);
}

static int classifier_ip(const uint32_t *act3_bits, int H, int W,
                         const uint32_t *W_bits, const uint32_t *bias_bits)
{
    xil_printf("[cls] GAP + FC C3=%d classes=%d\n", C3, NUM_CLASSES);
    float gap[32];
    for (int c = 0; c < C3; ++c)
        gap[c] = 0.0f;
    for (int c = 0; c < C3; ++c)
    {
        int base = c * H * W;
        for (int i = 0; i < H * W; ++i)
            gap[c] += bits_to_float(act3_bits[base + i]);
        gap[c] *= (1.0f / (float)(H * W));
    }
    // Prepare B once
    // accel_set_dims(1, C3, NUM_CLASSES);
    matrixmul_set_dimensions(MATRIXMUL_BASEADDR, 1, C3, NUM_CLASSES);
    static float Bf[32 * 10];
    // Arrange B as row-major [k][n] for the driver/IP: index k*NUM_CLASSES + n
    for (int k = 0; k < C3; ++k)
    {
        for (int n = 0; n < NUM_CLASSES; ++n)
        {
            Bf[k * NUM_CLASSES + n] = bits_to_float(W_bits[n * C3 + k]);
        }
    }
    // accel_load_B(Bf, C3, NUM_CLASSES);
    matrixmul_write_matrix_b(MATRIXMUL_BASEADDR, Bf, C3, NUM_CLASSES);
    // A
    // accel_load_A(gap, 1, C3);
    matrixmul_write_matrix_a(MATRIXMUL_BASEADDR, gap, 1, C3);
    // accel_start_wait();
    matrixmul_start(MATRIXMUL_BASEADDR);
    matrixmul_wait_done(MATRIXMUL_BASEADDR);
    static float Cf[10];
    // accel_read_C(Cf, 1, NUM_CLASSES);
    matrixmul_read_matrix_c(MATRIXMUL_BASEADDR, Cf, 1, NUM_CLASSES);
    int best = 0;
    float bestv = Cf[0] + bits_to_float(bias_bits[0]);
    for (int n = 1; n < NUM_CLASSES; ++n)
    {
        float v = Cf[n] + bits_to_float(bias_bits[n]);
        if (v > bestv)
        {
            bestv = v;
            best = n;
        }
    }
    xil_printf("best = %d\n", best);
    return best;
}

int run_inference_ip(const uint32_t *img)
{
    static uint32_t act1[IMG_SIZE * IMG_SIZE * C1];
    static uint32_t act2[IMG_SIZE * IMG_SIZE * C2];
    static uint32_t act3[IMG_SIZE * IMG_SIZE * C3];
    xil_printf("[infer] starting\n");
    conv_layer_ip(img, 1, C1, IMG_SIZE, IMG_SIZE, 
        WEIGHTS_CONV1_BITS, BIAS_CONV1_BITS, act1);
    conv_layer_ip(act1, C1, C2, IMG_SIZE, IMG_SIZE, 
        WEIGHTS_CONV2_BITS, BIAS_CONV2_BITS, act2);
    conv_layer_ip(act2, C2, C3, IMG_SIZE, IMG_SIZE, 
        WEIGHTS_CONV3_BITS, BIAS_CONV3_BITS, act3);
    int pred = classifier_ip(act3, IMG_SIZE, IMG_SIZE, 
        WEIGHTS_CLS_BITS, BIAS_CLS_BITS);
    xil_printf("[infer] done pred=%d\n", pred);
    return pred;
}

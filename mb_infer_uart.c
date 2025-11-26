#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include "platform.h"
#include "xparameters.h"
#include "AXI_MatrixMulEngine.h"

#include "weights.h" // const uint32_t WEIGHTS_* and BIAS_* arrays

// UARTLite MMIO registers (from xparameters.h canonical base)
#define UARTLITE_BASE XPAR_XUARTLITE_0_BASEADDR
#define UART_RX_FIFO (UARTLITE_BASE + 0x00)
#define UART_TX_FIFO (UARTLITE_BASE + 0x04)
#define UART_STATUS (UARTLITE_BASE + 0x08)
#define UART_CONTROL (UARTLITE_BASE + 0x0C)

#define UART_STATUS_RX_VALID (1u << 0)
#define UART_STATUS_TX_FULL (1u << 3)

#define MATRIXMUL_BASEADDR XPAR_AXI_MATRIXMULENGINE_0_BASEADDR

#define IMG_SIZE 10
#define C1 8
#define C2 16
#define C3 32
#define NUM_CLASSES 10

// Protocol bytes
#define PKT_START 0xA5
#define PKT_CMD_IMAGE 0x01
#define PKT_STOP 0x5A

// Helpers: bit-float conversions
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

// UART helpers (blocking)
static inline uint8_t uart_recv_byte(void)
{
    while ((AXI_MATRIXMULENGINE_mReadReg(UART_STATUS, 0) & UART_STATUS_RX_VALID) == 0)
    {
    }
    return (uint8_t)(AXI_MATRIXMULENGINE_mReadReg(UART_RX_FIFO, 0) & 0xFF);
}
static inline void uart_send_byte(uint8_t b)
{
    while (AXI_MATRIXMULENGINE_mReadReg(UART_STATUS, 0) & UART_STATUS_TX_FULL)
    {
    }
    AXI_MATRIXMULENGINE_mWriteReg(UART_TX_FIFO, 0, (uint32_t)b);
}

// Receive 100 words (400 bytes) into img_bits[100] using little-endian assembly
static int recv_image(uint32_t *img_bits)
{
    uint8_t start = uart_recv_byte();
    if (start != PKT_START)
        return -1;
    uint8_t cmd = uart_recv_byte();
    if (cmd != PKT_CMD_IMAGE)
        return -2;
    // Length: two bytes little-endian (100)
    uint8_t len_lo = uart_recv_byte();
    uint8_t len_hi = uart_recv_byte();
    uint16_t words = (uint16_t)(len_lo | (len_hi << 8));
    if (words != 100)
        return -3;
    for (uint16_t i = 0; i < words; ++i)
    {
        uint8_t b0 = uart_recv_byte();
        uint8_t b1 = uart_recv_byte();
        uint8_t b2 = uart_recv_byte();
        uint8_t b3 = uart_recv_byte();
        uint32_t u = (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
        img_bits[i] = u;
    }
    uint8_t stop = uart_recv_byte();
    if (stop != PKT_STOP)
        return -4;
    return 0;
}

// Accelerator helpers
static inline void accel_set_dims(uint32_t M, uint32_t K, uint32_t N)
{
    matrixmul_set_dimensions(MATRIXMUL_BASEADDR, M, K, N);
}
static inline void accel_load_A_float(const float *A, uint32_t M, uint32_t K)
{
    matrixmul_write_matrix_a(MATRIXMUL_BASEADDR, A, M, K);
}
static inline void accel_load_B_float(const float *B, uint32_t K, uint32_t N)
{
    matrixmul_write_matrix_b(MATRIXMUL_BASEADDR, B, K, N);
}
static inline void accel_start_wait(void)
{
    matrixmul_start(MATRIXMUL_BASEADDR);
    matrixmul_wait_done(MATRIXMUL_BASEADDR);
}
static inline void accel_read_C_float(float *C, uint32_t M, uint32_t N)
{
    matrixmul_read_matrix_c(MATRIXMUL_BASEADDR, C, M, N);
}

// Matmul for one row using IP: inputs/weights are provided as uint32 bits
static void matmul_row_ip(const uint32_t *A_bits, const uint32_t *B_bits, int K, int N, uint32_t *C_bits)
{
    // Convert to float buffers
    static float Af[144];      // max K
    static float Bf[144 * 32]; // max K*N
    static float Cf[32];
    for (int k = 0; k < K; ++k)
        Af[k] = bits_to_float(A_bits[k]);
    // B is oc-major contiguous: [n*K + k]
    for (int n = 0; n < N; ++n)
    {
        for (int k = 0; k < K; ++k)
        {
            Bf[n * K + k] = bits_to_float(B_bits[n * K + k]);
        }
    }
    accel_set_dims(1, (uint32_t)K, (uint32_t)N);
    accel_load_A_float(Af, 1, (uint32_t)K);
    accel_load_B_float(Bf, (uint32_t)K, (uint32_t)N);
    accel_start_wait();
    accel_read_C_float(Cf, 1, (uint32_t)N);
    for (int n = 0; n < N; ++n)
        C_bits[n] = float_to_bits(Cf[n]);
}

// Conv: input channel-major [c][h][w], output channel-major, 3x3 pad=1, ReLU via sign bit
static void conv_layer_ip(const uint32_t *in_bits, int Cin, int Cout, int H, int W,
                          const uint32_t *W_bits /* K*Cout */, const uint32_t *bias_bits /* Cout */,
                          uint32_t *out_bits)
{
    const int K = Cin * 9;
    uint32_t A_row[144]; // max K
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
                        {
                            v = in_bits[(c * H + ih) * W + iw];
                        }
                        A_row[idx++] = v;
                    }
                }
            }
            // matmul
            matmul_row_ip(A_row, W_bits, K, Cout, tmp);
            // bias + ReLU
            for (int oc = 0; oc < Cout; ++oc)
            {
                float y = bits_to_float(tmp[oc]) + bits_to_float(bias_bits[oc]);
                uint32_t yb = float_to_bits(y);
                out_bits[(oc * H + h) * W + w] = (yb & 0x80000000u) ? 0u : yb;
            }
        }
    }
}

// Classifier: GAP -> A(1x32) * B(32x10) via IP, then bias and argmax
static int classifier_ip(const uint32_t *act3_bits, int H, int W,
                         const uint32_t *W_bits /* 32*10 */, const uint32_t *bias_bits /* 10 */)
{
    float gap[32];
    for (int c = 0; c < C3; ++c)
        gap[c] = 0.0f;
    for (int c = 0; c < C3; ++c)
    {
        int base = c * H * W;
        for (int i = 0; i < H * W; ++i)
        {
            gap[c] += bits_to_float(act3_bits[base + i]);
        }
        gap[c] *= (1.0f / (float)(H * W)); // consider replacing with constant bits on MB
    }
    // Prepare float A(1x32)
    static float Af[32];
    for (int k = 0; k < C3; ++k)
        Af[k] = gap[k];
    // Prepare float B(32x10) from bits (oc-major contiguous)
    static float Bf[32 * 10];
    for (int n = 0; n < NUM_CLASSES; ++n)
    {
        for (int k = 0; k < C3; ++k)
        {
            Bf[n * C3 + k] = bits_to_float(W_bits[n * C3 + k]);
        }
    }
    static float Cf[10];
    accel_set_dims(1, C3, NUM_CLASSES);
    accel_load_A_float(Af, 1, C3);
    accel_load_B_float(Bf, C3, NUM_CLASSES);
    accel_start_wait();
    accel_read_C_float(Cf, 1, NUM_CLASSES);
    // bias + argmax
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
    return best;
}

// Top-level inference using IP
static int run_inference_ip(const uint32_t *img_bits)
{
    static uint32_t act1[IMG_SIZE * IMG_SIZE * C1];
    static uint32_t act2[IMG_SIZE * IMG_SIZE * C2];
    static uint32_t act3[IMG_SIZE * IMG_SIZE * C3];

    // Conv1, Conv2, Conv3
    conv_layer_ip(img_bits, 1, C1, IMG_SIZE, IMG_SIZE, WEIGHTS_CONV1_BITS, BIAS_CONV1_BITS, act1);
    conv_layer_ip(act1, C1, C2, IMG_SIZE, IMG_SIZE, WEIGHTS_CONV2_BITS, BIAS_CONV2_BITS, act2);
    conv_layer_ip(act2, C2, C3, IMG_SIZE, IMG_SIZE, WEIGHTS_CONV3_BITS, BIAS_CONV3_BITS, act3);
    // Classifier
    int pred = classifier_ip(act3, IMG_SIZE, IMG_SIZE, WEIGHTS_CLS_BITS, BIAS_CLS_BITS);
    return pred;
}

int main(void)
{
    init_platform();
    uint32_t input_img[100];
    while (1)
    {
        int rc = recv_image(input_img);
        if (rc == 0)
        {
            int pred = run_inference_ip(input_img);
            uart_send_byte((uint8_t)pred);
        }
        else
        {
            uart_send_byte(0xEE); // error code
        }
    }
    cleanup_platform();
    return 0;
}

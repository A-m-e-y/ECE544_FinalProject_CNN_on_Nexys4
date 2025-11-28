#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include "platform.h"
#include "xparameters.h"
#include "AXI_MatrixMulEngine.h"
#include "sleep.h"
#include "xtmrctr.h"
#include "xil_io.h"
#include "weights.h" // const uint32_t WEIGHTS_* and BIAS_* arrays

// Network constants (must match training / weights)
#define IMG_SIZE 10
#define C1 8
#define C2 16
#define C3 32
#define NUM_CLASSES 10

#define MATRIXMUL_BASEADDR XPAR_AXI_MATRIXMULENGINE_0_BASEADDR

// UARTLite MMIO registers (from xparameters.h canonical base)
#define UARTLITE_BASE XPAR_XUARTLITE_0_BASEADDR
#define UART_RX_FIFO (UARTLITE_BASE + 0x00)
#define UART_TX_FIFO (UARTLITE_BASE + 0x04)
#define UART_STATUS (UARTLITE_BASE + 0x08)
#define UART_CONTROL (UARTLITE_BASE + 0x0C)

#define UART_STATUS_RX_VALID (1u << 0)
#define UART_STATUS_TX_FULL (1u << 3)

// UART protocol bytes (match uart_send_image.py)
#define PKT_START 0xA5
#define PKT_CMD_IMAGE 0x01
#define PKT_STOP 0x5A

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

// Accelerator helper wrappers (reuse existing driver API)
static inline void accel_set_dims(uint32_t M, uint32_t K, uint32_t N) { matrixmul_set_dimensions(MATRIXMUL_BASEADDR, M, K, N); }
static inline void accel_load_A(const float *A, uint32_t M, uint32_t K) { matrixmul_write_matrix_a(MATRIXMUL_BASEADDR, A, M, K); }
static inline void accel_load_B(const float *B, uint32_t K, uint32_t N) { matrixmul_write_matrix_b(MATRIXMUL_BASEADDR, B, K, N); }
static inline void accel_start_wait(void)
{
    matrixmul_start(MATRIXMUL_BASEADDR);
    matrixmul_wait_done(MATRIXMUL_BASEADDR);
}
static inline void accel_read_C(float *C, uint32_t M, uint32_t N) { matrixmul_read_matrix_c(MATRIXMUL_BASEADDR, C, M, N); }

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
    accel_load_B(Bf, (uint32_t)K, (uint32_t)N);
}

// Single row matmul with current preloaded B: we still need to write A fresh.
static void matmul_row_preloadedB(const uint32_t *A_bits, int K, int N, uint32_t *C_bits)
{
    static float Af[144];
    static float Cf[32];
    for (int k = 0; k < K; ++k)
        Af[k] = bits_to_float(A_bits[k]);
    accel_load_A(Af, 1, (uint32_t)K);
    accel_start_wait();
    accel_read_C(Cf, 1, (uint32_t)N);
    for (int n = 0; n < N; ++n)
        C_bits[n] = float_to_bits(Cf[n]);
}

// Convolution layer using IP (channel-major activations) 3x3 pad=1
static void conv_layer_ip(const uint32_t *in_bits, int Cin, int Cout, int H, int W,
                          const uint32_t *W_bits, const uint32_t *bias_bits, uint32_t *out_bits)
{
    const int K = Cin * 9;
    xil_printf("[conv] Cin=%d Cout=%d K=%d H=%d W=%d\n", Cin, Cout, K, H, W);
    accel_set_dims(1, (uint32_t)K, (uint32_t)Cout);
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
    accel_set_dims(1, C3, NUM_CLASSES);
    static float Bf[32 * 10];
    // Arrange B as row-major [k][n] for the driver/IP: index k*NUM_CLASSES + n
    for (int k = 0; k < C3; ++k)
    {
        for (int n = 0; n < NUM_CLASSES; ++n)
        {
            Bf[k * NUM_CLASSES + n] = bits_to_float(W_bits[n * C3 + k]);
        }
    }
    accel_load_B(Bf, C3, NUM_CLASSES);
    // A
    accel_load_A(gap, 1, C3);
    accel_start_wait();
    static float Cf[10];
    accel_read_C(Cf, 1, NUM_CLASSES);
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
    xil_printf("[cls] logits[0]=%f logits[best=%d]=%f\n", Cf[0] + bits_to_float(bias_bits[0]), best, bestv);
    return best;
}

static int run_inference_ip(const uint32_t *img)
{
    static uint32_t act1[IMG_SIZE * IMG_SIZE * C1];
    static uint32_t act2[IMG_SIZE * IMG_SIZE * C2];
    static uint32_t act3[IMG_SIZE * IMG_SIZE * C3];
    xil_printf("[infer] starting\n");
    conv_layer_ip(img, 1, C1, IMG_SIZE, IMG_SIZE, WEIGHTS_CONV1_BITS, BIAS_CONV1_BITS, act1);
    conv_layer_ip(act1, C1, C2, IMG_SIZE, IMG_SIZE, WEIGHTS_CONV2_BITS, BIAS_CONV2_BITS, act2);
    conv_layer_ip(act2, C2, C3, IMG_SIZE, IMG_SIZE, WEIGHTS_CONV3_BITS, BIAS_CONV3_BITS, act3);
    int pred = classifier_ip(act3, IMG_SIZE, IMG_SIZE, WEIGHTS_CLS_BITS, BIAS_CLS_BITS);
    xil_printf("[infer] done pred=%d\n", pred);
    return pred;
}

// --- UART helpers and image receive ---
static inline void uart_send_byte(uint8_t b)
{
    // Wait until TX FIFO has space
    while ((Xil_In32(UART_STATUS) & UART_STATUS_TX_FULL) != 0u)
        ;
    Xil_Out32(UART_TX_FIFO, (uint32_t)b);
}

// Receive framed image: [START][CMD][LEN_L][LEN_H][payload 400 bytes][STOP]
// Returns 0 on success, <0 on error
// Helper: read next byte blocking
static inline uint8_t uart_getc_block(void)
{
    while ((Xil_In32(UART_STATUS) & UART_STATUS_RX_VALID) == 0u)
        ;
    return (uint8_t)(Xil_In32(UART_RX_FIFO) & 0xFFu);
}

// Simplified receive protocol:
// Frame: [START=0xA5] [100 words little-endian (400 bytes)] [STOP=0x5A]
// No command, no length, no ASCII hex parsing.
static int recv_image(uint32_t *out_words)
{
    // Wait for start byte (ignore anything else)
    uint8_t b;
    do
    {
        b = uart_getc_block();
    } while (b != PKT_START);
    // Read 100 words
    for (int i = 0; i < 100; ++i)
    {
        uint32_t w0 = (uint32_t)uart_getc_block();
        uint32_t w1 = (uint32_t)uart_getc_block();
        uint32_t w2 = (uint32_t)uart_getc_block();
        uint32_t w3 = (uint32_t)uart_getc_block();
        out_words[i] = (w0) | (w1 << 8) | (w2 << 16) | (w3 << 24);
    }
    uint8_t stop = uart_getc_block();
    if (stop != PKT_STOP)
    {
        xil_printf("[uart] bad STOP 0x%02x\n", stop);
        return -1;
    }
    return 0;
}

int do_init(void)
{

    uint32_t status; // status from Xilinx Lib calls

    // initial the RGB PWM clock to Nexys4IO
    // status = N4IO_RGB_timer_initialize();
    // if (status != XST_SUCCESS)
    // 	return XST_FAILURE;

    // initialize the Nexys4 driver
    status = NX4IO_initialize(XPAR_NEXYS4IO_0_BASEADDR);
    if (status != XST_SUCCESS)
    {
        return XST_FAILURE;
    }
    NX410_SSEG_setAllDigits(SSEGLO, CC_BLANK, CC_BLANK, CC_BLANK, CC_BLANK, 0);
    NX410_SSEG_setAllDigits(SSEGHI, CC_BLANK, CC_BLANK, CC_BLANK, CC_BLANK, 0);

    // Init the RGB2 Channel
    // NX4IO_RGBLED_setChnlEn(RGB2, true, true, true);

    return XST_SUCCESS;
}

int main(void)
{
    init_platform();
    xil_printf("=== MB Static Inference Test (MAC IP) ===\n");
    xil_printf("Clock freq (reported)=%lu Hz\n", (unsigned long)XPAR_CPU_CORE_CLOCK_FREQ_HZ);
    // Loop forever: wait for packet, run inference, report, then go back to waiting

    uint32_t sts = do_init();
    if (XST_SUCCESS != sts)
    {
        xil_printf("FATAL(main): System initialization failed\r\n");
        return 1;
    }

    while (1)
    {
        static uint32_t rx_img[100];
        xil_printf("\n[uart] Ready. Send frame: 0xA5 + 400 data bytes + 0x5A.\n");
        int rc = recv_image(rx_img);
        if (rc == 0)
        {
            xil_printf("[uart] image received (100 words)\n");
            xil_printf("[image] pix0=0x%08lx pix10=0x%08lx\n", (unsigned long)rx_img[0], (unsigned long)rx_img[10]);
            int pred = run_inference_ip(rx_img);
            xil_printf("PREDICTED CLASS: %d\n", pred);
            uart_send_byte((uint8_t)pred);
            NX4IO_SSEG_setDigit(SSEGLO, DIGIT0, pred);
        }
        else
        {
            xil_printf("[uart] receive failed rc=%d, Try again!\n", rc);
        }
        // Drain any leftover input bytes to avoid stale data affecting next packet
        for (;;)
        {
            uint32_t st = Xil_In32(UART_STATUS);
            if ((st & UART_STATUS_RX_VALID) == 0u)
                break;
            (void)Xil_In32(UART_RX_FIFO); // discard
        }
    }
    // not reached
    // cleanup_platform();
    // return 0;
}

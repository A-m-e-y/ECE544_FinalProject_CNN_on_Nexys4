
// Network constants (must match training / weights)
#define IMG_SIZE 10
#define C1 8
#define C2 16
#define C3 32
#define NUM_CLASSES 10

#define MATRIXMUL_BASEADDR XPAR_AXI_MATRIXMULENGINE_0_BASEADDR

static void preload_B_layer(const uint32_t *W_bits, int K, int N);
static void matmul_row_preloadedB(const uint32_t *A_bits, int K, int N, uint32_t *C_bits);
static void conv_layer_ip(const uint32_t *in_bits, int Cin, int Cout, int H, int W,
                          const uint32_t *W_bits, const uint32_t *bias_bits, uint32_t *out_bits);
static int classifier_ip(const uint32_t *act3_bits, int H, int W,
                         const uint32_t *W_bits, const uint32_t *bias_bits);
int run_inference_ip(const uint32_t *img);

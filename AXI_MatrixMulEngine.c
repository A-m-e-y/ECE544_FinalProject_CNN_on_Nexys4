
/****************** Include Files ********************/
#include "xil_io.h"
#include <string.h>
#include <xil_printf.h>
#include "sleep.h"
#include "xtmrctr.h"

#include "AXI_MatrixMulEngine.h"
// Helper function to convert float to 32-bit representation
static inline uint32_t float_to_uint32(float f) {
    union {
        float f;
        uint32_t u;
    } converter;
    converter.f = f;
    return converter.u;
}

// Helper function to convert 32-bit to float representation
static inline float uint32_to_float(uint32_t u) {
    union {
        float f;
        uint32_t u;
    } converter;
    converter.u = u;
    return converter.f;
}

/**
 * Set matrix dimensions
 */
void matrixmul_set_dimensions(uint32_t base_addr, uint32_t m, uint32_t k, uint32_t n) {
    // xil_printf("Setting Matrix Dimensions\n");
    AXI_MATRIXMULENGINE_mWriteReg(base_addr, MATRIXMUL_M_DIM_REG_OFFSET, m);
    AXI_MATRIXMULENGINE_mWriteReg(base_addr, MATRIXMUL_K_DIM_REG_OFFSET, k);
    AXI_MATRIXMULENGINE_mWriteReg(base_addr, MATRIXMUL_N_DIM_REG_OFFSET, n);
    // xil_printf("Done Matrix Dimensions\n");
}

/**
 * Write Matrix A to hardware (row-major order)
 */
void matrixmul_write_matrix_a(uint32_t base_addr, const float *matrix, uint32_t m, uint32_t k) {
    // xil_printf("Setting Matrix A\n");
    uint32_t control = MATRIXMUL_CTRL_MEMSEL_A;
    AXI_MATRIXMULENGINE_mWriteReg(base_addr, MATRIXMUL_CONTROL_REG_OFFSET, control);
    
    for (uint32_t i = 0; i < m * k; i++) {
        AXI_MATRIXMULENGINE_mWriteReg(base_addr, MATRIXMUL_MEM_ADDR_REG_OFFSET, i);
        AXI_MATRIXMULENGINE_mWriteReg(base_addr, MATRIXMUL_MEM_WDATA_REG_OFFSET, float_to_uint32(matrix[i]));
    }
    // xil_printf("Done Matrix A\n");
}

/**
 * Write Matrix B to hardware (row-major order)
 */
void matrixmul_write_matrix_b(uint32_t base_addr, const float *matrix, uint32_t k, uint32_t n) {
    // xil_printf("Setting Matrix B\n");
    uint32_t control = MATRIXMUL_CTRL_MEMSEL_B;
    AXI_MATRIXMULENGINE_mWriteReg(base_addr, MATRIXMUL_CONTROL_REG_OFFSET, control);
    
    for (uint32_t i = 0; i < k * n; i++) {
        AXI_MATRIXMULENGINE_mWriteReg(base_addr, MATRIXMUL_MEM_ADDR_REG_OFFSET, i);
        AXI_MATRIXMULENGINE_mWriteReg(base_addr, MATRIXMUL_MEM_WDATA_REG_OFFSET, float_to_uint32(matrix[i]));
    }
    // xil_printf("Done Matrix B\n");
}

/**
 * Start computation
 */
void matrixmul_start(uint32_t base_addr) {
    // xil_printf("Starting Matrix MUL\n");
    uint32_t control = MATRIXMUL_CTRL_START_MASK;
    AXI_MATRIXMULENGINE_mWriteReg(base_addr, MATRIXMUL_CONTROL_REG_OFFSET, control);
}

/**
 * Check if computation is done
 */
bool matrixmul_is_done(uint32_t base_addr) {
    uint32_t status = AXI_MATRIXMULENGINE_mReadReg(base_addr, MATRIXMUL_STATUS_REG_OFFSET);
    // xil_printf("Read status = %0d\n", status);
    return (status & MATRIXMUL_STATUS_DONE_MASK) != 0;
}

/**
 * Wait for computation to complete
 */
void matrixmul_wait_done(uint32_t base_addr) {
    while (!matrixmul_is_done(base_addr)) {
        // Polling loop
        // usleep(1000*1000);
    }
}

/**
 * Read result Matrix C from hardware (row-major order)
 */
void matrixmul_read_matrix_c(uint32_t base_addr, float *matrix, uint32_t m, uint32_t n) {
    uint32_t control = MATRIXMUL_CTRL_MEMSEL_C;
    AXI_MATRIXMULENGINE_mWriteReg(base_addr, MATRIXMUL_CONTROL_REG_OFFSET, control);
    
    for (uint32_t i = 0; i < m * n; i++) {
        AXI_MATRIXMULENGINE_mWriteReg(base_addr, MATRIXMUL_MEM_ADDR_REG_OFFSET, i);
        uint32_t data = AXI_MATRIXMULENGINE_mReadReg(base_addr, MATRIXMUL_MEM_RDATA_REG_OFFSET);
        matrix[i] = uint32_to_float(data);
    }
}

/**
 * Complete matrix multiplication: C = A * B
 * Returns 0 on success, -1 on error
 */
int matrixmul_compute(uint32_t base_addr, const float *A, const float *B, float *C,
                      uint32_t m, uint32_t k, uint32_t n) {
    // Validate dimensions
    if (m > MATRIXMUL_MAX_M || k > MATRIXMUL_MAX_K || n > MATRIXMUL_MAX_N) {
        return -1;
    }
    
    // Set dimensions
    matrixmul_set_dimensions(base_addr, m, k, n);
    
    // Load input matrices
    matrixmul_write_matrix_a(base_addr, A, m, k);
    matrixmul_write_matrix_b(base_addr, B, k, n);
    
    // Start computation
    matrixmul_start(base_addr);
    
    // Wait for completion
    matrixmul_wait_done(base_addr);
    
    // Read result
    matrixmul_read_matrix_c(base_addr, C, m, n);
    
    return 0;
}

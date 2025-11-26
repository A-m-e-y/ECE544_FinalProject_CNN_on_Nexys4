// main.c - Example application for MatrixMul AXI IP with MicroBlaze
// Uses the matrixmul_hw.h/.c API to load A/B, start, poll, and read C.
// Prints both hardware result and a software reference for verification.

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "platform.h"

#include "xparameters.h"     // Provides base address macros from Vivado
#include "AXI_MatrixMulEngine.h"    // Your hardware driver API

// Map to your instance name if different; check xparameters.h after Address Editor
#define MATRIXMUL_BASEADDR XPAR_AXI_MATRIXMULENGINE_0_BASEADDR


// Simple software reference: C = A (MxK) * B (KxN) in row-major
static void sw_matmul(const float* A, const float* B, float* C,
                      uint32_t M, uint32_t K, uint32_t N) {
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                acc += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = acc;
        }
    }
}

static void print_matrix(const char* name, const float* Mx, uint32_t M, uint32_t N) {
    printf("%s (%ux%u):\n", name, (unsigned)M, (unsigned)N);
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            printf("%8.3f ", Mx[i*N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void) {

    init_platform();


    printf("MatrixMul Hardware Accelerator Demo\n");
    printf("===================================\n\n");

    // Choose dimensions within your RTL limits (MAX_M/K/N = 16 by default)
    const uint32_t M = 4, K = 4, N = 4;

    // Example matrices (row-major). Feel free to edit these.
    // A: a simple 4x4
    float A[16] = {
        1.123f, 2.456f, 3.789f, 4.901f,
        5.123f, 6.456f, 7.789f, 8.901f,
        9.123f, 10.456f, 11.789f, 12.901f,
        13.123f, 14.456f, 15.789f, 16.901f
    };

    // B: identity, so C should equal A
    float B[16] = {
        0.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f
    };

    float C_hw[16] = {0};
    float C_sw[16] = {0};

    // Print inputs
    print_matrix("A", A, M, K);
    print_matrix("B", B, K, N);

    // Kick the hardware: set dims, write A/B, start, wait, read C
    printf("Programming accelerator @ 0x%08lx\n\n", (unsigned long)MATRIXMUL_BASEADDR);

    // Set dimensions
    matrixmul_set_dimensions(MATRIXMUL_BASEADDR, M, K, N);

    // Load input matrices
    matrixmul_write_matrix_a(MATRIXMUL_BASEADDR, A, M, K);
    matrixmul_write_matrix_b(MATRIXMUL_BASEADDR, B, K, N);

    // Start and wait for completion
    matrixmul_start(MATRIXMUL_BASEADDR);
    matrixmul_wait_done(MATRIXMUL_BASEADDR);

    // Read back hardware result
    matrixmul_read_matrix_c(MATRIXMUL_BASEADDR, C_hw, M, N);

    // Software reference for verification
    sw_matmul(A, B, C_sw, M, K, N);

    // Show results
    print_matrix("C (hardware)", C_hw, M, N);
    print_matrix("C (software ref)", C_sw, M, N);

    // Compare with a small tolerance (floating point)
    const float eps = 1e-3f;
    int errors = 0;
    for (uint32_t i = 0; i < M*N; ++i) {
        float diff = fabsf(C_hw[i] - C_sw[i]);
        if (diff > eps || isnan(C_hw[i])) {
            printf("Mismatch at idx %u: HW=%.6f SW=%.6f diff=%.6f\n",
                   (unsigned)i, C_hw[i], C_sw[i], diff);
            errors++;
        }
    }

    if (errors == 0) {
        printf("RESULT: PASS (hardware matches software within %.1e)\n", eps);
    } else {
        printf("RESULT: FAIL (%d mismatches)\n", errors);
    }

    cleanup_platform();
    return (errors == 0) ? 0 : 1;
}

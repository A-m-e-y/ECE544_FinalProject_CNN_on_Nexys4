
#ifndef AXI_MATRIXMULENGINE_H
#define AXI_MATRIXMULENGINE_H


/****************** Include Files ********************/
#include "xil_types.h"
#include "xstatus.h"
#include <stdint.h>
#include <stdbool.h>
#include "xil_io.h"

#define AXI_MATRIXMULENGINE_S00_AXI_SLV_REG0_OFFSET 0
#define AXI_MATRIXMULENGINE_S00_AXI_SLV_REG1_OFFSET 4
#define AXI_MATRIXMULENGINE_S00_AXI_SLV_REG2_OFFSET 8
#define AXI_MATRIXMULENGINE_S00_AXI_SLV_REG3_OFFSET 12
#define AXI_MATRIXMULENGINE_S00_AXI_SLV_REG4_OFFSET 16
#define AXI_MATRIXMULENGINE_S00_AXI_SLV_REG5_OFFSET 20
#define AXI_MATRIXMULENGINE_S00_AXI_SLV_REG6_OFFSET 24
#define AXI_MATRIXMULENGINE_S00_AXI_SLV_REG7_OFFSET 28


// Register offsets
#define MATRIXMUL_CONTROL_REG_OFFSET    0x00
#define MATRIXMUL_STATUS_REG_OFFSET     0x04
#define MATRIXMUL_M_DIM_REG_OFFSET      0x08
#define MATRIXMUL_K_DIM_REG_OFFSET      0x0C
#define MATRIXMUL_N_DIM_REG_OFFSET      0x10
#define MATRIXMUL_MEM_ADDR_REG_OFFSET   0x14
#define MATRIXMUL_MEM_WDATA_REG_OFFSET  0x18
#define MATRIXMUL_MEM_RDATA_REG_OFFSET  0x1C

// Control register bits
#define MATRIXMUL_CTRL_START_MASK       0x00000001
#define MATRIXMUL_CTRL_RESET_MASK       0x00000002
#define MATRIXMUL_CTRL_MEMSEL_MASK      0x0000000C
#define MATRIXMUL_CTRL_MEMSEL_A         0x00000000
#define MATRIXMUL_CTRL_MEMSEL_B         0x00000004
#define MATRIXMUL_CTRL_MEMSEL_C         0x00000008

// Status register bits
#define MATRIXMUL_STATUS_DONE_MASK      0x00000002
#define MATRIXMUL_STATUS_BUSY_MASK      0x00000004
// #define MATRIXMUL_STATUS_ERROR_MASK     0x00000004

// Maximum dimensions (change to match RTL parameters)
#define MATRIXMUL_MAX_M                 16
#define MATRIXMUL_MAX_K                 16
#define MATRIXMUL_MAX_N                 16

/**************************** Type Definitions *****************************/
/**
 *
 * Write a value to a AXI_MATRIXMULENGINE register. A 32 bit write is performed.
 * If the component is implemented in a smaller width, only the least
 * significant data is written.
 *
 * @param   BaseAddress is the base address of the AXI_MATRIXMULENGINEdevice.
 * @param   RegOffset is the register offset from the base to write to.
 * @param   Data is the data written to the register.
 *
 * @return  None.
 *
 * @note
 * C-style signature:
 * 	void AXI_MATRIXMULENGINE_mWriteReg(u32 BaseAddress, unsigned RegOffset, u32 Data)
 *
 */
#define AXI_MATRIXMULENGINE_mWriteReg(BaseAddress, RegOffset, Data) \
  	Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))

/**
 *
 * Read a value from a AXI_MATRIXMULENGINE register. A 32 bit read is performed.
 * If the component is implemented in a smaller width, only the least
 * significant data is read from the register. The most significant data
 * will be read as 0.
 *
 * @param   BaseAddress is the base address of the AXI_MATRIXMULENGINE device.
 * @param   RegOffset is the register offset from the base to write to.
 *
 * @return  Data is the data from the register.
 *
 * @note
 * C-style signature:
 * 	u32 AXI_MATRIXMULENGINE_mReadReg(u32 BaseAddress, unsigned RegOffset)
 *
 */
#define AXI_MATRIXMULENGINE_mReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))

/************************** Function Prototypes ****************************/
/**
 *
 * Run a self-test on the driver/device. Note this may be a destructive test if
 * resets of the device are performed.
 *
 * If the hardware system is not built correctly, this function may never
 * return to the caller.
 *
 * @param   baseaddr_p is the base address of the AXI_MATRIXMULENGINE instance to be worked on.
 *
 * @return
 *
 *    - XST_SUCCESS   if all self-test code passed
 *    - XST_FAILURE   if any self-test code failed
 *
 * @note    Caching must be turned off for this function to work.
 * @note    Self test may fail if data memory and device are not on the same bus.
 *
 */
XStatus AXI_MATRIXMULENGINE_Reg_SelfTest(void * baseaddr_p);

// Function prototypes
void matrixmul_set_dimensions(uint32_t base_addr, uint32_t m, uint32_t k, uint32_t n);
void matrixmul_write_matrix_a(uint32_t base_addr, const float *matrix, uint32_t m, uint32_t k);
void matrixmul_write_matrix_b(uint32_t base_addr, const float *matrix, uint32_t k, uint32_t n);
void matrixmul_start(uint32_t base_addr);
bool matrixmul_is_done(uint32_t base_addr);
void matrixmul_wait_done(uint32_t base_addr);
void matrixmul_read_matrix_c(uint32_t base_addr, float *matrix, uint32_t m, uint32_t n);
int matrixmul_compute(uint32_t base_addr, const float *A, const float *B, float *C,
                      uint32_t m, uint32_t k, uint32_t n);

#endif // AXI_MATRIXMULENGINE_H

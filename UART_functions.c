#include "xil_io.h"
#include "UART_functions.h"

// --- UART helpers and image receive ---
void uart_send_byte(uint8_t b)
{
    // Wait until TX FIFO has space
    while ((Xil_In32(UART_STATUS) & UART_STATUS_TX_FULL) != 0u)
        ;
    Xil_Out32(UART_TX_FIFO, (uint32_t)b);
}

// Receive framed image: [START][payload 400 bytes][STOP]
// Returns 0 on success, <0 on error
// Helper: read next byte blocking
uint8_t uart_getc_block(void)
{
    while ((Xil_In32(UART_STATUS) & UART_STATUS_RX_VALID) == 0u)
        ;
    return (uint8_t)(Xil_In32(UART_RX_FIFO) & 0xFFu);
}

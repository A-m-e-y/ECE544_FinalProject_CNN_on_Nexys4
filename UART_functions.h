
// UARTLite MMIO registers (from xparameters.h canonical base)
#define UARTLITE_BASE XPAR_XUARTLITE_0_BASEADDR
#define UART_RX_FIFO (UARTLITE_BASE + 0x00)
#define UART_TX_FIFO (UARTLITE_BASE + 0x04)
#define UART_STATUS (UARTLITE_BASE + 0x08)
#define UART_CONTROL (UARTLITE_BASE + 0x0C)

#define UART_STATUS_RX_VALID (1u << 0)
#define UART_STATUS_TX_FULL (1u << 3)

void uart_send_byte(uint8_t b);
uint8_t uart_getc_block(void);

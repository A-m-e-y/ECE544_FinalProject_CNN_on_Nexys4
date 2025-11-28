#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include "platform.h"
#include "xparameters.h"
#include "AXI_MatrixMulEngine.h"
#include "sleep.h"
#include "xtmrctr.h"
#include "nexys4io.h"
#include "CNN_Core.h"
#include "UART_functions.h"


// UART protocol bytes (match uart_send_image.py)
#define PKT_START 0xA5
#define PKT_CMD_IMAGE 0x01
#define PKT_STOP 0x5A

// Simplified receive protocol:
// Frame: [START=0xA5] [100 words little-endian (400 bytes)] [STOP=0x5A]
// No command, no length, no ASCII hex parsing.
static int recv_image(uint32_t *out_words)
{
    // Wait for start byte (ignore anything else)
    uint8_t start = uart_getc_block();
    if (start != PKT_START)
    {
        xil_printf("[uart] bad SART 0x%02x\n", start);
        return -1;
    }
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
            continue;
        }
    }
    // Shouldn't get here!
    cleanup_platform();
    return 0;
}

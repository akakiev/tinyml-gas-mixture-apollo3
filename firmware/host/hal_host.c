// HAL для ПК: той самий main.cc працює як консольна програма (stdin/stdout замість UART).
// Потрібен для перевірки прошивки і hil_host.py без плати:  python hil_host.py --exec ./gas_host
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <time.h>
#include "hal.h"

void hal_init(void) {}

void hal_write(const char *s, uint32_t n)
{
    fwrite(s, 1, n, stdout);
    fflush(stdout);
}

int hal_read(uint8_t *buf, uint32_t n)
{
    return fread(buf, 1, n, stdin) == n ? 0 : -1;
}

uint32_t hal_cycles(void)          // наносекунди замість тактів
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec);
}

uint32_t hal_cpu_hz(void) { return 1000000000u; }
void hal_profile_pin(int level) { (void)level; }

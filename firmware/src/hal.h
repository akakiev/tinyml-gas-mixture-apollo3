// Тонкий апаратний шар: усе, що залежить від плати, винесено сюди.
// Реалізації: hal_apollo3.c (SparkFun Edge, AmbiqSuite) і host/hal_host.c (ПК, для тестів).
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void     hal_init(void);                         // тактування, кеш, UART, лічильник тактів
void     hal_write(const char *s, uint32_t n);   // блокуючий запис у UART
int      hal_read(uint8_t *buf, uint32_t n);     // блокуюче читання рівно n байтів; 0 = успіх
uint32_t hal_cycles(void);                       // поточне значення лічильника тактів (DWT CYCCNT)
uint32_t hal_cpu_hz(void);                       // частота ядра, Гц
void     hal_profile_pin(int level);             // GPIO-мітка для профайлера живлення (якщо задано)

#ifdef __cplusplus
}
#endif

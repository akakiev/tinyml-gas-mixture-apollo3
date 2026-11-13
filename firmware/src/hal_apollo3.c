// HAL для Ambiq Apollo3 Blue (плата SparkFun Edge), AmbiqSuite SDK 2.x (форк SparkFun).
//
// Опції компіляції:
//   -DHIL_BAUD=115200     швидкість UART (через USB-UART плати)
//   -DUSE_BURST_MODE      увімкнути TurboSPOT 96 МГц (рядок «int8, 96 МГц» у табл. 3)
//   -DPROFILE_PIN=<GPIO>  номер вільного GPIO, який тримається в «1» під час бенчмарку
//                         (тригер для PPK2/Joulescope). За замовчуванням вимкнено:
//                         не використовуйте світлодіод — його струм спотворить вимірювання.
//
// Енергетичні режими (для табл. «Measurements on the SparkFun Edge»):
//   hal_sleep_forever()  — UART вимкнено, піни UART відключено, ядро в deep sleep назавжди
//                          (струм I_sleep; вихід — кнопка RST або перевідкриття порту, DTR).
//   hal_duty_*()         — режим «одне виведення за період»: між виведеннями deep sleep,
//                          пробудження від STIMER (LFRC ~1024 Гц, точність ±кілька %, тому
//                          реальний період дивіться на профайлері). SRAM зберігається повністю.
#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "hal.h"

#ifndef HIL_BAUD
#define HIL_BAUD 115200
#endif
#ifndef PROFILE_PIN
#define PROFILE_PIN (-1)
#endif

static void *s_uart;
static uint32_t s_cpu_hz = 48000000u;

void hal_init(void)
{
    am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0);   // 48 МГц
    am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
    am_hal_cachectrl_enable();
    am_bsp_low_power_init();

#ifdef USE_BURST_MODE
    am_hal_burst_avail_e avail;
    am_hal_burst_mode_e mode;
    if (am_hal_burst_mode_initialize(&avail) == AM_HAL_STATUS_SUCCESS &&
        avail == AM_HAL_BURST_AVAIL &&
        am_hal_burst_mode_enable(&mode) == AM_HAL_STATUS_SUCCESS &&
        mode == AM_HAL_BURST_MODE)
    {
        s_cpu_hz = 96000000u;
    }
#endif

    // UART без програмних черг: передача й прийом напряму через FIFO в блокуючому режимі,
    // переривання не потрібні.
    am_hal_uart_config_t cfg = {0};
    cfg.ui32BaudRate    = HIL_BAUD;
    cfg.ui32DataBits    = AM_HAL_UART_DATA_BITS_8;
    cfg.ui32Parity      = AM_HAL_UART_PARITY_NONE;
    cfg.ui32StopBits    = AM_HAL_UART_ONE_STOP_BIT;
    cfg.ui32FlowControl = AM_HAL_UART_FLOW_CTRL_NONE;
    cfg.ui32FifoLevels  = (AM_HAL_UART_TX_FIFO_1_2 | AM_HAL_UART_RX_FIFO_1_2);
    am_hal_uart_initialize(0, &s_uart);
    am_hal_uart_power_control(s_uart, AM_HAL_SYSCTRL_WAKE, false);
    am_hal_uart_configure(s_uart, &cfg);
    am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_TX, g_AM_BSP_GPIO_COM_UART_TX);
    am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_RX, g_AM_BSP_GPIO_COM_UART_RX);

#if PROFILE_PIN >= 0
    am_hal_gpio_pinconfig(PROFILE_PIN, g_AM_HAL_GPIO_OUTPUT);
    am_hal_gpio_output_clear(PROFILE_PIN);
#endif

    // Лічильник тактів ядра Cortex-M4
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

void hal_write(const char *s, uint32_t n)
{
    uint32_t done = 0;
    am_hal_uart_transfer_t t = {
        .ui32Direction = AM_HAL_UART_WRITE,
        .pui8Data = (uint8_t *)s,
        .ui32NumBytes = n,
        .ui32TimeoutMs = AM_HAL_UART_WAIT_FOREVER,
        .pui32BytesTransferred = &done,
    };
    am_hal_uart_transfer(s_uart, &t);
}

int hal_read(uint8_t *buf, uint32_t n)
{
    uint32_t done = 0;
    am_hal_uart_transfer_t t = {
        .ui32Direction = AM_HAL_UART_READ,
        .pui8Data = buf,
        .ui32NumBytes = n,
        .ui32TimeoutMs = AM_HAL_UART_WAIT_FOREVER,
        .pui32BytesTransferred = &done,
    };
    uint32_t st = am_hal_uart_transfer(s_uart, &t);
    return (st == AM_HAL_STATUS_SUCCESS && done == n) ? 0 : -1;
}

uint32_t hal_cycles(void) { return DWT->CYCCNT; }
uint32_t hal_cpu_hz(void) { return s_cpu_hz; }

void hal_profile_pin(int level)
{
#if PROFILE_PIN >= 0
    if (level) am_hal_gpio_output_set(PROFILE_PIN);
    else       am_hal_gpio_output_clear(PROFILE_PIN);
#else
    (void)level;
#endif
}

// ---------------------------------------------------------------- енергетичні режими
static volatile uint32_t s_tick_pending;
static uint32_t s_period_ticks = 1024u;
static uint32_t s_primask_saved;

// Обробник переривання STIMER COMPARE A (ім'я — з startup_gcc.c SDK; С-лінкування)
void am_stimer_cmpr0_isr(void)
{
    am_hal_stimer_int_clear(AM_HAL_STIMER_INT_COMPAREA);
    am_hal_stimer_compare_delta_set(0, s_period_ticks);
    s_tick_pending = 1;
}

static void uart_off(int tx_idle_high)
{
    am_hal_uart_tx_flush(s_uart);                                   // дочекатися відправлення
    am_hal_uart_power_control(s_uart, AM_HAL_SYSCTRL_DEEPSLEEP, true);
    if (tx_idle_high)
    {
        // TX тримаємо у «1» (стан спокою UART), щоб ПК не бачив сміття, поки UART вимкнено
        am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_TX, g_AM_HAL_GPIO_OUTPUT);
        am_hal_gpio_output_set(AM_BSP_GPIO_COM_UART_TX);
    }
    else
    {
        am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_TX, g_AM_HAL_GPIO_DISABLE);
    }
    am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_RX, g_AM_HAL_GPIO_DISABLE);
}

static void uart_on(void)
{
    am_hal_uart_power_control(s_uart, AM_HAL_SYSCTRL_WAKE, true);
    am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_TX, g_AM_BSP_GPIO_COM_UART_TX);
    am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_RX, g_AM_BSP_GPIO_COM_UART_RX);
}

void hal_sleep_forever(void)
{
    uart_off(0);
#if PROFILE_PIN >= 0
    am_hal_gpio_output_clear(PROFILE_PIN);
#endif
#ifdef USE_BURST_MODE
    {
        am_hal_burst_mode_e mode;
        am_hal_burst_mode_disable(&mode);
    }
#endif
    CoreDebug->DEMCR &= ~CoreDebug_DEMCR_TRCENA_Msk;                // вимкнути DWT/трасування
    am_hal_interrupt_master_disable();
    while (1)
    {
        am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
    }
}

void hal_duty_begin(uint32_t period_ms)
{
    uint32_t t = (period_ms * 1024u + 500u) / 1000u;
    s_period_ticks = t < 2u ? 2u : t;
    uart_off(1);
    s_tick_pending = 0;
    am_hal_stimer_int_clear(AM_HAL_STIMER_INT_COMPAREA);
    am_hal_stimer_config(AM_HAL_STIMER_CFG_CLEAR | AM_HAL_STIMER_CFG_FREEZE);
    am_hal_stimer_config(AM_HAL_STIMER_LFRC_1KHZ | AM_HAL_STIMER_CFG_COMPARE_A_ENABLE);
    am_hal_stimer_compare_delta_set(0, s_period_ticks);
    am_hal_stimer_int_enable(AM_HAL_STIMER_INT_COMPAREA);
    NVIC_EnableIRQ(STIMER_CMPR0_IRQn);
    s_primask_saved = am_hal_interrupt_master_enable();          // повернемо в hal_duty_end()
}

void hal_duty_wait(void)
{
    // Перевірка прапорця й засинання — під PRIMASK=1, щоб не пропустити тік між ними.
    // WFI прокидається від відкладеного переривання і при PRIMASK=1; обробник виконується
    // одразу після відновлення PRIMASK.
    for (;;)
    {
        uint32_t st = am_hal_interrupt_master_disable();
        if (s_tick_pending)
        {
            s_tick_pending = 0;
            am_hal_interrupt_master_set(st);
            return;
        }
        am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
        am_hal_interrupt_master_set(st);
    }
}

void hal_duty_end(void)
{
    NVIC_DisableIRQ(STIMER_CMPR0_IRQn);
    am_hal_stimer_int_disable(AM_HAL_STIMER_INT_COMPAREA);
    am_hal_stimer_int_clear(AM_HAL_STIMER_INT_COMPAREA);
    am_hal_stimer_config(AM_HAL_STIMER_CFG_CLEAR | AM_HAL_STIMER_CFG_FREEZE);
    am_hal_interrupt_master_set(s_primask_saved);
    uart_on();
}

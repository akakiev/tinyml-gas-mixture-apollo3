@echo off
rem Flash one firmware image to the SparkFun Edge via the Ambiq Secure Bootloader (ASB).
rem Usage:  flash.cmd COM8 builds\B_int8_48MHz.bin [baud]
rem         baud is optional: 115200 (default) or 921600 if 115200 fails.
rem Hold button 14, press and release RST, start this script, release 14 when "Sending Data Packet" appears.
setlocal
if "%~2"=="" (echo Usage: flash.cmd COMx builds\NAME.bin [baud] & exit /b 1)
set "BAUD=%~3"
if "%BAUD%"=="" set "BAUD=115200"
set "ASB=%~dp0tools\asb\asb.py"
if not exist "%ASB%" (echo tools\asb\asb.py not found - run build_all.sh in WSL first & exit /b 1)
python "%ASB%" --bin "%~2" --load-address-blob 0x20000 --magic-num 0xCB -o "%TEMP%\gas_tinyml" --version 0x0 --load-address-wired 0xC000 -i 6 --options 0x1 -b %BAUD% -port %1 -r 2 -v

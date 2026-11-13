@echo off
rem Flash one firmware image to the SparkFun Edge via the Ambiq Secure Bootloader (ASB).
rem Usage:  flash.cmd COM8 builds\B_int8_48MHz.bin
rem Hold button 14, press and release RST, start this script, release 14 when "Sending Data Packet" appears.
setlocal
if "%~2"=="" (echo Usage: flash.cmd COMx builds\NAME.bin & exit /b 1)
set "ASB=%~dp0tools\asb\asb.py"
if not exist "%ASB%" (echo tools\asb\asb.py not found - run build_all.sh in WSL first & exit /b 1)
python "%ASB%" --bin "%~2" --load-address-blob 0x20000 --magic-num 0xCB -o "%TEMP%\gas_tinyml" --version 0x0 --load-address-wired 0xC000 -i 6 --options 0x1 -b 115200 -port %1 -r 2 -v

@echo off
REM Apourvoir Optimized Launcher
REM Main launch script for Apourvoir with automatic hardware optimization
REM
REM Usage:
REM   launch.bat [options]
REM
REM Options:
REM   --model MODEL           Model to load (default: mistralai/Mixtral-8x7B-v0.1)
REM   --preset PRESET         Optimization preset (balanced|speed|quality|memory|latency)
REM   --no-api               Disable API server
REM   --no-ui                Disable web UI
REM   --api-host HOST        API server host (default: 0.0.0.0)
REM   --api-port PORT        API server port (default: 8080)
REM   --ui-port PORT         Web UI port (default: 8000)
REM   --config FILE          Load configuration from file
REM   --save-config FILE     Save configuration to file
REM   --log-level LEVEL      Log level (DEBUG|INFO|WARNING|ERROR)
REM   --info                 Show hardware info and exit
REM   --validate             Validate configuration and exit
REM   --help                Show this help message

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%~dp0..\..

REM Add project directory to PYTHONPATH
set PYTHONPATH=%PROJECT_DIR%;%PYTHONPATH%

REM Default values
set MODEL=mistralai/Mixtral-8x7B-v0.1
set PRESET=balanced
set ENABLE_API=true
set ENABLE_UI=true
set API_HOST=0.0.0.0
set API_PORT=8080
set UI_PORT=8000
set CONFIG_FILE=
set SAVE_CONFIG=
set LOG_LEVEL=INFO
set INFO_MODE=false
set VALIDATE_MODE=false

REM Parse arguments
:arg_loop
if "%~1"=="" goto end_args

if "%~1"=="--model" (
    set MODEL=%~2
    shift
    shift
    goto arg_loop
)

if "%~1"=="--preset" (
    set PRESET=%~2
    shift
    shift
    goto arg_loop
)

if "%~1"=="--no-api" (
    set ENABLE_API=false
    shift
    goto arg_loop
)

if "%~1"=="--no-ui" (
    set ENABLE_UI=false
    shift
    goto arg_loop
)

if "%~1"=="--api-host" (
    set API_HOST=%~2
    shift
    shift
    goto arg_loop
)

if "%~1"=="--api-port" (
    set API_PORT=%~2
    shift
    shift
    goto arg_loop
)

if "%~1"=="--ui-port" (
    set UI_PORT=%~2
    shift
    shift
    goto arg_loop
)

if "%~1"=="--config" (
    set CONFIG_FILE=%~2
    shift
    shift
    goto arg_loop
)

if "%~1"=="--save-config" (
    set SAVE_CONFIG=%~2
    shift
    shift
    goto arg_loop
)

if "%~1"=="--log-level" (
    set LOG_LEVEL=%~2
    shift
    shift
    goto arg_loop
)

if "%~1"=="--info" (
    set INFO_MODE=true
    shift
    goto arg_loop
)

if "%~1"=="--validate" (
    set VALIDATE_MODE=true
    shift
    goto arg_loop
)

if "%~1"=="--help" (
    echo Usage: %~nx0 [options]
    echo.
    echo Options:
    echo   --model MODEL           Model to load (default: %MODEL%)
    echo   --preset PRESET         Optimization preset (balanced^|speed^|quality^|memory^|latency)
    echo   --no-api               Disable API server
    echo   --no-ui                Disable web UI
    echo   --api-host HOST        API server host (default: %API_HOST%)
    echo   --api-port PORT        API server port (default: %API_PORT%)
    echo   --ui-port PORT         Web UI port (default: %UI_PORT%)
    echo   --config FILE          Load configuration from file
    echo   --save-config FILE     Save configuration to file
    echo   --log-level LEVEL      Log level (DEBUG^|INFO^|WARNING^|ERROR)
    echo   --info                 Show hardware info and exit
    echo   --validate             Validate configuration and exit
    echo   --help                Show this help message
    exit /b 0
)

REM Unknown option
echo Unknown option: %~1
echo Use --help for usage information
exit /b 1

:end_args

REM Build Python command
set PYTHON_CMD=python %PROJECT_DIR%\scripts\launcher\launch.py

REM Add arguments
set PYTHON_CMD=%PYTHON_CMD% --model %MODEL%
set PYTHON_CMD=%PYTHON_CMD% --preset %PRESET%

if %ENABLE_API%==false (
    set PYTHON_CMD=%PYTHON_CMD% --no-api
)

if %ENABLE_UI%==false (
    set PYTHON_CMD=%PYTHON_CMD% --no-ui
)

set PYTHON_CMD=%PYTHON_CMD% --api-host %API_HOST%
set PYTHON_CMD=%PYTHON_CMD% --api-port %API_PORT%
set PYTHON_CMD=%PYTHON_CMD% --ui-port %UI_PORT%
set PYTHON_CMD=%PYTHON_CMD% --log-level %LOG_LEVEL%

if not "%CONFIG_FILE%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% --config %CONFIG_FILE%
)

if not "%SAVE_CONFIG%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% --save-config %SAVE_CONFIG%
)

if %INFO_MODE%==true (
    set PYTHON_CMD=%PYTHON_CMD% --info
)

if %VALIDATE_MODE%==true (
    set PYTHON_CMD=%PYTHON_CMD% --validate
)

echo Starting Apourvoir Optimized Launcher...
echo Command: %PYTHON_CMD%
echo.

REM Run the command
%PYTHON_CMD%

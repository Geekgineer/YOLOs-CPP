@echo off
REM ============================================================================
REM YOLOs-CPP Windows Build Script (Batch)
REM ============================================================================
REM Usage:
REM   build.bat          - Build with CPU support
REM   build.bat gpu      - Build with GPU support
REM ============================================================================

setlocal enabledelayedexpansion

set VERSION=1.20.1
set GPU=0

if /i "%1"=="gpu" set GPU=1
if /i "%1"=="GPU" set GPU=1

echo ============================================
echo   YOLOs-CPP Windows Build Script
echo ============================================
echo.

REM Determine ONNX Runtime package
if %GPU%==1 (
    set ORT_PACKAGE=onnxruntime-win-x64-gpu-%VERSION%
    set ORT_URL=https://github.com/microsoft/onnxruntime/releases/download/v%VERSION%/onnxruntime-win-x64-gpu-%VERSION%.zip
    echo Building with GPU support...
) else (
    set ORT_PACKAGE=onnxruntime-win-x64-%VERSION%
    set ORT_URL=https://github.com/microsoft/onnxruntime/releases/download/v%VERSION%/onnxruntime-win-x64-%VERSION%.zip
    echo Building with CPU support...
)

set ORT_DIR=%~dp0%ORT_PACKAGE%

REM Download ONNX Runtime if not present
if not exist "%ORT_DIR%" (
    echo Downloading ONNX Runtime %VERSION%...
    powershell -Command "Invoke-WebRequest -Uri '%ORT_URL%' -OutFile '%ORT_PACKAGE%.zip'"
    if errorlevel 1 (
        echo Failed to download ONNX Runtime
        echo Please download manually from: %ORT_URL%
        exit /b 1
    )
    echo Extracting...
    powershell -Command "Expand-Archive -Path '%ORT_PACKAGE%.zip' -DestinationPath '%~dp0' -Force"
    del "%ORT_PACKAGE%.zip"
    echo ONNX Runtime downloaded successfully
) else (
    echo Using existing ONNX Runtime at: %ORT_DIR%
)

REM Create build directory
if not exist "build" mkdir build
cd build

REM Configure with CMake
echo.
echo Configuring with CMake...
cmake .. -DONNXRUNTIME_DIR="%ORT_DIR%" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 (
    echo CMake configuration failed
    cd ..
    exit /b 1
)

REM Build
echo.
echo Building...
cmake --build . --config Release --parallel
if errorlevel 1 (
    echo Build failed
    cd ..
    exit /b 1
)

cd ..

echo.
echo ============================================
echo   Build Successful!
echo ============================================
echo.
echo Executables are in: build\Release\
echo.
echo Run:
echo   build\Release\image_inference.exe models\yolo11n.onnx data\dog.jpg
echo.

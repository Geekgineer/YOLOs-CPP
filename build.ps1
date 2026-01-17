# ============================================================================
# YOLOs-CPP Windows Build Script (PowerShell)
# ============================================================================
# Usage:
#   .\build.ps1                    # Build with default settings (CPU)
#   .\build.ps1 -GPU               # Build with GPU support
#   .\build.ps1 -Version 1.20.1    # Specify ONNX Runtime version
# ============================================================================

param(
    [string]$Version = "1.20.1",
    [switch]$GPU,
    [switch]$Clean,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Green { param($msg) Write-Host $msg -ForegroundColor Green }
function Write-Yellow { param($msg) Write-Host $msg -ForegroundColor Yellow }
function Write-Red { param($msg) Write-Host $msg -ForegroundColor Red }
function Write-Cyan { param($msg) Write-Host $msg -ForegroundColor Cyan }

if ($Help) {
    Write-Host @"
YOLOs-CPP Windows Build Script

Usage:
    .\build.ps1 [options]

Options:
    -Version <ver>  ONNX Runtime version (default: 1.20.1)
    -GPU            Enable GPU/CUDA support
    -Clean          Clean build directory before building
    -Help           Show this help message

Examples:
    .\build.ps1                     # CPU build
    .\build.ps1 -GPU                # GPU build
    .\build.ps1 -Clean -GPU         # Clean GPU build
"@
    exit 0
}

Write-Cyan "============================================"
Write-Cyan "  YOLOs-CPP Windows Build Script"
Write-Cyan "============================================"
Write-Host ""

# Determine ONNX Runtime package
if ($GPU) {
    $OrtPackage = "onnxruntime-win-x64-gpu-$Version"
    $OrtUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/onnxruntime-win-x64-gpu-$Version.zip"
} else {
    $OrtPackage = "onnxruntime-win-x64-$Version"
    $OrtUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/onnxruntime-win-x64-$Version.zip"
}

$OrtDir = Join-Path $PSScriptRoot $OrtPackage

# Download ONNX Runtime if not present
if (-not (Test-Path $OrtDir)) {
    Write-Yellow "Downloading ONNX Runtime $Version..."
    $ZipFile = "$OrtPackage.zip"
    
    try {
        Invoke-WebRequest -Uri $OrtUrl -OutFile $ZipFile -UseBasicParsing
        Write-Host "Extracting..."
        Expand-Archive -Path $ZipFile -DestinationPath $PSScriptRoot -Force
        Remove-Item $ZipFile
        Write-Green "ONNX Runtime downloaded successfully"
    } catch {
        Write-Red "Failed to download ONNX Runtime: $_"
        Write-Host "Please download manually from: $OrtUrl"
        exit 1
    }
} else {
    Write-Host "Using existing ONNX Runtime at: $OrtDir"
}

# Create build directory
$BuildDir = Join-Path $PSScriptRoot "build"
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Yellow "Cleaning build directory..."
    Remove-Item -Recurse -Force $BuildDir
}
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

# Configure with CMake
Write-Cyan "Configuring with CMake..."
Push-Location $BuildDir

try {
    $CmakeArgs = @(
        "..",
        "-DONNXRUNTIME_DIR=$OrtDir",
        "-DCMAKE_BUILD_TYPE=Release"
    )
    
    # Use Visual Studio generator on Windows
    if (Get-Command "cmake" -ErrorAction SilentlyContinue) {
        cmake @CmakeArgs
        if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed" }
    } else {
        Write-Red "CMake not found. Please install CMake and add it to PATH."
        exit 1
    }
    
    # Build
    Write-Cyan "Building..."
    cmake --build . --config Release --parallel
    if ($LASTEXITCODE -ne 0) { throw "Build failed" }
    
    Write-Green ""
    Write-Green "============================================"
    Write-Green "  Build Successful!"
    Write-Green "============================================"
    Write-Host ""
    Write-Host "Executables are in: $BuildDir\Release\"
    Write-Host ""
    Write-Host "Run:"
    Write-Host "  .\Release\image_inference.exe ..\models\yolo11n.onnx ..\data\dog.jpg"
    
} finally {
    Pop-Location
}

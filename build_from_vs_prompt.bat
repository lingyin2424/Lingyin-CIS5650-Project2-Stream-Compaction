@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ===========================================
echo   使用 VS 命令提示符构建优化版本
echo ===========================================
echo.

:: 检查是否在 VS 环境中
if not defined VSINSTALLDIR (
    echo 错误: 请从 "x64 Native Tools Command Prompt for VS 2022" 运行此脚本！
    echo.
    echo 如何打开:
    echo   开始菜单 -^> Visual Studio 2022 
    echo        -^> x64 Native Tools Command Prompt for VS 2022
    echo.
    echo 然后在命令提示符中运行:
    echo   cd /d D:\LKDesktop\LINGYIN\coding_space\CIS5056\P2
    echo   build_from_vs_prompt.bat
    echo.
    pause
  exit /b 1
)

set PROJECT_ROOT=D:\LKDesktop\LINGYIN\coding_space\CIS5056\P2
cd /d "%PROJECT_ROOT%"

echo [1/4] 清理旧的构建...
if exist "out\build\x64-Release-VS" (
    rmdir /s /q "out\build\x64-Release-VS"
)
echo       已清理

echo [2/4] 配置 CMake (使用 Visual Studio 生成器)...
cmake -S . -B out\build\x64-Release-VS ^
    -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_CUDA_ARCHITECTURES=75

if errorlevel 1 (
    echo     配置失败！
    pause
    exit /b 1
)
echo       配置成功

echo [3/4] 编译项目 (Release 模式)...
cmake --build out\build\x64-Release-VS --config Release -j

if errorlevel 1 (
    echo 编译失败！
    pause
    exit /b 1
)
echo       编译成功

echo [4/4] 复制可执行文件...
if exist "out\build\x64-Release-VS\bin\Release\cis5650_stream_compaction_test.exe" (
    copy /y "out\build\x64-Release-VS\bin\Release\cis5650_stream_compaction_test.exe" ^
    "cis5650_stream_compaction_OPTIMIZED.exe"
    echo     ? 已生成: cis5650_stream_compaction_OPTIMIZED.exe
) else (
    echo       ? 未找到可执行文件
    pause
    exit /b 1
)

echo.
echo ===========================================
echo   构建完成！
echo ===========================================
echo.
echo 运行程序:
echo   cis5650_stream_compaction_OPTIMIZED.exe
echo.

pause

:: 询问是否运行
set /p RUN="是否立即运行程序？(Y/N): "
if /i "%RUN%"=="Y" (
    echo.
    cis5650_stream_compaction_OPTIMIZED.exe
)

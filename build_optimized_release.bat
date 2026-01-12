@echo off
chcp 65001 >nul
echo.
echo ===========================================
echo   生成优化的 Release 可执行程序
echo GPU: NVIDIA TITAN RTX (计算能力 7.5)
echo   优化级别: O3 + fast_math
echo ===========================================
echo.

set PROJECT_ROOT=D:\LKDesktop\LINGYIN\coding_space\CIS5056\P2
cd /d "%PROJECT_ROOT%"

echo [1/5] 清理旧的构建文件...
if exist "out\build\x64-Release-Optimized" (
    rmdir /s /q "out\build\x64-Release-Optimized"
    echo       已清理旧的构建目录
)

echo [2/5] 创建构建目录...
mkdir "out\build\x64-Release-Optimized"
cd "out\build\x64-Release-Optimized"

echo [3/5] 配置 CMake (Release 模式)...
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 ..\..\..
if errorlevel 1 (
    echo       CMake 配置失败！
    pause
  exit /b 1
)
echo       CMake 配置成功

echo [4/5] 编译项目 (可能需要几分钟)...
cmake --build . --config Release -j
if errorlevel 1 (
    echo       编译失败！
    pause
 exit /b 1
)
echo       编译成功

echo [5/5] 复制优化后的可执行文件...
if exist "bin\cis5650_stream_compaction_test.exe" (
    copy /y "bin\cis5650_stream_compaction_test.exe" "%PROJECT_ROOT%\cis5650_stream_compaction_OPTIMIZED.exe"
    echo       已复制到: %PROJECT_ROOT%\cis5650_stream_compaction_OPTIMIZED.exe
) else (
    echo       可执行文件未找到！
    pause
    exit /b 1
)

echo.
echo ===========================================
echo   构建完成！
echo ===========================================
echo.
echo 优化后的可执行文件位置:
echo   %PROJECT_ROOT%\cis5650_stream_compaction_OPTIMIZED.exe
echo.
echo 运行程序:
echo   cd %PROJECT_ROOT%
echo   cis5650_stream_compaction_OPTIMIZED.exe
echo.

cd /d "%PROJECT_ROOT%"
pause

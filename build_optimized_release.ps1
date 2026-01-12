# 生成完全优化的 Release 版本
# 无调试信息，O3 优化，针对 TITAN RTX (计算能力 7.5) 优化

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "  生成优化的 Release 可执行程序" -ForegroundColor Cyan
Write-Host "  GPU: NVIDIA TITAN RTX (计算能力 7.5)" -ForegroundColor Cyan
Write-Host "  优化级别: O3 + fast_math" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# 设置项目路径
$ProjectRoot = "D:\LKDesktop\LINGYIN\coding_space\CIS5056\P2"
Set-Location $ProjectRoot

# 检查是否在 Visual Studio 开发者命令提示符环境中
if (-not $env:VSINSTALLDIR) {
    Write-Host "警告: 未检测到 Visual Studio 环境" -ForegroundColor Yellow
    Write-Host "正在初始化 Visual Studio 2022 环境..." -ForegroundColor Yellow
    
    # 初始化 Visual Studio 2022 环境
    $vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community"
    $vcvarsPath = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"
    
    if (Test-Path $vcvarsPath) {
        # 在 cmd 中运行 vcvars64.bat 并导出环境变量
   $tempFile = [System.IO.Path]::GetTempFileName()
   cmd /c "`"$vcvarsPath`" && set" | Out-File -FilePath $tempFile -Encoding ASCII
    
        Get-Content $tempFile | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
           [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
  }
        }
        Remove-Item $tempFile
        Write-Host "? Visual Studio 环境已初始化" -ForegroundColor Green
    } else {
        Write-Host "错误: 找不到 Visual Studio 2022" -ForegroundColor Red
        Write-Host "请从 'x64 Native Tools Command Prompt for VS 2022' 运行此脚本" -ForegroundColor Yellow
        pause
        exit 1
    }
}

# 清理旧的构建
Write-Host "[1/5] 清理旧的构建文件..." -ForegroundColor Yellow
if (Test-Path "out\build\x64-Release-Optimized") {
    Remove-Item -Recurse -Force "out\build\x64-Release-Optimized"
    Write-Host "      已清理旧的构建目录" -ForegroundColor Green
}

# 创建构建目录
Write-Host "[2/5] 创建构建目录..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "out\build\x64-Release-Optimized" | Out-Null
Set-Location "out\build\x64-Release-Optimized"

# 配置 CMake
Write-Host "[3/5] 配置 CMake (Release 模式)..." -ForegroundColor Yellow
cmake -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CUDA_ARCHITECTURES=75 `
    ..\..\..

if ($LASTEXITCODE -ne 0) {
    Write-Host "      CMake 配置失败！" -ForegroundColor Red
    Write-Host ""
    Write-Host "尝试清理并使用 Ninja 生成器..." -ForegroundColor Yellow
Set-Location $ProjectRoot
    Remove-Item -Recurse -Force "out\build\x64-Release-Optimized"
    New-Item -ItemType Directory -Force -Path "out\build\x64-Release-Optimized" | Out-Null
    Set-Location "out\build\x64-Release-Optimized"
    
    cmake -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
        -DCMAKE_CUDA_ARCHITECTURES=75 `
      -DCMAKE_CUDA_HOST_COMPILER="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" `
        ..\..\..
    
    if ($LASTEXITCODE -ne 0) {
 Write-Host "      CMake 配置仍然失败！" -ForegroundColor Red
        pause
        exit 1
    }
}
Write-Host "      CMake 配置成功" -ForegroundColor Green

# 编译
Write-Host "[4/5] 编译项目 (可能需要几分钟)..." -ForegroundColor Yellow
cmake --build . --config Release -j

if ($LASTEXITCODE -ne 0) {
    Write-Host "      编译失败！" -ForegroundColor Red
 pause
    exit 1
}
Write-Host "      编译成功" -ForegroundColor Green

# 复制可执行文件到根目录
Write-Host "[5/5] 复制优化后的可执行文件..." -ForegroundColor Yellow
$ExePath = "bin\Release\cis5650_stream_compaction_test.exe"
if (-not (Test-Path $ExePath)) {
    $ExePath = "bin\cis5650_stream_compaction_test.exe"
}

if (Test-Path $ExePath) {
    Copy-Item $ExePath "$ProjectRoot\cis5650_stream_compaction_OPTIMIZED.exe" -Force
    Write-Host "      已复制到: $ProjectRoot\cis5650_stream_compaction_OPTIMIZED.exe" -ForegroundColor Green
} else {
    Write-Host "      可执行文件未找到！" -ForegroundColor Red
    Write-Host "      查找路径: $ExePath" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "  构建完成！" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "优化后的可执行文件位置:" -ForegroundColor Yellow
Write-Host "  $ProjectRoot\cis5650_stream_compaction_OPTIMIZED.exe" -ForegroundColor White
Write-Host ""
Write-Host "运行程序:" -ForegroundColor Yellow
Write-Host "  cd $ProjectRoot" -ForegroundColor White
Write-Host "  .\cis5650_stream_compaction_OPTIMIZED.exe" -ForegroundColor White
Write-Host ""

# 显示文件信息
$FileInfo = Get-Item "$ProjectRoot\cis5650_stream_compaction_OPTIMIZED.exe"
Write-Host "文件信息:" -ForegroundColor Yellow
Write-Host "  大小: $([math]::Round($FileInfo.Length / 1MB, 2)) MB" -ForegroundColor White
Write-Host "  创建时间: $($FileInfo.CreationTime)" -ForegroundColor White
Write-Host ""

# 询问是否立即运行
$Response = Read-Host "是否立即运行程序？(Y/N)"
if ($Response -eq "Y" -or $Response -eq "y") {
    Write-Host ""
 Write-Host "===========================================" -ForegroundColor Cyan
    Write-Host "  运行优化后的程序" -ForegroundColor Cyan
    Write-Host "===========================================" -ForegroundColor Cyan
    Write-Host ""
    Set-Location $ProjectRoot
    & ".\cis5650_stream_compaction_OPTIMIZED.exe"
}

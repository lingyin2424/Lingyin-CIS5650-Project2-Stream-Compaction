# 快速构建优化版本

Write-Host "`n?? 快速构建优化的 Release 版本`n" -ForegroundColor Cyan

$ProjectRoot = "D:\LKDesktop\LINGYIN\coding_space\CIS5056\P2"
cd $ProjectRoot

# 清理
Write-Host "清理旧文件..." -ForegroundColor Yellow
if (Test-Path "out\build\x64-Release-Optimized") {
    Remove-Item -Recurse -Force "out\build\x64-Release-Optimized"
}
New-Item -ItemType Directory -Force -Path "out\build\x64-Release-Optimized" | Out-Null

# 配置
Write-Host "配置 CMake..." -ForegroundColor Yellow
cd "out\build\x64-Release-Optimized"
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 ..\..\.. 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "? 配置成功" -ForegroundColor Green
} else {
    Write-Host "? 配置失败" -ForegroundColor Red
    exit 1
}

# 编译
Write-Host "编译中..." -ForegroundColor Yellow
cmake --build . --config Release -j 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "? 编译成功" -ForegroundColor Green
} else {
    Write-Host "? 编译失败" -ForegroundColor Red
    exit 1
}

# 复制
if (Test-Path "bin\cis5650_stream_compaction_test.exe") {
    Copy-Item "bin\cis5650_stream_compaction_test.exe" "$ProjectRoot\cis5650_stream_compaction_OPTIMIZED.exe" -Force
    $FileSize = [math]::Round((Get-Item "$ProjectRoot\cis5650_stream_compaction_OPTIMIZED.exe").Length / 1MB, 2)
    Write-Host "? 已生成: cis5650_stream_compaction_OPTIMIZED.exe ($FileSize MB)" -ForegroundColor Green
    
    cd $ProjectRoot
    Write-Host "`n运行命令: .\cis5650_stream_compaction_OPTIMIZED.exe`n" -ForegroundColor Cyan
} else {
    Write-Host "? 未找到可执行文件" -ForegroundColor Red
    exit 1
}

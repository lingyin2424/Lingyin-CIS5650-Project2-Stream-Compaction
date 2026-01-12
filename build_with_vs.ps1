# 使用 Visual Studio 直接构建优化版本（推荐方法）

Write-Host "`n?? 使用 Visual Studio 构建优化版本`n" -ForegroundColor Cyan

$ProjectRoot = "D:\LKDesktop\LINGYIN\coding_space\CIS5056\P2"
cd $ProjectRoot

Write-Host "方法说明:" -ForegroundColor Yellow
Write-Host "此方法直接使用 Visual Studio 的内置 CMake 支持，避免环境配置问题。`n" -ForegroundColor Gray

Write-Host "步骤 1: 在 Visual Studio 中打开项目" -ForegroundColor Cyan
Write-Host "  - 打开 Visual Studio 2022" -ForegroundColor White
Write-Host "  - 文件 -> 打开 -> CMake..." -ForegroundColor White
Write-Host "  - 选择: $ProjectRoot\CMakeLists.txt`n" -ForegroundColor White

Write-Host "步骤 2: 配置为 Release 模式" -ForegroundColor Cyan
Write-Host "  - 在工具栏中选择配置: x64-Release" -ForegroundColor White
Write-Host "  - 右键点击 CMakeLists.txt" -ForegroundColor White
Write-Host "  - 选择 '删除缓存并重新配置'`n" -ForegroundColor White

Write-Host "步骤 3: 构建项目" -ForegroundColor Cyan
Write-Host "  - 生成 -> 重新生成解决方案" -ForegroundColor White
Write-Host "  - 或按 Ctrl+Shift+B`n" -ForegroundColor White

Write-Host "步骤 4: 运行程序" -ForegroundColor Cyan
Write-Host "  - 调试 -> 开始执行（不调试）" -ForegroundColor White
Write-Host "  - 或按 Ctrl+F5`n" -ForegroundColor White

Write-Host "可执行文件位置:" -ForegroundColor Yellow
Write-Host "  $ProjectRoot\out\build\x64-Release\bin\cis5650_stream_compaction_test.exe`n" -ForegroundColor White

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "或者使用命令行方法（需要管理员权限）:" -ForegroundColor Cyan
Write-Host "================================================`n" -ForegroundColor Cyan

Write-Host "1. 打开 'x64 Native Tools Command Prompt for VS 2022'" -ForegroundColor Yellow
Write-Host "   位置: 开始菜单 -> Visual Studio 2022 -> x64 Native Tools Command Prompt`n" -ForegroundColor Gray

Write-Host "2. 运行以下命令:" -ForegroundColor Yellow
Write-Host @"
cd /d $ProjectRoot
rmdir /s /q out\build\x64-Release 2>nul
cmake -S . -B out\build\x64-Release -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build out\build\x64-Release --config Release -j
"@ -ForegroundColor White

Write-Host "`n按任意键打开 Visual Studio..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# 尝试打开 Visual Studio
$vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\devenv.exe"
if (Test-Path $vsPath) {
    & $vsPath "$ProjectRoot\CMakeLists.txt"
    Write-Host "`n? 已启动 Visual Studio" -ForegroundColor Green
} else {
    Write-Host "`n? 未找到 Visual Studio" -ForegroundColor Red
    Write-Host "请手动打开 Visual Studio 并打开 CMakeLists.txt" -ForegroundColor Yellow
}

# ?? 快速开始 - 生成优化 Release 版本

## ?? cudafe++ 崩溃问题已知

如果你遇到 `cudafe++ died with status 0xC0000005` 错误，不要担心！

---

## ? 解决方案：使用 Visual Studio IDE（推荐）

### 5 个简单步骤：

#### 1?? 打开 Visual Studio 2022

#### 2?? 打开 CMake 项目
```
文件 -> 打开 -> CMake...
选择: D:\LKDesktop\LINGYIN\coding_space\CIS5056\P2\CMakeLists.txt
```

#### 3?? 选择 Release 配置
- 在顶部工具栏，配置下拉菜单中选择: **x64-Release**

#### 4?? 重新配置
- 右键点击 `CMakeLists.txt`（在解决方案资源管理器中）
- 选择 **"删除缓存并重新配置"**
- 等待配置完成（可能需要 1-2 分钟）

#### 5?? 构建并运行
- **生成 -> 重新生成解决方案** (或按 `Ctrl+Shift+B`)
- **调试 -> 开始执行（不调试）** (或按 `Ctrl+F5`)

---

## ?? 完成！

可执行文件位置:
```
D:\LKDesktop\LINGYIN\coding_space\CIS5056\P2\out\build\x64-Release\bin\cis5650_stream_compaction_test.exe
```

---

## ?? 期望输出

```
****************
** SCAN TESTS **
****************
    [   5   1  30   9 ... ]
==== cpu scan, power-of-two ====
   elapsed time: 5.032ms
==== naive scan, power-of-two ====
   elapsed time: 2.3ms    ← 优化后更快
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 1.5ms      ← 优化后更快
    passed
```

---

## ??? 备选方案

### 方案 A: 使用 VS 命令提示符

1. 开始菜单搜索 **"x64 Native Tools Command Prompt for VS 2022"**
2. 以管理员身份运行
3. 执行:
```batch
cd /d D:\LKDesktop\LINGYIN\coding_space\CIS5056\P2
build_from_vs_prompt.bat
```

### 方案 B: 运行 PowerShell 脚本

```powershell
cd D:\LKDesktop\LINGYIN\coding_space\CIS5056\P2
.\build_with_vs.ps1
```

这会自动打开 Visual Studio 并引导你。

---

## ?? 详细文档

| 文档 | 说明 |
|------|------|
| **CUDAFE_FIX_GUIDE.md** | cudafe++ 崩溃详细解决方案 ?? |
| **RELEASE_BUILD_SUMMARY.md** | 优化配置完整说明 |
| **BUILD_OPTIMIZED_GUIDE.md** | 详细构建指南 |

---

## ? 常见问题

### Q: 为什么会崩溃？
**A**: Ninja 生成器在 Windows 上容易出现环境问题。使用 Visual Studio 生成器更稳定。

### Q: 性能会受影响吗？
**A**: 不会！无论使用哪个生成器，最终的优化级别是一样的。

### Q: 我必须用 Visual Studio 吗？
**A**: 在 Windows 上，这是最可靠的方法。如果必须用命令行，使用 VS 命令提示符。

---

## ? 优化配置（已自动应用）

- ? C++ O2 优化
- ? CUDA O3 优化
- ? 快速数学库 (fast_math)
- ? 链接时优化 (LTCG)
- ? 专为 TITAN RTX (7.5) 优化
- ? 无调试信息

---

## ?? 立即开始！

**打开 Visual Studio，开始构建！** ??

如果需要帮助，查看 `CUDAFE_FIX_GUIDE.md`

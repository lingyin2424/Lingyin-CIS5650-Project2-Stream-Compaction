# ?? 一键生成优化 Release 版本

## 最简单的方法

### Windows PowerShell（推荐）

```powershell
# 1. 打开 PowerShell
# 2. 进入项目目录
cd D:\LKDesktop\LINGYIN\coding_space\CIS5056\P2

# 3. 运行构建脚本（三选一）

# 选项 A: 完整版（详细输出，推荐第一次使用）
.\build_optimized_release.ps1

# 选项 B: 快速版（简洁输出，适合重复构建）
.\quick_build.ps1

# 选项 C: 批处理版（最简单）
.\build_optimized_release.bat
```

### 构建完成后

```powershell
# 运行优化后的程序
.\cis5650_stream_compaction_OPTIMIZED.exe
```

---

## 优化配置一览

### ? 已配置的优化

| 优化项 | 设置 | 效果 |
|--------|------|------|
| **C++ 优化** | `/O2 /Ob2 /Oi /Ot /GL` | 最大化速度 |
| **CUDA 优化** | `-O3 --use_fast_math` | 最高级别 + 快速数学 |
| **链接优化** | `/LTCG /OPT:REF /OPT:ICF` | 链接时代码生成 |
| **GPU 架构** | `75` (Turing) | 专为 TITAN RTX 优化 |
| **调试信息** | 无 | 更小体积 |

### ?? 预期效果

- **性能提升**: 2-3倍（相比 Debug）
- **文件大小**: 减小 3-4倍
- **编译时间**: 约 2-5 分钟

---

## 快速命令参考

```powershell
# 完整清理重建
Remove-Item -Recurse -Force out
.\build_optimized_release.ps1

# 只重新编译（不清理）
cd out\build\x64-Release-Optimized
cmake --build . --config Release -j

# 运行程序
cd D:\LKDesktop\LINGYIN\coding_space\CIS5056\P2
.\cis5650_stream_compaction_OPTIMIZED.exe
```

---

## ?? 详细文档

| 文档 | 用途 |
|------|------|
| `RELEASE_BUILD_SUMMARY.md` | 完整的优化说明和验证 |
| `BUILD_OPTIMIZED_GUIDE.md` | 详细的构建指南 |
| `PROJECT_SUMMARY.md` | 项目总体修改总结 |

---

## ?? 注意事项

1. **首次构建**: 使用 `build_optimized_release.ps1` 查看详细输出
2. **脚本权限**: 如果 PowerShell 报错，运行：
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```
3. **GPU 架构**: 当前只编译 TITAN RTX (7.5) 架构，其他 GPU 无法运行

---

## ? 验证

运行后应该看到类似输出：

```
****************
** SCAN TESTS **
****************
==== naive scan, power-of-two ====
 elapsed time: 2.3ms  ← 优化后的时间
  passed
==== work-efficient scan, power-of-two ====
   elapsed time: 1.5ms      ← 优化后的时间
    passed
```

如果时间明显快于之前，说明优化成功！??

---

**现在就运行脚本生成优化版本吧！** ??

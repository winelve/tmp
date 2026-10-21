# SuiteSparse SpMM 性能测试

在服务器仓库根目录运行，使用已安装本仓库 SDAA 扩展的 Python 环境。
需要 torch、torch_sdaa、torch_scatter、NumPy 和 SciPy。
脚本强制加载当前仓库的 torch_sparse，扩展必须已编译到 torch_sparse 目录。

服务器不需要接收本地 data 目录。八个矩阵的下载链接和分类已内置在
benchmark/suitesparse_sdaa.py 中，不依赖 data 内的任何脚本。
先下载全部数据，只需 Python 标准库，不需要安装计算依赖：

```bash
python benchmark/suitesparse_sdaa.py --download-only
```

优先使用佛罗里达大学旧站，失败时尝试 SuiteSparse 新站。
缺少的目录自动创建，已有非空文件直接复用；下载先写临时文件，
检查响应长度和 MATLAB 文件头后才保存为 .mat。下载失败可重新运行，
已完成的文件不会重复下载；未完成的文件从头下载。
这些检查不替代文件校验和验证，正式测试仍会解析矩阵并核对计算结果。
下载完全位于计时之外。

先用一个小矩阵确认环境和结果正确：

```bash
python benchmark/suitesparse_sdaa.py --matrices yeast --sizes 16 --warmup 2 --iterations 3 --rounds 2
```

正式测试全部矩阵：

```bash
python benchmark/suitesparse_sdaa.py --device sdaa:0 --sizes 16 32 64 128
```

默认测试内置的八个矩阵，在 data/suitesparse/mat 中递归查找已有文件，
缺少时下载到 small、medium、large 子目录。不加 --download-only 时下载后继续测试。
可通过 --matrices 指定部分矩阵，也可指定目录中已有的其他矩阵文件名。
如果数据位于其他位置，添加 `--data-dir /实际路径/data/suitesparse/mat`。
`--sizes` 指普通矩阵 X 的列数 K：计算 A[M,N] × X[N,K]，结果为 Y[M,K]。
默认保留 A 的原始数值，转换成 float32，合并重复位置并去掉零值。
索引使用 int64。默认只测求和的前向运算，不计算梯度。
如需复现旧脚本的无权重输入，添加 `--value-mode implicit-ones`；两种结果必须分开比较。

每个案例先用 CPU 上的 SciPy 计算核对全部输出，再预热 10 次。
校验逐项要求绝对误差不超过 `1e-4 + 1e-4 * (abs(A) @ abs(X))`，
允许浮点加法顺序造成的误差，并记录最大绝对误差；不通过就停止。
校验分块处理输出，不会把稀疏 A 展开成普通矩阵。

正式计时重复 5 轮，每轮连续调用 20 次：

1. 等待当前 SDAA 设备完成已有工作。
2. 开始主机时钟计时，连续调用 `SparseTensor @ X`。
3. 等待设备完成本轮全部计算，停止计时。
4. 总耗时除以 20，换算成单次毫秒数。

计时包含 Python 调用、算子调用和输出分配及释放；不包含文件读取、
格式整理、输入传输、正确性校验和预热。这是数据已在设备上时的算子调用耗时，
不是单独设备计算代码的耗时。mean_ms 是各轮单次耗时的平均值，
median_ms 是各轮单次耗时的中位数，不是逐次调用耗时的中位数。

结果自动写到 benchmark/results 下独立的时间目录：

- results.csv：来源、M/N/K、非零元素数、精度、配置、正确性误差和耗时。
- environment.json：软件版本、代码提交号、工作区状态和 teco-smi 输出。
- runner.py：本次运行脚本的副本，便于归档；重新运行时放回仓库 benchmark 目录。

每个成功案例立即保存。计算失败会记录失败行并停止，已有结果保留。
读文件或构建输入失败会直接停止，错误显示在终端。
若内存不足，另起一次运行减小 K，不要把失败当作正常时间。

全部案例在一个进程中依次运行，只使用指定设备；脚本不会并行使用四个设备编号。
它不能阻止其他账号或进程使用同一张物理卡。运行前协调占用时间，使用约定的账号，
确认整卡空闲，并避免同时启动另一份测试。
数据文件尚未加入 Git，交付时需另行包含数据或来源清单。

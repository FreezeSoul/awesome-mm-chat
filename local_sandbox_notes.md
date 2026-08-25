# Local Sandbox 设计与风险说明

这份文档记录 Dressage 当前 `local_bwrap` sandbox 的使用方式、核心实现和安全边界。结论先说：它是一个基于 Ray slot pool + bubblewrap 的本地执行环境，适合在本机批量跑 blackbox/whitebox rollout，但默认不是强安全容器。它主要隔离文件写入路径和进程生命周期，不默认隔离网络，也不默认提供完整 cgroup 资源限制。

## 1. local sandbox 是什么

Dressage 的 sandbox provider 由 `DRESSAGE_SANDBOX_PROVIDER` 选择：

- `local_bwrap`：本地 bubblewrap slot pool。
- `e2b`：远端 E2B sandbox。

`local_bwrap` 不是每次 rollout 临时启动一个全新容器，而是先在 Ray 上启动一个本地 slot pool。每个 slot 有自己的目录、端口和运行状态。rollout 需要 sandbox 时，从 pool 里 acquire 一个 slot；用完后 release，supervisor 再 reset 这个 slot。

它支持两种 pool mode：

- `blackbox`：每个 slot 里启动一个 BlackboxServer，对外暴露 HTTP endpoint。blackbox agent，比如 opencode/openclaw，通过这个 server 被注册、调用和清理。
- `command_only`：不启动 BlackboxServer，只提供命令执行和文件读写能力。whitebox paddock 的工具调用会走这一路。

对应关系大致是：

| Paddock mode | Sandbox provider | Local pool mode | 用途 |
| --- | --- | --- | --- |
| `blackbox` | `local_bwrap` | `blackbox` | 跑黑盒 agent rollout |
| `whitebox` | `local_bwrap` | `command_only` | 跑白盒工具调用 |

代码会检查 paddock mode 和 pool mode 是否匹配，避免 blackbox paddock 拿到 command-only slot 这种错误组合。

## 2. 如何使用

blackbox local sandbox 的典型配置：

```bash
export DRESSAGE_PADDOCK_MODE=blackbox
export DRESSAGE_SANDBOX_PROVIDER=local_bwrap
export DRESSAGE_LOCAL_BWRAP_POOL_MODE=blackbox
export DRESSAGE_BLACKBOX_TYPE=opencode

ray start --head --resources='{"local_bwrap_slots":4,"local_bwrap_node":1}'
dressage-local-bwrap-start
dressage-local-bwrap-status
```

whitebox local sandbox 的典型配置：

```bash
export DRESSAGE_PADDOCK_MODE=whitebox
export DRESSAGE_SANDBOX_PROVIDER=local_bwrap
export DRESSAGE_LOCAL_BWRAP_POOL_MODE=command_only

ray start --head --resources='{"local_bwrap_slots":4,"local_bwrap_node":1}'
dressage-local-bwrap-start
dressage-local-bwrap-status
```

核心命令：

- `dressage-local-bwrap-start`：启动 Ray manager/supervisor，并初始化 slot pool。
- `dressage-local-bwrap-status`：查看 pool、node、slot 状态。
- rollout 进程启动后，会通过 sandbox factory 自动连接已有 manager，并 acquire/release slot。

## 3. 核心实现链路

入口在 `dressage/sandbox/factory.py`。当 provider 选择 `local_bwrap` 时，会创建 `LocalBwrapSandboxProvider`。

`LocalBwrapSandboxProvider` 本身不直接管理进程，而是连接 Ray detached actor：

- `dressage/sandbox/local/bwrap/manager.py`
  - `LocalBwrapClusterManager` 是集群级 manager。
  - 它扫描 Ray nodes 上的 `local_bwrap_slots` 资源。
  - 每个 node 创建一个 `LocalBwrapNodeSupervisor`。
  - `acquire()` 时从有空闲 slot 的 node 里拿 lease。
  - `release()` 后触发后台 reset。

- `dressage/sandbox/local/bwrap/supervisor.py`
  - 每个 node 一个 supervisor。
  - 负责 slot 生命周期：启动、健康检查、租约、reset、重启失败 slot。
  - blackbox mode 下，slot 会运行 BlackboxServer。
  - command-only mode 下，只准备目录和命令执行环境。

- `dressage/sandbox/local/bwrap/runner.py`
  - 负责拼 bubblewrap 命令。
  - 设置只读挂载、可写挂载、环境变量、工作目录、进程清理策略。
  - 可选通过 systemd scope 加内存和进程数限制。

- `dressage/sandbox/local/bwrap/slot.py`
  - 定义 slot 的目录结构。
  - 每个 slot 有 `home/`、`work/`、`runtime/`、`tmp/`、`logs/`、`archives/` 等目录。

blackbox paddock 的链路是：

1. `BlackboxAgentPaddock.init()` 创建 `SandboxSpec`，请求一个带 `blackbox` service 的 local sandbox。
2. provider acquire 一个 slot，返回 BlackboxServer endpoint。
3. paddock 调 `/register_agent`，告诉 BlackboxServer 使用哪个 agent backend、LLM proxy URL 等。
4. rollout 调 `/call_agent`。
5. rollout 结束后 paddock terminate lease，provider release slot。
6. supervisor reset slot，必要时归档 `home/work/runtime/tmp`。

whitebox paddock 的链路是：

1. `WhiteboxPaddock` 请求 command-only slot。
2. 工具调用走 provider 的 `run_command/read_file/write_file`。
3. provider 把 sandbox 内路径映射到 slot 的 host 目录。
4. release 后 reset slot。

## 4. bubblewrap 里具体挂了什么

默认不是从完整 container image 启动，而是从空 namespace 开始，把宿主机上需要的路径 bind 进去。

默认只读挂载包括：

- `/usr`
- `/bin`
- `/sbin`
- `/lib`
- `/lib64`
- `/etc`
- 当前 Python 路径
- `PYTHONPATH` 相关路径
- 常见 opencode/openclaw 安装目录，比如 `~/.opencode`、`~/.openclaw`、`~/.local`

默认可写挂载包括：

- slot 的 `home_dir` -> `/home/blackbox`
- slot 的 `work_dir` -> `/workspace`
- slot 的 `runtime_dir` -> `/workspace_sandbox/blackbox_server_runtime`
- slot 的 `tmp_dir` -> `/tmp`

默认环境变量会把：

- `HOME` 指到 `/home/blackbox`
- `XDG_CONFIG_HOME` 指到 `/home/blackbox/.config`
- `XDG_CACHE_HOME` 指到 `/home/blackbox/.cache`
- `TMPDIR` 指到 `/tmp`
- `CUDA_VISIBLE_DEVICES` 置空
- `NVIDIA_VISIBLE_DEVICES=void`

所以大部分 agent 自己产生的配置、cache、临时文件、repo 修改、构建产物，都会落在 slot 私有目录，而不是写到宿主机的真实 home/workspace 系统路径。

## 5. apt-get 会不会破坏宿主机

默认 `local_bwrap` 下，`apt-get install` 通常不会破坏宿主机环境。

原因是：

- `/usr`、`/bin`、`/lib`、`/etc` 等系统路径是只读挂载。
- dpkg/apt 需要写系统路径和数据库，通常会因为权限或只读文件系统失败。
- sandbox 内默认 uid/gid 是当前用户，不是宿主机 root。
- 可写路径只限 slot 的 `/home/blackbox`、`/workspace`、runtime、`/tmp`。

因此常见结果是：

- `apt-get update/install` 失败。
- 下载文件或 cache 可能写到 slot 的可写目录。
- 不会把宿主机 `/usr`、`/etc`、dpkg 数据库写坏。

但以下情况有风险：

- 设置了 `DRESSAGE_BLACKBOX_RUNNER_MODE=direct`。direct mode 不走 bwrap 隔离，命令就是在宿主机环境跑。
- 通过 `DRESSAGE_BLACKBOX_BWRAP_EXTRA_ARGS` 额外把宿主机系统目录以可写方式 bind 进去。
- 自定义 `DRESSAGE_BLACKBOX_READONLY_MOUNTS` 或 `DRESSAGE_BLACKBOX_ROOTFS` 时改变了原有只读边界。
- agent 本身拿到了宿主机高权限凭据或可写挂载。

所以默认情况下 apt 不会污染宿主机；但这个保证依赖 runner mode 和 mount 配置没有被放宽。

## 6. 安全边界和限制

这个 local sandbox 不能按强安全容器理解。默认配置里有几个重要边界：

### 6.1 网络默认不隔离

`DRESSAGE_BLACKBOX_BWRAP_UNSHARE_NET` 默认是 false。也就是说 sandbox 默认使用宿主机网络 namespace。

这是为了让 BlackboxServer endpoint、LLM proxy、agent backend 等本地服务更容易互通。但代价是：

- agent 可以访问宿主机可达的网络。
- 对恶意代码来说，这不是网络隔离环境。
- 如果启用 `--unshare-net`，BlackboxServer 可达性和本地服务访问可能需要额外设计。

### 6.2 PID 默认不隔离

`DRESSAGE_BLACKBOX_BWRAP_UNSHARE_PID` 默认是 false。

runner 会通过进程组和环境变量 marker 清理残留进程。默认能看到宿主 `/proc` 的只读视图，这有利于清理，但不是强进程隔离。

### 6.3 资源限制默认不强

配置里支持 `DRESSAGE_BLACKBOX_USE_SYSTEMD_SCOPE=1`，开启后可以通过 systemd scope 设置：

- `MemoryHigh`
- `MemoryMax`
- `TasksMax`
- `CPUWeight`

但默认不开启。不开启时，内存、进程数等更多依赖外层 Ray/系统环境，不是严格 cgroup 限制。

### 6.4 文件系统是主要隔离面

默认最有用的保护是：

- 系统路径只读。
- agent 的 home/work/tmp/runtime 都映射到 slot 私有目录。
- slot release 后 reset，可选归档。

这能避免大多数普通工具调用污染宿主机项目和系统目录，但不能抵御所有恶意行为。

## 7. slot reset 和归档

slot 用完后会 release，supervisor 负责 reset。blackbox mode 下通常是 hard reset：

1. 停 BlackboxServer 进程。
2. 清理残留子进程。
3. 可选归档 `home/work/runtime/tmp`。
4. 重置 slot 目录。
5. 重新启动 BlackboxServer。

归档相关配置：

```bash
export DRESSAGE_BLACKBOX_PRESERVE_SESSION_ARTIFACTS=1
export DRESSAGE_BLACKBOX_SESSION_ARCHIVE_DIRS=home,work,runtime,tmp
export DRESSAGE_BLACKBOX_SESSION_ARCHIVE_MAX_PER_SLOT=20
export DRESSAGE_BLACKBOX_SESSION_ARCHIVE_TTL_SEC=86400
```

归档用于排查失败 rollout：可以看到 agent 最后留下的 workspace、cache、日志和 runtime 状态。

## 8. 适合与不适合的场景

适合：

- 本地批量 rollout。
- blackbox agent 的工程验证。
- whitebox 工具调用的轻量隔离。
- 防止普通 agent 把系统目录或真实 home 目录写乱。
- 调试 rollout 失败现场。

不适合：

- 运行完全不可信代码并期待强安全隔离。
- 需要默认断网的安全沙箱。
- 需要严格 CPU/内存/进程数隔离但未开启 systemd scope。
- 需要可变系统环境，比如在 sandbox 内真的 `apt-get install` 系统包。

如果需要更强隔离，优先考虑 E2B、真正的 container/VM，或者至少显式开启网络/PID namespace、cgroup 限制，并仔细收紧 bind mounts。

## 9. 关键结论

- `local_bwrap` 是本地 slot pool，不是每次全新容器。
- 它用 bubblewrap 做文件 namespace，系统路径默认只读。
- agent 可写内容主要进入 slot 的 `home/work/runtime/tmp`。
- 默认网络和 PID 不强隔离。
- `apt-get install` 默认不会破坏宿主机，通常会失败。
- `direct` mode 或额外可写 bind mount 会显著扩大风险。
- 它适合作为 rollout 工程隔离层，但不应该被当作高强度安全边界。

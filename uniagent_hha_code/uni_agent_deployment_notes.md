# Uni-Agent Deployment 模块说明

> 日期：2026-07-12  
> 位置：`uni_agent/deployment/`  
> 目标：解释 Uni-Agent 里各种 deployment 后端到底是什么、怎么执行命令、安全边界在哪里。

## 1. Deployment 是什么

在 Uni-Agent 里，agent 的工具调用最终通常会变成 shell 命令或文件操作。`deployment` 决定这些命令在哪里执行。

统一入口是：

```text
AgentInteraction
  -> AgentEnv
  -> deployment.runtime
  -> run_in_session / upload / read_file / write_file
```

也就是说，模型不是直接操作机器。模型调用 tool，tool 转成 action，`AgentEnv` 把 action 发给某个 deployment 的 runtime。

当前配置类型在 `uni_agent/deployment/config.py`：

```text
host
local_native
local
local_attach
modal
vefaas
```

可以先按隔离强度理解：

```text
host / local_native:
  无容器，直接宿主机执行，最不隔离。

local / local_attach:
  本机容器 sandbox，隔离强很多。

modal / vefaas:
  云端远程 sandbox，适合大规模并发和不可信任务。
```

## 2. host

文件：

```text
uni_agent/deployment/host/deployment.py
```

含义：

```text
直接在当前宿主机启动 bash，tool command 在宿主机执行。
不启动容器，不连接 HTTP server。
```

执行链路：

```text
type: host
  -> HostDeployment.start()
  -> HostRuntime.create_session()
  -> asyncio.create_subprocess_exec("bash", "--norc", "--noprofile")
  -> run_in_session(BashAction)
```

`HostRuntime` 做了一些运行稳定性处理：

- 用 `_io_lock` 串行化命令，避免 stdout/stdin 交错。
- 每条命令后面追加 marker，用 marker 判断输出结束和 exit code。
- timeout 时给前台子进程组发 `SIGINT`。
- timeout 后尝试 drain 残留 stdout，避免污染下一条命令。
- bash 死亡后标记 dead，不偷偷重启新 shell。
- 使用 `--norc --noprofile`，清空 `PS1/PS2/PROMPT_COMMAND`，避免 prompt 干扰 marker。

但它没有安全隔离：

- 没有 chroot。
- 没有容器。
- 没有 namespace/cgroup/seccomp。
- 没有降权。
- 没有路径限制。
- 没有命令 allowlist/denylist。
- `read_file/write_file/upload` 直接操作宿主机文件系统。

所以：

```text
host 的安全边界主要取决于你暴露哪些 tool。
```

如果只暴露安全收敛的工具，例如：

```yaml
tools:
  - name: search
  - name: finish
```

风险相对小。`examples/search_agent/agent_config.yaml` 就是这么用的，它需要访问本机/集群检索服务，不需要任意 bash。

如果暴露：

```yaml
tools:
  - name: execute_bash
```

那就是允许模型以当前用户权限在宿主机执行任意 shell，非常危险。

适合场景：

- 本地可信调试。
- 只暴露收敛工具，比如 search。
- 工具需要访问宿主机已有服务或认证状态。

不适合：

- SWE-Bench / Terminal-Bench / R2E 这类不可信代码任务。
- 任意 `execute_bash`。
- 会读写项目文件或系统文件的任务。

## 3. local_native

文件：

```text
uni_agent/deployment/local_native/deployment.py
uni_agent/deployment/local_native/runtime.py
```

含义：

```text
也是直接在宿主机执行命令，但用 pexpect + PTY 控制 bash。
不启动容器，不走 HTTP server。
```

和 `host` 的区别不是安全边界，而是控制 bash 的方式。

```text
host:
  asyncio subprocess pipe

local_native:
  pexpect + PTY
```

### 什么是 PTY

PTY 是 pseudo terminal，伪终端。它让程序以为自己连着一个真实终端。

普通 pipe：

```text
Python -> stdin pipe -> bash
Python <- stdout pipe <- bash
```

PTY：

```text
Python/pexpect <-> 伪终端 <-> bash
```

有些程序在 terminal 和 pipe 下行为不同。PTY 更像真人打开终端操作。

### 什么是 pexpect

`pexpect` 是 Python 自动操作终端的库。

一个最小例子：

```python
import pexpect

child = pexpect.spawn("bash", encoding="utf-8")
child.sendline("echo hello")
child.expect("hello")
print(child.before + child.after)
child.sendline("exit")
```

它做的事：

```text
启动 bash
输入 echo hello
等待输出里出现 hello
读取输出
退出
```

### 为什么需要 local_native

`host` 使用 `asyncio.create_subprocess_exec()`，bash 的 stdin/stdout pipe 会绑定创建它的 event loop。

如果上层代码用类似：

```python
asyncio.run(env.start())
asyncio.run(env.communicate("echo hi"))
```

每次 `asyncio.run()` 都创建新的 event loop。第一个 loop 创建了 bash pipe，第二个 loop 再读这个 pipe，可能报：

```text
got Future attached to a different loop
```

`local_native` 用 `pexpect + PTY`，底层同步控制终端，不绑定某个 asyncio event loop，所以更适合 sync-style 调用和 `auto_await` 场景。

安全性：

```text
local_native == host
```

它仍然是在宿主机执行命令，没有 sandbox。

适合场景：

- 需要复用宿主机认证状态，例如本机已登录 `lark-cli`。
- 希望程序像真实终端一样操作工具。
- 需要规避 `host` 的 asyncio event loop 问题。

不适合：

- 不可信任务。
- 暴露任意 bash 给模型。

## 4. local

文件：

```text
uni_agent/deployment/local/deployment.py
```

含义：

```text
Uni-Agent 在本机启动一个容器镜像作为 sandbox。
容器里启动 swerex.server。
AgentEnv 通过 HTTP + token 连接容器 runtime。
```

默认配置：

```yaml
deployment:
  type: local
  image: python:3.12
  command: >
    python3 -m pip install -q swe-rex &&
    python3 -m swerex.server --host 0.0.0.0 --port {port} --auth-token {token}
```

执行链路：

```text
type: local
  -> LocalDeployment.start()
  -> docker/podman/apptainer run image
  -> 容器内启动 swerex.server
  -> RemoteRuntime 连接 http://host:port
  -> create bash session
  -> tool command 在容器内执行
```

支持的 container runtime：

```text
apptainer / singularity
docker
podman
```

runtime 选择逻辑：

- 先看环境变量：
  - `UNI_AGENT_CONTAINER_RUNTIME`
  - `LOCAL_CONTAINER_RUNTIME`
- 再按顺序找：
  - `apptainer`
  - `singularity`
  - `docker`
  - `podman`
- 找不到时默认写 `apptainer`。

`local` 比 `host` 安全很多，因为：

- 命令在容器内执行。
- 默认文件系统隔离。
- 进程空间隔离。
- 依赖安装和测试执行都在容器内。
- 每个 run 可独立容器。
- stop 时 Docker/Podman 会 `rm -f` 容器。

但它不是绝对安全：

- 如果 `extra_run_args` bind mount 宿主机目录，模型能访问挂载目录。
- 如果使用 host network，网络隔离变弱。
- 如果容器 privileged 或挂载 Docker socket，安全边界会被破坏。
- 容器默认可能有网络，可访问内网/外网。
- `swerex.server` 暴露端口，虽然有 auth token，但端口暴露范围要控制。
- Apptainer/Singularity 的隔离模型和 Docker 不完全一样，HPC 环境常有默认 bind mount。

适合场景：

- 本机调试 coding agent。
- SWE-Bench/R2E/Terminal-Bench 小规模跑通。
- 需要比 host 更安全，但暂时不想用云端 sandbox。

不适合：

- 大规模高并发训练。
- 强安全隔离需求。
- 不信任镜像、需要严格网络隔离的任务。

## 5. local_attach

文件：

```text
uni_agent/deployment/local_attach/deployment.py
```

含义：

```text
用户提前启动好容器和 swerex.server。
Uni-Agent 不负责启动/停止容器，只 attach 到已有服务。
```

典型配置：

```yaml
deployment:
  type: local_attach
  host: http://127.0.0.1
  port: 18000
  auth_token: CHANGEME
```

用户先手动启动：

```bash
docker run --rm -it -p 18000:8000 python:3.12 bash -lc '
  python3 -m pip install -q swe-rex &&
  python3 -m swerex.server --host 0.0.0.0 --port 8000 --auth-token CHANGEME'
```

执行链路：

```text
用户启动容器 + swerex.server
  -> Uni-Agent LocalAttachDeployment.start()
  -> RemoteRuntime 连接 host:port
  -> create bash session
  -> tool command 在已有容器内执行
```

和 `local` 的区别：

```text
local:
  Uni-Agent 负责启动和删除容器。

local_attach:
  用户负责启动和删除容器。
  Uni-Agent 只连接。
```

适合场景：

- 用户想自己控制容器启动参数。
- 容器需要复杂初始化。
- 想保留容器状态用于调试。
- app 场景，例如 Lark chat sandbox，需要手动挂载认证或 memory 目录。

安全性取决于用户启动的容器参数。默认比 host 好，但如果挂载宿主机敏感目录或使用 privileged，也会变危险。

## 6. modal

文件：

```text
uni_agent/deployment/modal/deployment.py
```

含义：

```text
使用 Modal 作为远程 sandbox 后端。
每个任务可启动远程隔离环境。
```

配置里常见字段：

```yaml
deployment:
  type: modal
  image: ...
  startup_timeout: 600
  runtime_timeout: 300
  deployment_timeout: 3600
  modal_sandbox_kwargs:
    cpu: ...
    memory: ...
```

优点：

- 不占本机容器资源。
- 更适合并发。
- 远程隔离强于 host/local。
- task image 可以 per-sample 指定。

代价：

- 需要 Modal 账号和 token。
- 冷启动/镜像拉取有延迟。
- 网络和代理配置更复杂。
- 云端资源/额度限制。

适合：

- 大规模 SWE-Bench / Terminal-Bench 验证。
- 并发 rollout。
- 不想在本机承载大量 sandbox。

## 7. vefaas

文件：

```text
uni_agent/deployment/vefaas/deployment.py
```

含义：

```text
使用火山 veFaaS / 函数平台作为远程 sandbox 后端。
```

配置字段：

```yaml
deployment:
  type: vefaas
  image: ...
  command: python3 -m swerex.server --auth-token {token}
  function_id: ...
  function_route: ...
  timeout: ...
  startup_timeout: ...
```

适合场景和 Modal 类似：

- 高并发。
- 远程 sandbox。
- 大规模 rollout。

主要区别是云厂商和函数平台接口不同。

## 8. 安全边界对比

| deployment | 是否容器 | 谁启动环境 | 命令在哪里执行 | 隔离强度 | 典型用途 |
|---|---:|---|---|---|---|
| `host` | 否 | Uni-Agent | 宿主机 bash | 很弱 | search 等收敛工具、本地可信调试 |
| `local_native` | 否 | Uni-Agent | 宿主机 PTY bash | 很弱 | 需要真实终端/宿主机认证状态 |
| `local` | 是 | Uni-Agent | 本机容器 | 中等 | 本机 coding agent sandbox |
| `local_attach` | 是 | 用户 | 用户已有容器 | 中等，取决于启动参数 | 手动控制 sandbox |
| `modal` | 是/远程 sandbox | Modal | 云端 sandbox | 较强 | 大规模并发验证/训练 |
| `vefaas` | 是/远程 sandbox | veFaaS | 云端 sandbox | 较强 | 大规模并发验证/训练 |

简单决策：

```text
只调用 search/finish 等收敛工具:
  host 可以接受。

需要 execute_bash / 文件编辑，但任务可信、本机调试:
  local 更合适。

不可信任务、benchmark、大规模训练:
  modal / vefaas 更合适。

需要复用宿主机认证状态:
  local_native 或 local_attach，取决于是否能放进容器。
```

## 9. Tool 和 Deployment 的关系

Deployment 决定命令在哪里跑，tool 决定模型能发什么命令。

真正的安全边界是二者组合：

```text
host + search/finish:
  风险相对可控。

host + execute_bash:
  高风险，模型可任意操作宿主机。

local + execute_bash:
  常见 coding sandbox 形态，风险限制在容器内。

local + bind mount 宿主机敏感目录 + execute_bash:
  风险重新变高。

modal/vefaas + execute_bash:
  适合 benchmark，但仍要控制凭据、网络和镜像。
```

所以看配置时要同时看：

```yaml
env:
  deployment:
    type: ...

tools:
  - name: ...
```

不能只看 deployment。

## 10. 本地开发建议

如果你只是理解框架或跑 demo：

```text
search agent:
  host + search/finish OK

代码修改类 agent:
  local 或 local_attach

Lark/本机认证工具:
  local_native 或 local_attach
```

如果你要让模型执行任意 shell，建议至少使用：

```yaml
deployment:
  type: local
```

并避免：

- `--privileged`
- 挂载 `/`
- 挂载 Docker socket
- host network
- 把真实 token/secret 放进容器环境
- 把工作机 SSH key 或云凭据挂进去

如果要大规模训练或跑不可信 benchmark，优先考虑：

```text
modal / vefaas
```

## 11. 总结

`deployment` 是 Uni-Agent 的环境执行抽象。

最关键的理解：

```text
host/local_native:
  方便，但不是 sandbox。

local/local_attach:
  本机容器 sandbox，安全性明显更好。

modal/vefaas:
  远程 sandbox，适合规模化。
```

`host` 是否危险，取决于 tool；`local` 是否安全，取决于容器参数。对于 agent RL 和 benchmark，默认思路应该是让模型动作发生在 sandbox，而不是宿主机。

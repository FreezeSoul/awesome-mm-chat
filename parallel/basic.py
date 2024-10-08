import numpy as np
import math
import torch
from mmengine.runner import set_random_seed


def qk_tie():
    import torch

    q = torch.arange(20 * 5).reshape(20, 5)
    k = torch.arange(16 * 5).reshape(16, 5)
    c = torch.matmul(q, k.T)
    print(c)  # (20, 16)
    print(c.float().mean())

    # 分块计算
    q_bucket_size = 2
    k_bucket_size = 4

    q_chunks = q.split(q_bucket_size, dim=0)
    k_chunks = k.split(k_bucket_size, dim=0)
    print(len(q_chunks), len(k_chunks))

    # (20, 16)
    out = []
    for q_index, q_chunk in enumerate(q_chunks):  # 遍历 q 分块
        weights = []
        for k_index, k_chunk in enumerate(k_chunks):  # 内部遍历 k 分块
            # (2,5) * (5,4) -> (2, 4)
            weight = torch.matmul(q_chunk, k_chunk.T)
            weights.append(weight)
        # 此时就得到每个 q 相比所有 k 的值
        all_weights = torch.cat(weights, dim=-1)
        out.append(all_weights)
    # 此时就得到所有 q 相比所有 k 的值
    out = torch.cat(out, dim=0)
    print(out)  # (20, 16)
    print(out.float().mean())


def numpy_softmax():
    def softmax(x):
        # 为了数值稳定性，减去最大值
        # 虽然这种实现特别快，但是如果 logits 长度是 1b，那么这个计算会 OOM
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    # 示例输入
    logits = np.array([2.0, 1.0, 0.1, 0.8])

    # 计算 Softmax
    softmax_output = softmax(logits)
    print("Softmax 输出:", softmax_output)


def python_softmax():
    logits = [2.0, 1.0, 0.1, 0.8]

    softmax_output = []
    sum_exp = 0
    max_value = -math.inf
    # 相比于上面的实现，使用 for 计算方式有一个好处：
    # 如果 logits 超级长，或者说 logits 就无法一次性 load 到内存中
    # 那么我们可以 for 进行分段计算
    # 比如 0 cpu 先计算 0~1000，然后把 max_value 值传递给 1 cpu，1 cpu 计算 1000~2000
    # 通过这种串行方式就可以解决上述为问题。

    # 先求最大值
    for logit in logits:
        max_value = max(max_value, logit)
    # 分母减掉最大值
    for logit in logits:
        sum_exp += math.exp(logit - max_value)
    # 计算每个元素
    for logit in logits:
        softmax_output.append(np.exp(logit - max_value) / sum_exp)
    print("Softmax 输出:", softmax_output)


# https://zhuanlan.zhihu.com/p/668888063
# Safe softmax with online normalizer calculation
def python_online_softmax():
    logits = [2.0, 1.0, 0.1, 0.8]
    softmax_output = []
    pre_max_value = -math.inf
    sum_exp = 0

    # 上述做法虽然可以解决问题，但是有 3次 for 训练，也就是 logins 要 load 3 次，如果可以减少 for 次数那么就更好了
    # online 做法就可以变成 2 次，采用了类似在线方式

    # 只有两次 for 循环
    for logit in logits:
        max_value_ = max(pre_max_value, logit)
        # https://zhuanlan.zhihu.com/p/668888063 有推导，迭代形式来源
        sum_exp = sum_exp * math.exp(pre_max_value - max_value_) + math.exp(logit - max_value_)
        pre_max_value = max_value_

    # 计算每个元素
    for logit in logits:
        softmax_output.append(np.exp(logit - pre_max_value) / sum_exp)
    print("Softmax 输出:", softmax_output)


# Multi-Pass 版本 FlashAttention
def python_qkv_online_softmax():
    set_random_seed(42)
    q = torch.randn((20, 5))
    k = torch.randn((16, 5))
    v = torch.randn((16, 5))
    c = torch.matmul(q, k.T)
    weight = torch.softmax(c, dim=-1)
    value = torch.matmul(weight, v)
    print(value)
    print(value.mean())

    # 每一行是完全独立的，可以 kernel 并行计算
    o = torch.zeros(value.shape)
    for k_index in range(q.shape[0]):  # 遍历 q
        # q: 1x5 k 16x5 v 1x5
        # 1x5 * 5x16 -> 1x16(softmax 值)
        # 16x1 * 1x5(v) -> 16x5 -> sum -> 1x5
        x_s = []
        pre_max_value = -math.inf
        d_i = 0
        for i in range(k.shape[0]):  # 遍历 k
            # 1x5 * 5x1 -> 1x1 -> 1
            x_i = torch.matmul(q[k_index:k_index + 1, :], (k[i:i + 1, :]).T).squeeze()
            max_value_ = max(pre_max_value, x_i)
            d_i = d_i * math.exp(pre_max_value - max_value_) + math.exp(x_i - max_value_)
            pre_max_value = max_value_
            x_s.append(x_i)

        # 1x5
        o_i = torch.zeros((1, v.shape[1]))
        for i in range(k.shape[0]):  # 遍历 k
            a_i = math.exp(x_s[i] - pre_max_value) / d_i  # softmax 值
            o_i += a_i * v[i:i + 1, :]
        o[k_index, :] = o_i

    print(o)
    print(o.mean())
    assert torch.allclose(value, o, atol=1e-6)


# https://zhuanlan.zhihu.com/p/668888063
# flash attention v1 核心
def python_qkv_online_softmax_one_pass():
    set_random_seed(42)
    q = torch.randn((20, 5))
    k = torch.randn((16, 5))
    v = torch.randn((16, 5))
    c = torch.matmul(q, k.T)
    weight = torch.softmax(c, dim=-1)
    value = torch.matmul(weight, v)
    print(value)
    print(value.mean())

    o = torch.zeros(value.shape)
    for k_index in range(q.shape[0]):  # 遍历 q
        # q: 1x5 k 16x5 v 1x5
        pre_max_value = -math.inf
        d_i_pre = 0
        o_i = o[k_index:k_index + 1, :]
        # 现在只有 1 个 for 循环了
        for i in range(k.shape[0]):  # 遍历 k
            # 1x5 * 5x1 -> 1x1 -> 1
            x_i = torch.matmul(q[k_index:k_index + 1, :], (k[i:i + 1, :]).T).squeeze()
            max_value_ = max(pre_max_value, x_i)
            d_i = d_i_pre * math.exp(pre_max_value - max_value_) + math.exp(x_i - max_value_)
            o_i = o_i * (d_i_pre * math.exp(pre_max_value - max_value_) / d_i) + \
                math.exp(x_i - max_value_) / d_i * v[i:i + 1, :]

            pre_max_value = max_value_
            d_i_pre = d_i

        o[k_index, :] = o_i

    print(o)
    print(o.mean())
    assert torch.allclose(value, o, atol=1e-6)


# https://zhuanlan.zhihu.com/p/663932651
# https://zhuanlan.zhihu.com/p/668888063
# flash attention v1 核心 + 分块计算，提高效率
def torch_qkv_chunk_online_softmax_one_pass():
    set_random_seed(42)
    Q = torch.randn((20, 5))
    K = torch.randn((16, 5))
    V = torch.randn((16, 5))
    c = torch.matmul(Q, K.T)
    weight = torch.softmax(c, dim=-1)
    value = torch.matmul(weight, V)
    print(value)
    print(value.mean())

    # qkv 都分块计算,每块的长度是 chunk_size
    chunk_size = 4

    NEG_INF = -1e10  # -infinity
    EPSILON = 1e-10
    O = torch.zeros_like(Q, requires_grad=True)
    l = torch.zeros(Q.shape[:-1])[..., None]
    m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF

    Q_BLOCKS = torch.split(Q, chunk_size, dim=0)
    K_BLOCKS = torch.split(K, chunk_size, dim=0)
    V_BLOCKS = torch.split(V, chunk_size, dim=0)
    O_BLOCKS = list(torch.split(O, chunk_size, dim=0))
    l_BLOCKS = list(torch.split(l, chunk_size, dim=0))
    m_BLOCKS = list(torch.split(m, chunk_size, dim=0))

    Tr = Q.shape[0] // chunk_size
    Tc = K.shape[0] // chunk_size
    for j in range(Tc):  # 为何变成 K 开始遍历？
        Kj = K_BLOCKS[j]
        Vj = V_BLOCKS[j]
        for i in range(Tr):
            Qi = Q_BLOCKS[i]
            Oi = O_BLOCKS[i]
            li = l_BLOCKS[i]
            mi = m_BLOCKS[i]

            S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi, Kj)
            m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
            P_ij = torch.exp(S_ij - m_block_ij)
            l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON
            P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)

            mi_new = torch.maximum(m_block_ij, mi)

            li_new = torch.exp(mi - mi_new) * li \
                     + torch.exp(m_block_ij - mi_new) * l_block_ij

            O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi \
                          + (torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
            # print(f'-----------Attn : Q{i}xK{j}---------')
            #         print(O_BLOCKS[i].shape)
            print(O_BLOCKS[0])
            print(O_BLOCKS[1])
            print('\n')

            l_BLOCKS[i] = li_new
            m_BLOCKS[i] = mi_new

    O = torch.cat(O_BLOCKS, dim=0)
    print(O)
    print(O.mean())
    assert torch.allclose(value, O, atol=1e-6)


def python_online_softmax_parallel():
    logits = [2.0, 1.0, 0.1, 0.8]
    softmax_output = []

    def online_softmax_update(m0, d0, m1, d1):
        m = max(m0, m1)
        d = d0 * np.exp(m0 - m) + d1 * np.exp(m1 - m)
        return m, d

    # # 上述做法虽然少了一次 for，但是必须串行运行，无法真正并行
    # # 假设下面代码是并行计算，一共 2 张卡，每张卡计算一半数据

    # 假设切分为 2 段独立计算,每块可以独立算
    chunk_size = 2
    exp_weights = []
    weight_maxes = []
    for i in range(chunk_size):
        pre_max_value = -math.inf
        sum_exp = 0
        # 每段单独算
        for logit in logits[i * chunk_size: i * chunk_size + chunk_size]:
            pre_max_value, sum_exp = online_softmax_update(pre_max_value, sum_exp, logit, 1)
        exp_weights.append(sum_exp)
        weight_maxes.append(pre_max_value)

    # 然后在所有卡上面都运行这个代码，保证所有卡的结果一致
    pre_max_value = -math.inf
    sum_exp = 0
    # 逐渐合并得到最终值
    for i in range(len(exp_weights)):
        pre_max_value, sum_exp = online_softmax_update(pre_max_value, sum_exp, weight_maxes[i], exp_weights[i])

    # 计算每个元素，这个自然的并行
    for logit in logits:
        softmax_output.append(np.exp(logit - pre_max_value) / sum_exp)

    print("Softmax 输出:", softmax_output)


# 类似 memory efficient attention
# 相比于上面代码，
# 1. 引入了实际需要的 qk 分块计算
# 2. 引入了 online softmax parallel 矩阵计算(之前是 for)
def torch_online_qk_softmax_parallel():
    q = torch.randn((20, 5))
    k = torch.randn((16, 5))
    c = torch.matmul(q, k.T)
    c = torch.softmax(c, dim=-1)
    print(c)
    print(c.mean())

    # 采用矩阵分块+ online softmax
    q_bucket_size = 2
    k_bucket_size = 4

    q_chunks = q.split(q_bucket_size, dim=0)
    k_chunks = k.split(k_bucket_size, dim=0)
    print(len(q_chunks), len(k_chunks))

    # (20, 16)
    out = []
    for q_index, q_chunk in enumerate(q_chunks):  # 遍历 q 分块
        # 整个for 循环内部对应上述的一次 online_softmax_update
        exp_weights = []
        weight_maxes = []
        # 这个 for 是可以并行计算的
        # q_i 和 k_0,k_1,k_2,k_3 可以分别放到不同 kernel 上面并行算，或者放到不同卡上面算
        for k_index, k_chunk in enumerate(k_chunks):  # 内部遍历 k 分块
            # (2,5) * (5,4) -> (2, 4)
            weight = torch.matmul(q_chunk, k_chunk.T)

            weight_max = weight.amax(dim=-1, keepdim=True)
            weight = weight - weight_max  # safe softmax
            exp_weight = weight.exp()
            exp_weights.append(exp_weight)
            weight_maxes.append(weight_max.repeat(1, weight.shape[-1]))

        # 这种做法的弊端是： 会占用大量显存，因为 exp_weights 的维度是 (q_chunk, k), k 是整个序列都在
        # 好在实际上我们并不实际上需要 exp_weights，只需要最终的 v 输出，因此有后面的做法
        # 合并当前结果
        # 此时就得到每个 q 相比所有 k 的值
        weight_maxes = torch.cat(weight_maxes, dim=-1)
        exp_weights = torch.cat(exp_weights, dim=-1)

        global_max = weight_maxes.amax(dim=-1, keepdim=True)
        renorm_factor = (weight_maxes - global_max).exp()
        # 矩阵算法
        exp_weights = exp_weights * renorm_factor

        exp_weights = exp_weights / exp_weights.sum(dim=-1, keepdim=True)
        out.append(exp_weights)
    # 此时就得到所有 q 相比所有 k 的值
    out = torch.cat(out, dim=0)
    print(out)  # (20, 16)
    print(out.float().mean())
    assert torch.allclose(c, out)


# https://zhuanlan.zhihu.com/p/668888063
# https://github.com/lucidrains/memory-efficient-attention-pytorch
# 在加入 v 后算法就有比较大的不同，因为我们要的是最终的 v 而不是之前的 softmax 输出
# 所以实现上是不同的
def torch_online_qkv_attention_parallel():
    q = torch.randn((20, 5))
    k = torch.randn((16, 5))
    v = torch.randn((16, 5))
    c = torch.matmul(q, k.T)
    weight = torch.softmax(c, dim=-1)
    value = torch.matmul(weight, v)
    print(value)
    print(value.mean())

    # 采用矩阵分块+ online softmax
    q_bucket_size = 2
    k_bucket_size = 4

    q_chunks = q.split(q_bucket_size, dim=0)
    k_chunks = k.split(k_bucket_size, dim=0)
    v_chunks = v.split(k_bucket_size, dim=0)
    print(len(q_chunks), len(k_chunks), len(v_chunks))

    # (20, 16)
    out = []
    for q_index, q_chunk in enumerate(q_chunks):  # 遍历 q 分块
        # 整个for 循环内部对应上述的一次 online_softmax_update
        exp_weights = []
        weight_maxes = []
        _values = []
        # 这个 for 是可以并行计算的
        # q_i 和 k_0,k_1,k_2,k_3 可以分别放到不同 kernel 上面并行算，或者放到不同卡上面算
        for k_index, (k_chunk, v_chunk) in enumerate(zip(k_chunks, v_chunks)):  # 内部遍历 k 分块
            # (2,5) * (5,4) -> (2, 4)
            weight = torch.matmul(q_chunk, k_chunk.T)

            weight_max = weight.amax(dim=-1, keepdim=True)
            weight = weight - weight_max  # safe softmax
            exp_weight = weight.exp()
            _value = torch.matmul(exp_weight, v_chunk)
            exp_weights.append(exp_weight.sum(dim=-1))  # 注意这里
            _values.append(_value)
            weight_maxes.append(weight_max.squeeze(dim=-1))

        # 合并当前结果
        # 此时就得到每个 q 相比所有 k 的值
        weight_maxes = torch.stack(weight_maxes, dim=-1)
        _values = torch.stack(_values, dim=-1)
        exp_weights = torch.stack(exp_weights, dim=-1)

        global_max = weight_maxes.amax(dim=-1, keepdim=True)
        renorm_factor = (weight_maxes - global_max).exp()

        exp_weights = exp_weights * renorm_factor
        _values = _values * renorm_factor.unsqueeze(dim=-2)

        all_values = _values.sum(dim=-1)
        all_weights = exp_weights.sum(dim=-1)

        normalized_values = all_values / (all_weights[:, None] + 1e-8)
        out.append(normalized_values)
    out = torch.cat(out, dim=0)
    print(out)  # (20, 5)
    print(out.float().mean())
    assert torch.allclose(value, out, atol=1e-6)


if __name__ == '__main__':
    # qk_tie()
    # numpy_softmax()
    # python_softmax()
    # python_online_softmax()
    # python_qkv_online_softmax()
    python_qkv_online_softmax_one_pass()
    torch_qkv_chunk_online_softmax_one_pass()
    # python_online_softmax_parallel()
    # torch_online_qk_softmax_parallel()
    # torch_online_qkv_attention_parallel()

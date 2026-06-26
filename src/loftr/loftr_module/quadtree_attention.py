import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from QuadtreeAttention.functions.quadtree_attention import (
    score_computation_op,
    value_aggregation_op,
)
from timm.models.layers import DropPath, trunc_normal_
from torch.autograd import Function


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dwconv(x, H, W)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class QTAttA(nn.Module):
    def __init__(
        self,
        nhead,
        dim,
        topks=[32, 32, 32, 32],
        scale=None,
        use_dropout=False,
        attention_dropout=0.1,
    ):
        super().__init__()
        self.use_dropout = use_dropout
        self.topks = topks
        self.nhead = nhead
        self.dim = dim

    def process_coarse_level(self, query, key, value, topk):
        bs, c, h, w = key.shape
        cur_dim = key.shape[1] // self.nhead

        key = rearrange(key, "b c h w -> b (h w) c").view(
            bs, -1, self.nhead, cur_dim
        )  # [N, S, H, D]
        value = rearrange(value, "b c h w -> b (h w) c").view(
            bs, -1, self.nhead, cur_dim
        )  # [N, S, H, D]
        query = rearrange(query, "b c h w -> b (h w) c").view(
            bs, -1, self.nhead, cur_dim
        )

        QK = torch.einsum("nlhd,nshd->nlsh", query, key)
        softmax_temp = 1.0 / cur_dim**0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=-2)

        # mask out top K tokens
        topk_score, topk_idx = torch.topk(A, dim=-2, k=topk, largest=True)
        mask = torch.ones_like(A)
        mask = mask.scatter(
            dim=-2, index=topk_idx, src=torch.zeros_like(topk_idx).float()
        )

        # message is only computed within the unmasked
        message = torch.einsum(
            "nlsh,nshd->nlhd", A * mask, value
        )  # .reshape(bs, h, w, self.nhead, cur_dim)

        return A, message, topk_score, topk_idx

    def process_fine_level(
        self, query, key, value, topk_score, topk_pos, topk_prev, topk, final=False
    ):
        bs, c, h, w = key.shape

        cur_dim = key.shape[1] // self.nhead
        key = rearrange(key, "b c h w -> b (h w) c").view(
            bs, -1, self.nhead, cur_dim
        )  # [N, S, H, D]
        value = rearrange(value, "b c h w -> b (h w) c").view(
            bs, -1, self.nhead, cur_dim
        )  # [N, S, H, D]

        query = query.view(bs, c, h // 2, 2, w // 2, 2)
        query = rearrange(query, "b c h t1 w t2-> b (h w) (t1 t2) c ").view(
            bs, -1, 4, self.nhead, cur_dim
        )

        # convert 2d coordinates to 1d index
        idx_gather = []
        topk_pos = topk_pos * 2
        for x in [0, 1]:
            for y in [0, 1]:
                idx = (topk_pos[0] + x) * w + topk_pos[1] + y  # convert to index
                idx_gather.append(idx)

        idx = torch.stack(idx_gather, dim=3)  # [N, L, K, 4, H, D]

        # Compute score
        # query: [b, N, 4, H, D]
        # key: [b, 4N, H, D]
        # idx: [b, N, K, 4, H]
        # QK: [b, N, 4, 4K, H]
        QK = score_computation_op(
            query, key.contiguous(), idx.view(bs, -1, topk_prev * 4, self.nhead)
        )
        QK = rearrange(QK, "n l w (k f) h -> n l w k f h", k=topk_prev, f=4)
        softmax_temp = 1.0 / cur_dim**0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=-2)  # [N, L//scale**i, K, 4, H]
        # Score redistribution
        topk_score = topk_score.unsqueeze(-2).unsqueeze(2)
        A = (A * topk_score).reshape(bs, -1, 4, topk_prev * 4, self.nhead)
        idx = idx.view(bs, -1, 1, topk_prev * 4, self.nhead).repeat(
            1, 1, 4, 1, 1
        )  # [N, L,4, K*4, H]
        topk_score, topk_idx = torch.topk(A, dim=-2, k=topk, largest=True)

        if not final:
            mask = torch.ones_like(A)
            mask = mask.scatter(
                dim=-2, index=topk_idx, src=torch.zeros_like(topk_idx).float()
            )
            message = value_aggregation_op(A * mask, value.contiguous(), idx)
        else:
            message = value_aggregation_op(A, value.contiguous(), idx)

        if not final:
            topk_idx = torch.gather(idx, index=topk_idx, dim=-2)
            topk_idx = rearrange(
                topk_idx, "b (h w) (t1 t2) k nh -> b (h t1 w t2) k nh", h=h // 2, t1=2
            )  # reshape back
            topk_score = rearrange(
                topk_score, "b (h w) (t1 t2) k nh -> b (h t1 w t2) k nh", h=h // 2, t1=2
            )  # reshape back

        return A, message, topk_score, topk_idx

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head quadtree attention
        Args:
            queries: Query pyramid [N, C, H, W]
            keys: Key pyramid [N, C, H, W]
            values: Value pyramid [N, C, H, W]
        Returns:
            message: (N, C, H, W)
        """

        bs = queries[0].shape[0]
        messages = []
        topk = self.topks[0]

        for i, (query, key, value) in enumerate(
            zip(reversed(queries), reversed(keys), reversed(values))
        ):
            bs, c, h, w = key.shape
            if i == 0:
                A, message, topk_score, topk_idx = self.process_coarse_level(
                    query, key, value, topk
                )  # Full attention for coarest level
            else:
                topk_prev = topk
                topk = self.topks[i]
                final = True if i == len(queries) - 1 else False
                A, message, topk_score, topk_idx = self.process_fine_level(
                    query, key, value, topk_score, topk_pos, topk_prev, topk, final
                )  # Quadtree attention

            messages.append(message)
            if topk_idx is not None:
                topk_pos = torch.stack(
                    [topk_idx // w, topk_idx % w]
                )  # convert to coordinate

        final_message = 0
        for i, m in enumerate(messages):
            if i == 0:
                final_message = m
            else:
                final_message = final_message.unsqueeze(2) + m
                final_message = rearrange(
                    final_message,
                    "b (H W) (t1 t2) h d -> b (H t1 W t2) h d",
                    t1=2,
                    t2=2,
                    H=queries[-i].shape[2],
                )

        return final_message


class QTAttB(nn.Module):
    def __init__(
        self,
        nhead,
        dim,
        scale,
        topks=[32, 32, 32, 32],
        use_dropout=False,
        attention_dropout=0.1,
        lepe=False,
    ):
        super().__init__()
        self.use_dropout = use_dropout
        self.topks = topks
        self.nhead = nhead
        self.dim = dim
        self.lepe = lepe
        if lepe:  # locally enhanced position encoding
            self.get_vs = nn.ModuleList(
                [
                    nn.Conv2d(
                        dim * nhead,
                        dim * nhead,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=dim * nhead,
                    )
                    for _ in range(scale)
                ]
            )
        self.register_parameter("weight", nn.Parameter(torch.randn(scale)))

    def process_coarse_level(self, query, key, value, topk):
        bs, c, h, w = key.shape

        cur_dim = key.shape[1] // self.nhead
        key = rearrange(key, "b c h w -> b (h w) c").view(
            bs, -1, self.nhead, cur_dim
        )  # [N, S, H, D]
        value = rearrange(value, "b c h w -> b (h w) c").view(
            bs, -1, self.nhead, cur_dim
        )  # [N, S, H, D]
        query = rearrange(query, "b c h w -> b (h w) c").view(
            bs, -1, self.nhead, cur_dim
        )
        QK = torch.einsum("nlhd,nshd->nlsh", query, key)
        softmax_temp = 1.0 / cur_dim**0.5  # sqrt(D)

        A = torch.softmax(softmax_temp * QK, dim=-2)
        topk_score, topk_idx = torch.topk(A, dim=-2, k=topk, largest=True)

        message = torch.einsum(
            "nlsh,nshd->nlhd", A, value
        )  # .reshape(bs, h, w, self.nhead, cur_dim)

        return A, message, topk_score, topk_idx

    def process_fine_level(
        self, query, key, value, topk_score, topk_pos, topk_prev, topk, final=False
    ):
        bs, c, h, w = key.shape

        cur_dim = key.shape[1] // self.nhead
        key = rearrange(key, "b c h w -> b (h w) c").view(
            bs, -1, self.nhead, cur_dim
        )  # [N, S, H, D]
        value = rearrange(value, "b c h w -> b (h w) c").view(
            bs, -1, self.nhead, cur_dim
        )  # [N, S, H, D]

        query = query.view(bs, c, h // 2, 2, w // 2, 2)
        query = rearrange(query, "b c h t1 w t2-> b (h w) (t1 t2) c ").view(
            bs, -1, 4, self.nhead, cur_dim
        )

        # convert 2D coordiantes to 1D index
        topk_pos = topk_pos * 2
        idx_gather = []
        for x in [0, 1]:
            for y in [0, 1]:
                idx = (topk_pos[0] + x) * w + topk_pos[1] + y  # convert to index
                idx_gather.append(idx)
        idx = torch.stack(idx_gather, dim=3)  # [N, L, K, 4, H, D]

        # score computation
        # query: [b, N, 4, H, D]
        # key: [b, 4N, H, D]
        # idx: [b, N, K, 4, H]
        # QK: [b, N, 4, 4K, H]
        QK = score_computation_op(
            query, key.contiguous(), idx.view(bs, -1, topk_prev * 4, self.nhead)
        )
        softmax_temp = 1.0 / cur_dim**0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=-2)  # [N, L//scale**i, K, 4, H]
        A = A.reshape(bs, -1, 4, topk_prev * 4, self.nhead)
        idx = idx.view(bs, -1, 1, topk_prev * 4, self.nhead).repeat(
            1, 1, 4, 1, 1
        )  # [N, L,4, K*4, H]

        topk_score, topk_idx = torch.topk(A, dim=-2, k=topk, largest=True)
        message = value_aggregation_op(A, value.contiguous(), idx)
        topk_idx = torch.gather(idx, index=topk_idx, dim=-2)
        topk_idx = rearrange(
            topk_idx, "b (h w) (t1 t2) k nh -> b (h t1 w t2) k nh", h=h // 2, t1=2
        )  # reshape back
        topk_score = rearrange(
            topk_score, "b (h w) (t1 t2) k nh -> b (h t1 w t2) k nh", h=h // 2, t1=2
        )  # reshape back

        return A, message, topk_score, topk_idx

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head quadtree attention
        Args:
            queries: Query pyramid [N, C, H, W]
            keys: Key pyramid [N, C, H, W]
            values: Value pyramid [N, C, H, W]
        Returns:
            message: (N, C, H, W)
        """

        bs = queries[0].shape[0]

        messages = []
        topk = self.topks[0]
        for i, (query, key, value) in enumerate(
            zip(reversed(queries), reversed(keys), reversed(values))
        ):
            bs, c, h, w = key.shape
            if i == 0:  # Full attention for the coarest level
                A, message, topk_score, topk_idx = self.process_coarse_level(
                    query, key, value, topk
                )
            else:
                topk_prev = topk
                topk = self.topks[i]
                final = True if i == len(queries) - 1 else False
                A, message, topk_score, topk_idx = self.process_fine_level(
                    query, key, value, topk_score, topk_pos, topk_prev, topk, final
                )

            messages.append(message)
            topk_pos = torch.stack(
                [topk_idx // w, topk_idx % w]
            )  # convert to coordinate

        # Merge messages of different layers
        final_message = 0

        weight = torch.softmax(self.weight, dim=0)
        for i, m in enumerate(messages):
            if self.lepe:
                H, W = values[-(i + 1)].shape[-2:]
                lepe = self.get_vs[i](values[-(i + 1)])

            if i == 0:
                if self.lepe:
                    lepe = rearrange(
                        lepe, "b (hd d) H W -> b (H W) hd d", hd=self.nhead
                    )
                    final_message = (m + lepe) * weight[i]
                else:
                    final_message = m * weight[i]
            else:
                if self.lepe:
                    lepe = rearrange(
                        lepe,
                        "b (hd d) (H t1) (W t2) -> b (H W) (t1 t2) hd d",
                        hd=self.nhead,
                        t1=2,
                        t2=2,
                    )
                    final_message = final_message.unsqueeze(2) + (m + lepe) * weight[i]
                else:
                    final_message = final_message.unsqueeze(2) + m * weight[i]

                final_message = rearrange(
                    final_message,
                    "b (H W) (t1 t2) h d -> b (H t1 W t2) h d",
                    t1=2,
                    t2=2,
                    H=queries[-i].shape[2],
                )
        return final_message


class QuadtreeAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        topks,
        value_branch=False,
        act=nn.GELU(),
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        scale=1,
        attn_type="B",
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        if attn_type == "A":
            self.py_att = QTAttA(num_heads, dim // num_heads, scale=scale, topks=topks)
        else:
            self.py_att = QTAttB(num_heads, dim // num_heads, scale=scale, topks=topks)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.scale = scale

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            trunc_normal_(m.weight, std=0.02)
            m.init = True
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, target, H, W, msg=None):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        target = target.permute(0, 2, 1).reshape(B, C, H, W)
        keys = []
        values = []
        queries = []

        q = self.q_proj(x)
        k = self.k_proj(target)
        v = self.v_proj(target)
        for i in range(self.scale):
            keys.append(k)
            values.append(v)
            queries.append(q)

            if i != self.scale - 1:
                k = F.avg_pool2d(k, kernel_size=2, stride=2)
                q = F.avg_pool2d(q, kernel_size=2, stride=2)
                v = F.avg_pool2d(v, kernel_size=2, stride=2)

        msg = self.py_att(queries, keys, values).view(B, -1, C)

        x = self.proj(msg)
        x = self.proj_drop(x)

        return x


class QuadtreeBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        topks,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        scale=1,
        attn_type="B",
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = QuadtreeAttention(
            dim,
            num_heads=num_heads,
            topks=topks,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            scale=scale,
            attn_type=attn_type,
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if hasattr(m, "init"):
            return
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, target, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(target), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

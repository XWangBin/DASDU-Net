import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        # x: [B, C, H, W] -> [B, H, W, C] -> Norm -> [B, C, H, W]
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# --------------------------------------------------------------------------------
# 注意力与前馈网络模块
# --------------------------------------------------------------------------------

class MA2(nn.Module):
    def __init__(self, dim, heads, dim_head, is_cross=False):
        super().__init__()
        self.num_heads = heads
        self.scale = nn.Parameter(torch.ones(heads, 1, 1))
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.is_cross = is_cross
        if is_cross:
            self.to_ql = nn.Linear(dim, inner_dim, bias=False)
            self.to_kl = nn.Linear(dim, inner_dim, bias=False)
            self.scale_l = nn.Parameter(torch.ones(heads, 1, 1))

        self.proj = nn.Linear(inner_dim, dim)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
        )

    def forward(self, x, xl=None):
        b, h, w, c = x.shape
        x_flat = x.reshape(b, h * w, c)

        q = rearrange(self.to_q(x_flat), 'b n (h d) -> b h d n', h=self.num_heads)
        k = rearrange(self.to_k(x_flat), 'b n (h d) -> b h d n', h=self.num_heads)
        v = rearrange(self.to_v(x_flat), 'b n (h d) -> b h d n', h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (k @ q.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if self.is_cross and xl is not None:
            xl_flat = xl.reshape(b, h * w, c)
            ql = F.normalize(rearrange(self.to_ql(xl_flat), 'b n (h d) -> b h d n', h=self.num_heads), dim=-1)
            kl = F.normalize(rearrange(self.to_kl(xl_flat), 'b n (h d) -> b h d n', h=self.num_heads), dim=-1)
            attn_l = (kl @ ql.transpose(-2, -1)) * self.scale_l
            attn_l = attn_l.softmax(dim=-1)
            v = attn_l @ v

        out = attn @ v
        out = rearrange(out, 'b h d (ph pw) -> b ph pw (h d)', ph=h, pw=w)

        out_c = self.proj(out)
        out_p = self.pos_emb(rearrange(x_flat, 'b (h w) c -> b c h w', h=h, w=w)).permute(0, 2, 3, 1)
        return out_c + out_p


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, groups=dim * mult, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, bias=False),
        )

    def forward(self, x):
        # x: [B, H, W, C]
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


# --------------------------------------------------------------------------------
# 骨干网络单元 (DAU)
# --------------------------------------------------------------------------------

class MA2B(nn.Module):
    def __init__(self, dim, heads, dim_head, is_cross=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MA2(dim, heads, dim_head, is_cross)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x, xl=None):
        x = x + self.attn(self.norm1(x), xl)
        x = x + self.ff(self.norm2(x))
        return x


class DAU_Unit(nn.Module):
    def __init__(self, in_dim=31, dim=31, stage=2, num_blocks=[1, 1, 1], is_cross=False):
        super().__init__()
        self.stage = stage
        self.embedding = nn.Conv2d(in_dim, dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        curr_dim = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                nn.Sequential(
                    *[MA2B(curr_dim, curr_dim // dim, dim, is_cross) for _ in range(num_blocks[i])]),
                nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1, bias=False)
            ]))
            curr_dim *= 2

        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[MA2B(curr_dim, curr_dim // dim, dim, is_cross) for _ in range(num_blocks[-1])])

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, 2, 2),
                nn.Conv2d(curr_dim, curr_dim // 2, 1, bias=False),
                nn.Sequential(*[MA2B(curr_dim // 2, (curr_dim // 2) // dim, dim, is_cross) for _ in
                                range(num_blocks[stage - 1 - i])])
            ]))
            curr_dim //= 2

        self.mapping = nn.Conv2d(dim, in_dim, 3, 1, 1, bias=False)

    def forward(self, x, xl_list=None):
        fea = self.embedding(x)
        fea_enc = []
        new_xl = []

        # Encoder
        for i, (blocks, down) in enumerate(self.encoder_layers):
            fea_permuted = fea.permute(0, 2, 3, 1)
            xl_item = xl_list[i] if xl_list else None
            for block in blocks:
                fea_permuted = block(fea_permuted, xl_item)
            fea = fea_permuted.permute(0, 3, 1, 2)

            fea_enc.append(fea)
            new_xl.append(fea)
            fea = down(fea)

        # Bottleneck
        fea_permuted = fea.permute(0, 2, 3, 1)
        xl_bot = xl_list[self.stage] if xl_list else None
        for block in self.bottleneck:
            fea_permuted = block(fea_permuted, xl_bot)
        fea = fea_permuted.permute(0, 3, 1, 2)
        new_xl.append(fea)

        # Decoder
        for i, (up, fusion, blocks) in enumerate(self.decoder_layers):
            fea = up(fea)
            fea = fusion(torch.cat([fea, fea_enc[self.stage - 1 - i]], dim=1))
            fea_permuted = fea.permute(0, 2, 3, 1)
            xl_dec = xl_list[self.stage + 1 + i] if xl_list else None
            for block in blocks:
                fea_permuted = block(fea_permuted, xl_dec)
            fea = fea_permuted.permute(0, 3, 1, 2)

            new_xl.append(fea)

        return self.mapping(fea) + x, new_xl


# --------------------------------------------------------------------------------
# 物理退化与重建模型模块
# --------------------------------------------------------------------------------

class SpeDown(nn.Module):
    """光谱降维模块 (模拟 RGB 响应)"""
    def __init__(self, out_channels=3):
        super().__init__()
        self.spedown = nn.Conv2d(31, out_channels, 1, bias=False)
        nn.init.uniform_(self.spedown.weight, 0, 1)

    def forward(self, x):
        return self.spedown(x)


class ModelBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.sped = SpeDown()

    def forward(self, x, x0):
        # 计算测量残差: y - Φx
        return self.sped(x) - x0


class ReconstructionSup(nn.Module):
    def __init__(self, band=31):
        super().__init__()
        self.rt = nn.Sequential(
            nn.Conv2d(3, band, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(band, band, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, residual):
        return x + self.rt(residual)


# --------------------------------------------------------------------------------
# DASDU-Net 主架构
# --------------------------------------------------------------------------------

class DASDUNet(nn.Module):
    def __init__(self, in_c=3, out_c=31, n_feat=31):
        super().__init__()
        self.conv_in = nn.Conv2d(in_c, n_feat, 3, 1, 1, bias=False)

        self.dau1 = DAU_Unit(dim=n_feat, is_cross=False)
        self.dau2 = DAU_Unit(dim=n_feat, is_cross=True)
        self.dau3 = DAU_Unit(dim=n_feat, is_cross=True)

        self.mb = ModelBlock()
        self.sup = nn.ModuleList([ReconstructionSup(n_feat) for _ in range(3)])

        self.conv_out = nn.Conv2d(n_feat, out_c, 3, 1, 1, bias=False)

    def forward(self, x):
        # 1. 自动 Padding
        b, c, h, w = x.shape
        factor = 8
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 2. 初始特征提取
        fea = self.conv_in(x_pad)

        # Stage 1
        s1, xl = self.dau1(fea)
        v1 = self.sup[0](s1, self.mb(s1, x_pad))

        # Stage 2
        s2, xl = self.dau2(v1, xl)
        v2 = self.sup[1](s2, self.mb(s2, x_pad))

        # Stage 3
        s3, _ = self.dau3(v2, xl)
        v3 = self.sup[2](s3, self.mb(s3, x_pad))

        # 3. 输出
        out = self.conv_out(v3)
        return out[:, :, :h, :w]


# --------------------------------------------------------------------------------
# 测试脚本
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cpu"
    model = DASDUNet().to(device)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total Parameters: {params:.2f} M")

    dummy_input = torch.randn(1, 3, 128, 128).to(device)
    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output = model(dummy_input)
        end.record()

        torch.cuda.synchronize()
        print(f"Output shape: {output.shape}")
        print(f"Inference time: {start.elapsed_time(end):.2f} ms")
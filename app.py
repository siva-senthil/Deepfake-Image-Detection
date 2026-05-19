import io
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import streamlit as st

from PIL import Image
from torchvision import transforms
from functools import partial
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# ── CSWin Transformer architecture ────────────────────────────────────────────

def img2windows(img, H_sp, W_sp):
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    return img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    return img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None,
                 num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()
        self.dim        = dim
        self.dim_out    = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            raise ValueError(f"Invalid idx {idx}")
        self.H_sp    = H_sp
        self.W_sp    = W_sp
        self.get_v   = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        return x.reshape(-1, self.H_sp * self.W_sp, self.num_heads,
                         C // self.num_heads).permute(0, 2, 1, 3).contiguous()

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)
        lepe = func(x)
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads,
                            H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        x = x.reshape(-1, self.num_heads, C // self.num_heads,
                      self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W
        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)
        attn = nn.functional.softmax((q * self.scale) @ k.transpose(-2, -1),
                                     dim=-1, dtype=q.dtype)
        x = (self.attn_drop(attn) @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)
        return windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)


class CSWinBlock(nn.Module):
    def __init__(self, dim, reso, num_heads, split_size=7, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim               = dim
        self.num_heads         = num_heads
        self.patches_resolution = reso
        self.split_size        = split_size
        self.mlp_ratio         = mlp_ratio
        self.qkv               = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1             = norm_layer(dim)
        if reso == split_size:
            last_stage = True
        self.branch_num = 1 if last_stage else 2
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        idx_list = [-1] if last_stage else [0, 1]
        head_div = 1 if last_stage else 2
        dim_div  = 1 if last_stage else 2
        self.attns = nn.ModuleList([
            LePEAttention(dim // dim_div, resolution=reso, idx=i,
                          split_size=split_size, num_heads=num_heads // head_div,
                          dim_out=dim // dim_div, qk_scale=qk_scale,
                          attn_drop=attn_drop, proj_drop=drop)
            for i in idx_list
        ])
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp       = Mlp(in_features=dim,
                             hidden_features=int(dim * mlp_ratio),
                             out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2     = norm_layer(dim)

    def forward(self, x):
        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W
        img  = self.norm1(x)
        qkv  = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        if self.branch_num == 2:
            attened_x = torch.cat([self.attns[0](qkv[:, :, :, :C // 2]),
                                   self.attns[1](qkv[:, :, :, C // 2:])], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        x = x + self.drop_path(self.proj(attened_x))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        return self.norm(x.view(B, C, -1).transpose(-2, -1).contiguous())


class CSWinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=96, depth=[2, 2, 6, 2], split_size=[3, 5, 7],
                 num_heads=[6, 12, 12, 24], mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, use_chk=False):
        super().__init__()
        self.use_chk       = use_chk
        self.num_classes   = num_classes
        self.num_features  = self.embed_dim = embed_dim
        heads              = num_heads
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 7, 4, 2),
            Rearrange('b c h w -> b (h w) c', h=img_size // 4, w=img_size // 4),
            nn.LayerNorm(embed_dim)
        )
        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]
        self.stage1 = nn.ModuleList([
            CSWinBlock(dim=curr_dim, num_heads=heads[0], reso=img_size // 4,
                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                       split_size=split_size[0], drop=drop_rate,
                       attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.merge1  = Merge_Block(curr_dim, curr_dim * 2); curr_dim *= 2
        self.stage2  = nn.ModuleList([
            CSWinBlock(dim=curr_dim, num_heads=heads[1], reso=img_size // 8,
                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                       split_size=split_size[1], drop=drop_rate,
                       attn_drop=attn_drop_rate,
                       drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.merge2  = Merge_Block(curr_dim, curr_dim * 2); curr_dim *= 2
        self.stage3  = nn.ModuleList([
            CSWinBlock(dim=curr_dim, num_heads=heads[2], reso=img_size // 16,
                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                       split_size=split_size[2], drop=drop_rate,
                       attn_drop=attn_drop_rate,
                       drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer)
            for i in range(depth[2])])
        self.merge3  = Merge_Block(curr_dim, curr_dim * 2); curr_dim *= 2
        self.stage4  = nn.ModuleList([
            CSWinBlock(dim=curr_dim, num_heads=heads[3], reso=img_size // 32,
                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                       split_size=split_size[-1], drop=drop_rate,
                       attn_drop=attn_drop_rate,
                       drop_path=dpr[np.sum(depth[:-1]) + i],
                       norm_layer=norm_layer, last_stage=True)
            for i in range(depth[3])])
        self.norm = norm_layer(curr_dim)
        self.head = nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.stage1_conv_embed(x)
        for blk in self.stage1:
            x = checkpoint.checkpoint(blk, x) if self.use_chk else blk(x)
        for pre, blocks in zip([self.merge1, self.merge2, self.merge3],
                               [self.stage2, self.stage3, self.stage4]):
            x = pre(x)
            for blk in blocks:
                x = checkpoint.checkpoint(blk, x) if self.use_chk else blk(x)
        return torch.mean(self.norm(x), dim=1)

    def forward(self, x):
        return self.head(self.forward_features(x))


# ── Model loader ──────────────────────────────────────────────────────────────

class CustomUnpickler(pickle.Unpickler):
    """Resolves CSWin class references when loading a pkl saved from another module."""
    _map = {
        'CSWinTransformer': CSWinTransformer,
        'CSWinBlock':       CSWinBlock,
        'LePEAttention':    LePEAttention,
        'Merge_Block':      Merge_Block,
        'Mlp':              Mlp,
    }
    def find_class(self, module, name):
        if name in self._map:
            return self._map[name]
        return super().find_class(module, name)


@st.cache_resource(show_spinner="Loading model…")
def load_model(model_path: str):
    with open(model_path, 'rb') as f:
        model = CustomUnpickler(f).load()
    model.eval()
    return model


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

CLASS_NAMES = ["Fake", "Real"]


def predict(model, img: Image.Image):
    tensor = IMAGE_TRANSFORM(img).unsqueeze(0)
    with torch.inference_mode():
        logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)
    label  = CLASS_NAMES[torch.argmax(probs, dim=1).item()]
    conf   = round(probs.max().item() * 100, 2)
    return label, conf


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Deep Fake Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Deep Fake Face Detector")
st.caption(
    "Powered by the **CSWin Transformer** · "
    "[Published in Discover Computing, Springer 2025]"
    "(https://doi.org/10.1007/s10791-025-09586-2)"
)
st.divider()

# Model weight upload (since weights are not bundled in the repo)
st.sidebar.header("⚙️ Model")
weight_file = st.sidebar.file_uploader(
    "Upload model weights (`cswinmodel.pkl`)",
    type=["pkl", "pt", "pth"]
)

if weight_file is None:
    st.info(
        "👈 Upload your trained `cswinmodel.pkl` in the sidebar to get started. "
        "Model weights are not included in this repo due to file size — "
        "train the model using the scripts in `baselines/` or contact the authors."
    )
    st.stop()

# Save uploaded weights to a temp file and load
import tempfile, os
with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
    tmp.write(weight_file.read())
    tmp_path = tmp.name

try:
    model = load_model(tmp_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()
finally:
    os.unlink(tmp_path)

st.sidebar.success("Model loaded ✅")

# Image upload
st.subheader("Upload a face image")
uploaded = st.file_uploader(
    "Supported formats: JPG, PNG, WEBP",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    with st.spinner("Analysing…"):
        label, confidence = predict(model, img)

    st.divider()

    # Result card
    if label == "Fake":
        st.error(f"### ⚠️ FAKE — {confidence}% confidence")
        st.caption("This image appears to have been synthetically generated or manipulated.")
    else:
        st.success(f"### ✅ REAL — {confidence}% confidence")
        st.caption("This image appears to be authentic.")

    # Confidence bar
    st.progress(int(confidence))
    st.caption(f"Confidence: **{confidence}%**")

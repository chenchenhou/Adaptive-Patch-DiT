import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import einops
from torch.nn.utils.rnn import pad_sequence

try:
    from timm.layers import resample_abs_pos_embed, to_2tuple
except ImportError:
    def to_2tuple(x): return (x, x) if isinstance(x, int) else x

def compute_patch_entropy_batched(images, patch_size=16, num_scales=2, bins=512, pad_value=1e6):
    batch_size, channels, H, W = images.shape
    device = images.device
    if channels == 3:
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 3, 1, 1)
        grayscale_images = (images * weights).sum(dim=1)
    else:
        grayscale_images = images[:, 0]
    
    batch_entropy_maps = {}
    patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
    
    for ps in patch_sizes:
        num_patches_h = (H + ps - 1) // ps
        num_patches_w = (W + ps - 1) // ps
        pad_h, pad_w = num_patches_h * ps - H, num_patches_w * ps - W
        padded = F.pad(grayscale_images, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        flat_patches = padded.unfold(1, ps, ps).unfold(2, ps, ps).reshape(batch_size, num_patches_h, num_patches_w, -1)
        flat_patches_int = (flat_patches * (bins / 256.0)).long().clamp(0, bins-1)
        
        one_hot = torch.zeros(flat_patches_int.numel(), bins, device=device)
        one_hot.scatter_(1, flat_patches_int.view(-1, 1), 1)
        histograms = one_hot.view(batch_size, num_patches_h, num_patches_w, -1, bins).sum(dim=3)
        
        probs = histograms.float() / (ps * ps)
        entropy_map = -torch.sum(probs * torch.log2(probs + 1e-10), dim=3)
        if pad_h > 0: entropy_map[:, -1, :] = pad_value
        if pad_w > 0: entropy_map[:, :, -1] = pad_value
        batch_entropy_maps[ps] = entropy_map
    return batch_entropy_maps

def select_patches_by_threshold(entropy_maps, thresholds):
    patch_sizes = sorted(list(entropy_maps.keys()))
    if len(patch_sizes) == 1: return {patch_sizes[0]: torch.ones_like(entropy_maps[patch_sizes[0]])}
    if len(thresholds) != len(patch_sizes) - 1: thresholds = [thresholds[0]] * (len(patch_sizes) - 1)

    masks = {patch_sizes[0]: torch.ones_like(entropy_maps[patch_sizes[0]])}
    for i in range(len(patch_sizes)-1, 0, -1):
        cur, thresh = patch_sizes[i], thresholds[i-1]
        masks[cur] = (entropy_maps[cur] < thresh).float()
        
    for i in range(len(patch_sizes)-1, 0, -1):
        cur = patch_sizes[i]
        for j in range(i):
            small = patch_sizes[j]
            scale = cur // small
            upscaled = masks[cur].repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)
            masks[small] = masks[small] * (1 - upscaled[:, :masks[small].shape[1], :masks[small].shape[2]])
    return masks

class PatchTokenizer(nn.Module):
    def __init__(self, num_scales, base_patch_size, image_size, thresholds, mean=[0.5]*3, std=[0.5]*3, method='entropy'):
        super().__init__()
        self.num_scales, self.base_patch_size, self.image_size = num_scales, base_patch_size, image_size
        self.thresholds = thresholds
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, images):
        with torch.no_grad():
            unnorm = torch.clamp((images * self.std + self.mean) * 255.0, 0, 255)
            importance_maps = compute_patch_entropy_batched(unnorm, self.base_patch_size, self.num_scales)
        
        masks = select_patches_by_threshold(importance_maps, self.thresholds)
        batch_size = images.shape[0]
        device = images.device
        
        output_dict = {"masks": masks, "output_mask": [], "seqlens": torch.zeros(batch_size, device=device)}
        temp_masks = [torch.ones((batch_size, 1), device=device) * -1] # CLS placeholder

        for idx in range(self.num_scales):
            ps = self.base_patch_size * 2**idx
            flat_mask = masks[ps].flatten(1)
            output_dict["seqlens"] += flat_mask.sum(1)
            temp_masks.append(flat_mask * (idx + 1))
            
            # Patch construction
            scale_img = F.interpolate(images, scale_factor=0.5**idx, mode="bilinear") if idx > 0 else images
            patches = einops.rearrange(scale_img, "b c (h p) (w q) -> b h w c p q", p=self.base_patch_size, q=self.base_patch_size)
            output_dict[f"resized_patches_{ps}"] = patches[masks[ps].bool()]
            output_dict[f"pos_embed_mask_{ps}"] = masks[ps].flatten(1).bool()
            
            if idx > 0:
                constituent = einops.rearrange(images, "b c (h n1 p) (w n2 q) -> b h w (n1 n2) c p q", 
                                             p=self.base_patch_size, q=self.base_patch_size, 
                                             n1=ps//self.base_patch_size, n2=ps//self.base_patch_size)
                output_dict[f"full_patches_{ps}"] = constituent[masks[ps].bool()]

        output_dict["output_mask"] = torch.cat(temp_masks, dim=1)
        output_dict["output_mask"] = output_dict["output_mask"][output_dict["output_mask"] != 0]
        output_dict["seqlens"] = output_dict["seqlens"].int().tolist()
        return output_dict

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward_patch(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class TokenizedZeroConvPatchAttn(nn.Module):
    def __init__(self, image_size=224, patch_size=16, embed_dim=768, num_scales=2):
        super().__init__()
        self.img_size, self.base_patch_size, self.embed_dim = image_size, patch_size, embed_dim
        self.patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
        self.patch_embed = PatchEmbed(image_size, patch_size, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.patch_attn = nn.Conv2d(embed_dim, embed_dim, 2, 2)
        self.base_mini_pos_embed = nn.Parameter(torch.randn(1, 4, embed_dim) * .02)
        self.zero_conv = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.zero_conv.weight); nn.init.zeros_(self.zero_conv.bias)

    def forward(self, x, base_pos_embed, input_dict):
        B = x.shape[0]
        output_mask = input_dict["output_mask"]
        expanded_outputs = torch.zeros((output_mask.shape[0], self.embed_dim), device=x.device, dtype=x.dtype)
        
        # Scale 0
        base16 = input_dict[f"resized_patches_{self.base_patch_size}"]
        posmask = input_dict[f"pos_embed_mask_{self.base_patch_size}"]
        
        # Pos Embed logic
        if base_pos_embed.shape[1] > posmask.shape[1]: 
            grid_pos = base_pos_embed[:, -posmask.shape[1]:]
            cls_pos = base_pos_embed[:, :1]
        else:
            grid_pos = base_pos_embed
            cls_pos = torch.zeros(1, 1, self.embed_dim, device=x.device)

        pos_embed = grid_pos.repeat(B, 1, 1)[posmask]
        embed16 = self.patch_embed.forward_patch(base16).squeeze(1) + pos_embed
        
        expanded_outputs[output_mask == -1] = self.cls_token + cls_pos
        expanded_outputs[output_mask == 1] = embed16

        # Larger Scales
        for idx, ps in enumerate(self.patch_sizes[1:]):
            posmask = input_dict[f"pos_embed_mask_{ps}"]
            if posmask.sum() == 0: continue
            
            # Resample Pos Embed
            new_grid = self.img_size // ps
            old_grid = self.img_size // self.base_patch_size
            resampled_pos = resample_abs_pos_embed(grid_pos, (new_grid, new_grid), (old_grid, old_grid), 0)
            pos_embed = resampled_pos.repeat(B, 1, 1)[posmask]
            
            # Embed
            base = input_dict[f"resized_patches_{ps}"]
            embed_scale = self.patch_embed.forward_patch(base).squeeze(1)
            
            # Attention
            full = input_dict[f"full_patches_{ps}"]
            n = ps // self.base_patch_size
            full = self.patch_embed.forward_patch(full.view(-1, 3, self.base_patch_size, self.base_patch_size)).squeeze(1)
            full = full.view(-1, n, n, self.embed_dim)
            
            mini_pos = resample_abs_pos_embed(self.base_mini_pos_embed, (n, n), (2, 2), 0)
            full = (full + mini_pos).permute(0, 3, 1, 2)
            
            for _ in range(idx + 1): full = self.patch_attn(full)
            attn_res = full.flatten(2).mean(2)
            
            embed_final = embed_scale + pos_embed + self.zero_conv(attn_res)
            expanded_outputs[output_mask == (idx + 2)] = embed_final

        # Vectorized Padding
        seqlens = input_dict["seqlens"]
        split_tokens = torch.split(expanded_outputs, seqlens)
        padded_batch = pad_sequence(split_tokens, batch_first=True) # Fast C++ impl
        
        # Create mask
        max_len = padded_batch.shape[1]
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < torch.tensor(seqlens, device=x.device).unsqueeze(1)
        
        return padded_batch, mask, max_len, None, None
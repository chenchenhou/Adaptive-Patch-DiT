import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import einops
from typing import Dict, List, Tuple, Union, Optional, Callable

# Try importing timm helpers, provide fallbacks if necessary
try:
    from timm.layers import resample_abs_pos_embed, to_2tuple
except ImportError:
    # Minimal fallback if timm is not installed, though U-ViT requires it.
    def to_2tuple(x):
        return (x, x) if isinstance(x, int) else x

# ==========================================
# PART 1: Entropy and Metric Utilities
# ==========================================

def compute_patch_entropy_batched(images, patch_size=16, num_scales=2, bins=512, pad_value=1e6):
    """
    Compute entropy maps for multiple patch sizes in a batch of images using fully vectorized operations.
    """
    batch_size, channels, H, W = images.shape
    device = images.device
    
    # Convert batch of images to grayscale using vectorized operations
    if channels == 3:
        # Apply RGB to grayscale conversion
        grayscale_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 3, 1, 1)
        grayscale_images = (images * grayscale_weights).sum(dim=1)
    else:
        grayscale_images = images[:, 0]
    
    batch_entropy_maps = {}
    patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
    
    for ps in patch_sizes:
        num_patches_h = (H + ps - 1) // ps
        num_patches_w = (W + ps - 1) // ps
        
        pad_h = num_patches_h * ps - H
        pad_w = num_patches_w * ps - W
        padded_images = F.pad(grayscale_images, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        patches = padded_images.unfold(1, ps, ps).unfold(2, ps, ps)
        flat_patches = patches.reshape(batch_size, num_patches_h, num_patches_w, ps*ps)
        
        flat_patches_int = (flat_patches * (bins / 256.0)).long().clamp(0, bins-1)
        reshaped_patches = flat_patches_int.reshape(-1, ps*ps)
        
        # Vectorized histogram
        one_hot = torch.zeros(reshaped_patches.size(0), ps*ps, bins, device=device)
        one_hot = one_hot.scatter_(2, reshaped_patches.unsqueeze(2), 1)
        histograms = one_hot.sum(1)
        histograms = histograms.reshape(batch_size, num_patches_h, num_patches_w, bins)
        
        probabilities = histograms.float() / (ps * ps)
        epsilon = 1e-10
        entropy_map = -torch.sum(probabilities * torch.log2(probabilities + epsilon), dim=3)
        
        # Pad high entropy to edges
        if pad_h > 0:
            entropy_map[:, -1, :] = pad_value
        if pad_w > 0:
            entropy_map[:, :, -1] = pad_value
        
        batch_entropy_maps[ps] = entropy_map
    
    return batch_entropy_maps

def select_patches_by_threshold(entropy_maps, thresholds):
    """
    Vectorized version of patch selection based on entropy thresholds.
    """
    patch_sizes = sorted(list(entropy_maps.keys()))
    
    if len(patch_sizes) == 1:
        masks = {}
        masks[patch_sizes[0]] = torch.ones_like(entropy_maps[patch_sizes[0]])
        return masks
    
    if len(thresholds) != len(patch_sizes) - 1:
        # Default fallback if thresholds mismatch
        thresholds = [thresholds[0]] * (len(patch_sizes) - 1)

    masks = {}
    masks[patch_sizes[0]] = torch.ones_like(entropy_maps[patch_sizes[0]])
    
    # Process each scale from largest to smallest
    for i in range(len(patch_sizes)-1, 0, -1):
        current_size = patch_sizes[i]
        threshold = thresholds[i-1]
        masks[current_size] = (entropy_maps[current_size] < threshold).float()
        
    for i in range(len(patch_sizes)-1, 0, -1):
        current_size = patch_sizes[i]
        for j in range(i):
            smaller_size = patch_sizes[j]
            scale_factor = current_size // smaller_size 
            mask_upscaled = masks[current_size].repeat_interleave(scale_factor, dim=1).repeat_interleave(scale_factor, dim=2)
            
            H_small, W_small = entropy_maps[smaller_size].shape[1:]
            mask_upscaled = mask_upscaled[:, :H_small, :W_small]
            
            masks[smaller_size] = masks[smaller_size] * (1 - mask_upscaled)
    
    return masks

# ==========================================
# PART 2: Embedding Helpers
# ==========================================

class PatchEmbed(nn.Module):
    """ 
    2D Image to Patch Embedding with support for forward_patch (required by APT)
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size = to_2tuple(img_size)
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

    def forward_patch(self, patches):
        """
        Forward pass for pre-sliced patches (B, C, H, W)
        Used by APT to embed patches of varying sizes after they've been resized or sliced.
        """
        x = self.proj(patches)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

# ==========================================
# PART 3: Tokenizer
# ==========================================

class PatchTokenizer(nn.Module):
    """
    Tokenizer for mixed-resolution patches.
    Computes importance maps and generates masks/groups for the embedder.
    """
    def __init__(
        self,
        num_scales: int,
        base_patch_size: int,
        image_size: int,
        thresholds: List[float],
        mean: List[float] = [0.5, 0.5, 0.5],
        std: List[float] = [0.5, 0.5, 0.5],
        method: str = 'entropy',
    ):
        super().__init__()
        self.num_scales = num_scales
        self.base_patch_size = base_patch_size
        self.image_size = image_size
        self.thresholds = thresholds
        self.method = method

        # Normalization/Unnormalization for entropy calculation
        # We assume input images are normalized, so we unnormalize to [0, 255] for entropy
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))

    def unnorm(self, x):
        return x * self.std + self.mean

    def construct_masks(self, importance_maps):
        masks = select_patches_by_threshold(importance_maps, thresholds=self.thresholds)
        batch_size = masks[self.base_patch_size].shape[0]
        
        # Set up output mask
        # We include a placeholder for CLS token (-1) as APT logic expects it, 
        # even if U-ViT might separate it later.
        device = importance_maps[self.base_patch_size].device
        temp_masks = [torch.ones((batch_size, 1), device=device) * -1]
        seqlens = torch.ones((batch_size), device=device)
        
        for idx in range(0, self.num_scales):
            cur_patch_size = self.base_patch_size * 2 ** idx
            # flatten spatial dim
            temp_mask = masks[cur_patch_size].flatten(1)
            seqlens += temp_mask.sum(1)
            # Mark mask with scale index (1-based: 1, 2, 3...)
            temp_masks.append(temp_mask * (idx + 1))

        output_mask = torch.cat(temp_masks, dim=1)
        # Filter out 0s (dropped patches that were covered by larger scales)
        output_mask = output_mask[output_mask != 0]
        
        # If batch size > 1, the above flattening mixes batches. 
        seqlens = seqlens.int().tolist()
        return masks, output_mask, seqlens

    def construct_patch_groups(self, images, masks):
        output_dict = {}
        B = images.shape[0]

        for idx in range(0, self.num_scales):
            cur_patch_size = self.base_patch_size * 2 ** idx
            cur_mask = masks[cur_patch_size].bool()
            
            # 1. Full patches (downsampled from original if idx > 0)
            scale_img = images
            if idx > 0:
                scale_img = F.interpolate(scale_img, scale_factor=0.5 ** idx, mode="bilinear")
                
                # We also need the "constituent" patches (high res crops) for attention
                # Split image into patches of this scale
                constituent_patches = einops.rearrange(
                    images,
                    "b c (h n1 p3) (w n2 p4) -> b h w (n1 n2) c p3 p4",
                    h=self.image_size // cur_patch_size,
                    w=self.image_size // cur_patch_size,
                    n1=cur_patch_size // self.base_patch_size,
                    n2=cur_patch_size // self.base_patch_size,
                    p3=self.base_patch_size,
                    p4=self.base_patch_size
                )
                selected_constituent_patches = constituent_patches[cur_mask]
                output_dict[f"full_patches_{cur_patch_size}"] = selected_constituent_patches

            # 2. Resized patches (the actual token input)
            scaled_patches = einops.rearrange(
                scale_img, 
                "b c (h p1) (w p2) -> b h w c p1 p2",
                p1=self.base_patch_size,
                p2=self.base_patch_size
            )
        
            selected_patches = scaled_patches[masks[cur_patch_size].bool()]
            output_dict[f"resized_patches_{cur_patch_size}"] = selected_patches
            
            flat_mask = masks[cur_patch_size].flatten(1).bool()
            output_dict[f"pos_embed_mask_{cur_patch_size}"] = flat_mask

        return output_dict

    def forward(self, images, importance_maps=None):
        B, C, H, W = images.shape
        max_patches = B * H * W / (self.base_patch_size ** 2)
        
        if importance_maps is None:
            with torch.no_grad():
                unnorm_imgs = self.unnorm(images)
                unnorm_imgs = torch.clamp(unnorm_imgs * 255.0, 0, 255)
                importance_maps = compute_patch_entropy_batched(
                    unnorm_imgs, 
                    patch_size=self.base_patch_size, 
                    num_scales=self.num_scales
                )

        masks, output_mask, seqlens = self.construct_masks(importance_maps)
        output_dict = self.construct_patch_groups(images, masks)
        
        output_dict["output_mask"] = output_mask
        output_dict["seqlens"] = seqlens
        
        # cu_seqlens for Flash Attention
        cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=images.device),
                                torch.tensor(seqlens, dtype=torch.int32, device=images.device).cumsum(0)])
        output_dict["cu_seqlens"] = cu_seqlens
        output_dict["max_seqlen"] = max(seqlens) if seqlens else 0

        retained_patches = sum(seqlens)
        output_dict["retained_frac"] = retained_patches / max_patches if max_patches > 0 else 0

        return output_dict

# ==========================================
# PART 4: Adaptive Patch Attention/Embedder
# ==========================================

class TokenizedZeroConvPatchAttn(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        embed_dim=768,
        num_scales=2,
        thresholds=None,
    ):
        super().__init__()
        self.img_size = image_size
        self.base_patch_size = patch_size
        self.num_scales = num_scales
        self.patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
        self.embed_dim = embed_dim
        
        # Internal Patch Embedder (Standard 16x16 embedder)
        self.patch_embed = PatchEmbed(
            img_size=image_size, 
            patch_size=patch_size, 
            embed_dim=embed_dim
        )
        
        # CLS Token (Required for APT internal logic)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=.02)

        # Patch Attention for aggregating larger patches
        self.patch_attn = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=2,
            stride=2
        )
        
        # Pos embed for the mini-patches inside a larger patch
        self.base_mini_pos_embed = nn.Parameter(torch.randn(1, 4, embed_dim) * .02)

        # Zero conv for adding in attention
        self.zero_conv = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, x, base_pos_embed, input_dict):
        """
        x: Input images
        base_pos_embed: The global position embedding from the main model (1, N_patches+Extras, D)
        input_dict: The output from PatchTokenizer
        """
        batch_size = x.shape[0]
        output_mask = input_dict["output_mask"]
        
        # --- Scale 0 (Base Patch Size) ---
        base16 = input_dict[f"resized_patches_{self.base_patch_size}"]
        posmask_16 = input_dict[f"pos_embed_mask_{self.base_patch_size}"]

        # Handle Pos Embed Slicing
        # Check if base_pos_embed includes extra tokens (like CLS/Time)
        # We assume base_pos_embed is passed in appropriately sliced or matches grid
        if base_pos_embed.shape[1] > posmask_16.shape[1]:
             # Assume extras are at the start, slice them off for the grid
             # U-ViT usually puts extras at start. 
             # Calculate grid size
             grid_len = (self.img_size // self.base_patch_size)**2
             pos_embed_grid = base_pos_embed[:, -grid_len:]
             cls_token_pos_embed = base_pos_embed[:, :1] # Just take first as dummy CLS pos if needed
        else:
             pos_embed_grid = base_pos_embed
             cls_token_pos_embed = torch.zeros(1, 1, self.embed_dim, device=x.device)

        # Expand global pos embed to batch
        pos_embed16 = pos_embed_grid.repeat(batch_size, 1, 1)
        # Select pos embeds for active 16x16 patches
        pos_embed16 = pos_embed16[posmask_16]

        # Embed the base patches
        embed16 = self.patch_embed.forward_patch(base16) + pos_embed16

        # Initialize Output Container
        # The output is a flat sequence of tokens for Flash Attention
        total_tokens = output_mask.shape[0]
        expanded_outputs = torch.zeros((total_tokens, self.embed_dim), device=embed16.device, dtype=embed16.dtype)
        
        # Place CLS token
        expanded_outputs[output_mask == -1] = self.cls_token + cls_token_pos_embed
        # Place Scale 0 tokens
        expanded_outputs[output_mask == 1] = embed16

        cls_tok_loc = (output_mask == -1).nonzero().squeeze(1)

        # --- Larger Scales ---
        for scale_idx, cur_patch_size in enumerate(self.patch_sizes[1:]):
            base_patches = input_dict[f"resized_patches_{cur_patch_size}"]
            full_patches = input_dict[f"full_patches_{cur_patch_size}"]
            pos_embed_masks = input_dict[f"pos_embed_mask_{cur_patch_size}"]
            
            # Resample Global Position Embedding for this scale
            new_grid_size = self.img_size // cur_patch_size
            old_grid_size = self.img_size // self.base_patch_size
            
            resampled_pos_embed = resample_abs_pos_embed(
                pos_embed_grid,
                new_size=(new_grid_size, new_grid_size),
                old_size=(old_grid_size, old_grid_size),
                num_prefix_tokens=0,
            )
            
            pos_embed = resampled_pos_embed.repeat(batch_size, 1, 1)
            pos_embed = pos_embed[pos_embed_masks]
            
            # Resample Mini Pos Embed (for attention inside the patch)
            scale_factor = cur_patch_size // self.base_patch_size
            resampled_mini_pos_embed = resample_abs_pos_embed(
                    self.base_mini_pos_embed,
                    new_size=(scale_factor, scale_factor),
                    num_prefix_tokens=0,
                )
            
            if pos_embed_masks.sum() > 0:
                # 1. Embed as a single coarse patch (downsampled)
                embed_scale = self.patch_embed.forward_patch(base_patches)
                
                # 2. Compute Attention on fine-grained constituents
                n_patches = cur_patch_size // self.base_patch_size
                # Process full patches through embedder -> (N, n_patches^2, D)
                full_patches = full_patches.view(-1, 3, self.base_patch_size, self.base_patch_size)
                full_patches = self.patch_embed.forward_patch(full_patches)
                full_patches = full_patches.view(-1, n_patches, n_patches, self.embed_dim)
                
                # Add mini pos embed
                full_patches = full_patches + resampled_mini_pos_embed.view(n_patches, n_patches, self.embed_dim)
                full_patches = full_patches.permute(0, 3, 1, 2) # N, D, H, W
                
                # Apply convolution attention (recursively if needed for >32)
                for _ in range(scale_idx + 1):
                    full_patches = self.patch_attn(full_patches)
                
                attn_scale = full_patches.flatten(2).mean(2) # Global Average Pooling of attention result
                
                # Combine: Coarse Embed + Global Pos + Attention Residual
                embed_scale = embed_scale + pos_embed + self.zero_conv(attn_scale)
            else:
                # Handle empty case to keep graphs happy if needed
                embed_scale = torch.zeros((0, self.embed_dim), device=x.device)

            # Place in output
            # Scale index in mask is scale_idx + 2 (because 1 is base, -1 is CLS)
            expanded_outputs[output_mask == (scale_idx+2)] = embed_scale.float()

        # Reshape to (B, L, D) - Wait, FlashAttn usually takes flattened (Total_L, D)
        # But U-ViT expects (B, L, D).
        # We need to unflatten based on input_dict["seqlens"]
        
        # For simplicity in U-ViT integration, we return the flattened 'expanded_outputs' 
        # but unsqueezed to (1, Total, D) if batch size > 1 is handled via cu_seqlens,
        # OR we pad it to (B, Max_L, D). 
        #
        # U-ViT code usually iterates: `for blk in blocks: x = blk(x)`. 
        # If using FlashAttn Varlen, we return (1, Total, D) and pass cu_seqlens.
        
        # 1. Get sequence lengths per image from input_dict
        seqlens = input_dict["seqlens"] # List of ints, e.g. [120, 114, 126...]
        max_len = max(seqlens)
        
        # 2. Split the flat `expanded_outputs` back into per-image chunks
        # expanded_outputs is currently (Total_Tokens, D)
        # We split it based on the lengths
        split_tokens = torch.split(expanded_outputs, seqlens, dim=0)
        
        # 3. Pad each sequence to max_len
        padded_batch = []
        # Create a mask to tell Attention which tokens are real vs padding
        # (B, Max_Len) -> 0 for real, 1 for padding (or inverse, check U-ViT attention implementation)
        # Usually for Attention, False/0 is "do not mask" (keep), True/-inf is "mask" (remove).
        # Let's create a standard boolean mask: True = Real Token
        attn_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=x.device)
        
        for i, seq in enumerate(split_tokens):
            seq_len = seq.shape[0]
            # Pad with zeros
            padding = torch.zeros((max_len - seq_len, self.embed_dim), device=x.device, dtype=seq.dtype)
            padded_seq = torch.cat([seq, padding], dim=0)
            padded_batch.append(padded_seq)
            
            # Mark valid positions
            attn_mask[i, :seq_len] = True

        padded_batch = torch.stack(padded_batch, dim=0) # (B, Max_Len, D)
        
        # 4. Return compatible shapes
        # Return padded_batch as the main 'x'.
        # We pass 'attn_mask' so you can use it in Block/Attention if you want strict correctness.
        return padded_batch, attn_mask, max_len, cls_tok_loc, None
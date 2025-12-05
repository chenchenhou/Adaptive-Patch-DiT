import torch
import torch.nn as nn
from .timm import trunc_normal_
from .uvit import Block, timestep_embedding  # Reuse components from original uvit
from .apt_utils import PatchTokenizer, TokenizedZeroConvPatchAttn

class UViT_APT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True, 
                 # APT Specific Args
                 num_scales=2, apt_thresholds=[0.5]): # Lower threshold = more large patches
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.img_size = img_size
        self.extras = 1 # At least time token

        # --- APT Initialization ---
        # 1. The Tokenizer (calculates entropy and masks)
        self.tokenizer = PatchTokenizer(
            num_scales=num_scales,
            base_patch_size=patch_size,
            image_size=img_size,
            thresholds=apt_thresholds,
            mean=[0.5] * in_chans, std=[0.5] * in_chans,
            method='entropy'
        )

        # 2. The Adaptive Embedder
        self.apt_embed = TokenizedZeroConvPatchAttn(
            image_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_scales=num_scales
        )
        
        # 3. Multiple Output Heads for Reconstruction
        # Each scale needs its own projection back to pixels
        self.decoder_heads = nn.ModuleList()
        for i in range(num_scales):
            cur_size = patch_size * (2**i)
            # Output dim: (Patch_Size^2 * Channels)
            head = nn.Linear(embed_dim, (cur_size**2) * in_chans, bias=True)
            self.decoder_heads.append(head)

        # --- Standard U-ViT Components ---
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras += 1

        # Global Positional Embedding (Base)
        # We store enough for the finest grid. APT will resample this for larger scales.
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        
        # Final conv layer 
        self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1) if conv else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def reconstruct_image(self, x_tokens, masks):
        """
        Reconstructs the image from mixed-scale tokens.
        x_tokens: (B, Max_Len, Embed_Dim) - Padded batch of tokens
        masks: Dict[patch_size -> (B, H_grid, W_grid)] - Spatial masks from tokenizer
        """
        B, L, D = x_tokens.shape
        canvas = torch.zeros((B, self.in_chans, self.img_size, self.img_size), device=x_tokens.device)
        
        # APT Tokenizer usually processes scales in order: Smallest (Base) -> Largest
        # We must iterate in the same order to pull tokens correctly from the sequence.
        sorted_patch_sizes = sorted(masks.keys())
        
        # We iterate PER IMAGE because the padding is per-image
        for b in range(B):
            cursor = 0 # Track where we are in the token sequence for this image
            
            for i, patch_size in enumerate(sorted_patch_sizes):
                # 1. Get the spatial mask for this scale
                # Mask shape is (H_grid, W_grid)
                spatial_mask = masks[patch_size][b] 
                
                # 2. Count how many tokens exist for this scale
                num_tokens = spatial_mask.sum().int().item()
                
                if num_tokens == 0:
                    continue
                
                # 3. Slice the tokens from the sequence
                # These are the tokens for image 'b' at 'scale'
                tokens_scale = x_tokens[b, cursor : cursor + num_tokens] # (num_tokens, D)
                cursor += num_tokens
                
                # 4. Project tokens to pixels using the specific head for this scale
                decoder = self.decoder_heads[i]
                pixels = decoder(tokens_scale) # (num_tokens, patch_size*patch_size*C)
                
                # 5. Reshape to spatial patches (num_tokens, C, H, W)
                pixels = pixels.view(num_tokens, self.in_chans, patch_size, patch_size)
                
                # 6. Place on canvas
                # We find the grid coordinates (y, x) where mask is 1
                # nonzero returns (num_tokens, 2) -> [[y1, x1], [y2, x2]...]
                grid_indices = torch.nonzero(spatial_mask, as_tuple=False) 
                
                for k in range(num_tokens):
                    grid_y, grid_x = grid_indices[k]
                    
                    # Convert grid coord to pixel coord
                    pixel_y = grid_y * patch_size
                    pixel_x = grid_x * patch_size
                    
                    # Paste
                    canvas[b, :, pixel_y : pixel_y+patch_size, pixel_x : pixel_x+patch_size] = pixels[k]
                    
        return canvas

    def forward(self, x, timesteps, y=None):
        # 1. Tokenize inputs (Compute Entropy & Masks)
        # Note: If x is noisy, entropy calculation might be unstable. 
        # Ideally, pass clean 'x' if available, otherwise this computes on noisy x.
        tokenizer_out = self.tokenizer(x) 
        
        # 2. Adaptive Embedding
        # Pass only the patch part of pos_embed (remove extras like time/label)
        base_pos_embed_patches = self.pos_embed[:, self.extras:]
        
        # x_tokens is (B, Max_Len, D) - Padded
        # attn_mask is (B, Max_Len) - Bool (True for Real, False for Pad)
        x_tokens, attn_mask, _, _, _ = self.apt_embed(
            x, 
            base_pos_embed=base_pos_embed_patches, 
            input_dict=tokenizer_out
        )
        
        B, L, D = x_tokens.shape

        # 3. Add Time and Label Embeddings
        to_cat = []
        
        # Time
        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        to_cat.append(time_token)
        
        # Label
        if y is not None and self.num_classes > 0:
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            to_cat.append(label_emb)
            
        # Concatenate: [Time, Label, ... Tokens ...]
        x = torch.cat(to_cat + [x_tokens], dim=1)
        
        # Add pos embed for the extras
        x[:, :self.extras] = x[:, :self.extras] + self.pos_embed[:, :self.extras]
        
        # Update Attention Mask for extras
        # Extras are valid tokens, so prepend True to mask
        if self.extras > 0:
            extras_mask = torch.ones((B, self.extras), dtype=torch.bool, device=x.device)
            attn_mask = torch.cat([extras_mask, attn_mask], dim=1)

        # 4. Transformer Backbone
        skips = []
        for blk in self.in_blocks:
            # Note: Standard U-ViT blocks usually don't take a mask argument.
            # If your blocks support it, pass `mask=attn_mask`.
            # If not, it will attend to padding zeros.
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)

        # 5. Reconstruction
        # Strip extras
        x = x[:, self.extras:, :]
        
        # Reconstruct image using the spatial masks from tokenizer
        x = self.reconstruct_image(x, tokenizer_out["masks"])
        
        # Final conv
        x = self.final_layer(x)
        
        return x
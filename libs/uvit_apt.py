import torch
import torch.nn as nn
from .timm import trunc_normal_
from .uvit import Block, timestep_embedding
from .apt_utils import PatchTokenizer, TokenizedZeroConvPatchAttn

class UViT_APT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True, 
                 num_scales=2, apt_thresholds=[0.5]):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.img_size = img_size
        self.extras = 1 

        # --- APT Components ---
        self.tokenizer = PatchTokenizer(
            num_scales=num_scales,
            base_patch_size=patch_size,
            image_size=img_size,
            thresholds=apt_thresholds,
            mean=[0.5] * in_chans, std=[0.5] * in_chans,
            method='entropy'
        )

        self.apt_embed = TokenizedZeroConvPatchAttn(
            image_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_scales=num_scales
        )
        
        self.decoder_heads = nn.ModuleList()
        for i in range(num_scales):
            cur_size = patch_size * (2**i)
            head = nn.Linear(embed_dim, (cur_size**2) * in_chans, bias=True)
            self.decoder_heads.append(head)

        # --- U-ViT Components ---
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras += 1

        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
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
        Vectorized reconstruction of the image from tokens.
        Eliminates per-patch loops to restore training speed.
        """
        B, L, D = x_tokens.shape
        # Canvas: (B, C, H, W)
        canvas = torch.zeros((B, self.in_chans, self.img_size, self.img_size), 
                             dtype=x_tokens.dtype, device=x_tokens.device)
        
        # Track position in the sequence per image (all start at 0)
        current_offset = torch.zeros(B, dtype=torch.long, device=x_tokens.device)
        
        # Helper range for vectorized masking
        max_len = x_tokens.shape[1]
        range_tensor = torch.arange(max_len, device=x_tokens.device).unsqueeze(0) # (1, L)

        sorted_scales = sorted(masks.keys())
        
        for i, scale in enumerate(sorted_scales):
            spatial_mask = masks[scale] # (B, H_grid, W_grid)
            
            # 1. Count tokens per image for this scale
            counts = spatial_mask.reshape(B, -1).sum(dim=1).long() # (B,)
            
            # 2. Vectorized Select: Grab the next 'counts' tokens for each image
            # We want tokens where: offset <= index < offset + counts
            start = current_offset.unsqueeze(1)
            end = (current_offset + counts).unsqueeze(1)
            scale_token_mask = (range_tensor >= start) & (range_tensor < end) # (B, L)
            
            # Update offsets for next iteration
            current_offset += counts
            
            # Gather all valid tokens for this scale across the entire batch
            # Shape: (Total_Active_Tokens_In_Batch, D)
            valid_tokens = x_tokens[scale_token_mask]
            
            if valid_tokens.shape[0] == 0:
                continue
                
            # 3. Project to pixels
            decoder = self.decoder_heads[i]
            pixels = decoder(valid_tokens) # (Total_Tokens, C * s * s)
            
            # 4. Reshape to blocks
            pixels = pixels.view(-1, self.in_chans, scale, scale) # (Total_Tokens, C, s, s)
            
            # 5. Vectorized Paste
            # Reshape canvas to expose the grid: (B, C, H//s, s, W//s, s)
            H_grid, W_grid = self.img_size // scale, self.img_size // scale
            
            # View as blocks: (B, C, H_grid, s, W_grid, s)
            canvas_view = canvas.view(B, self.in_chans, H_grid, scale, W_grid, scale)
            
            # Permute to put spatial grid dimensions together: (B, H_grid, W_grid, C, s, s)
            canvas_perm = canvas_view.permute(0, 2, 4, 1, 3, 5)
            
            # Flatten spatial grid to match mask: (B, Grid_Size, ...), spatial_mask is (B, Grid_Size)
            flat_mask = spatial_mask.reshape(B, -1)
            
            # Assign all pixels at once
            canvas_perm[spatial_mask.bool()] = pixels.to(canvas.dtype)
            
        return canvas

    def forward(self, x, timesteps, y=None):
        tokenizer_out = self.tokenizer(x) 
        base_pos_embed_patches = self.pos_embed[:, self.extras:]
        
        x_tokens, attn_mask, _, _, _ = self.apt_embed(
            x, base_pos_embed=base_pos_embed_patches, input_dict=tokenizer_out
        )
        
        to_cat = []
        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        to_cat.append(time_token.unsqueeze(dim=1))
        
        if y is not None and self.num_classes > 0:
            label_emb = self.label_emb(y)
            to_cat.append(label_emb.unsqueeze(dim=1))
            
        x = torch.cat(to_cat + [x_tokens], dim=1)
        x[:, :self.extras] = x[:, :self.extras] + self.pos_embed[:, :self.extras]
        
        if self.extras > 0:
            extras_mask = torch.ones((x.shape[0], self.extras), dtype=torch.bool, device=x.device)
            attn_mask = torch.cat([extras_mask, attn_mask], dim=1)

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)
        x = self.mid_block(x)
        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        x = x[:, self.extras:, :]
        x = self.reconstruct_image(x, tokenizer_out["masks"])
        x = self.final_layer(x)
        
        return x
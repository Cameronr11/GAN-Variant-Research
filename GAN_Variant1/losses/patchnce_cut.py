"""PatchNCE loss for Contrastive Unpaired Translation (CUT)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchNCELoss(nn.Module):
    """
    PatchNCE loss from CUT paper.
    Encourages corresponding patches in input/output to be similar in feature space.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        num_patches: int = 256,
        nce_layers: list = [0, 4, 8, 12, 16]
    ):
        super().__init__()
        self.temperature = temperature
        self.num_patches = num_patches
        self.nce_layers = nce_layers
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, src_feats, tgt_feats):
        """
        Args:
            src_feats: list of feature maps from source (input) image
            tgt_feats: list of feature maps from target (output) image
        
        Returns:
            loss: PatchNCE loss
        """
        total_loss = 0.0
        
        for src_feat, tgt_feat in zip(src_feats, tgt_feats):
            loss = self._compute_nce_loss(src_feat, tgt_feat)
            total_loss += loss
        
        return total_loss / len(src_feats)
    
    def _compute_nce_loss(self, src_feat, tgt_feat):
        """
        Compute NCE loss for a single layer.
        
        Args:
            src_feat: (B, C, H, W) source features
            tgt_feat: (B, C, H, W) target features
        
        Returns:
            loss: NCE loss for this layer
        """
        B, C, H, W = src_feat.shape
        
        # Flatten spatial dimensions
        src_feat = src_feat.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        tgt_feat = tgt_feat.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        
        # Sample patches
        num_patches = min(self.num_patches, H * W)
        
        # Random sampling
        patch_ids = torch.randint(0, H * W, (num_patches,), device=src_feat.device)
        
        # Extract patches
        src_patches = []
        tgt_patches = []
        
        for b in range(B):
            src_patches.append(src_feat[b, patch_ids, :])  # (num_patches, C)
            tgt_patches.append(tgt_feat[b, patch_ids, :])  # (num_patches, C)
        
        src_patches = torch.stack(src_patches, dim=0)  # (B, num_patches, C)
        tgt_patches = torch.stack(tgt_patches, dim=0)  # (B, num_patches, C)
        
        # Normalize with epsilon for numerical stability
        src_patches = F.normalize(src_patches, dim=2, eps=1e-6)
        tgt_patches = F.normalize(tgt_patches, dim=2, eps=1e-6)
        
        # Compute similarity matrix
        # For each query (tgt_patch), compute similarity with all keys (src_patches)
        loss = 0.0
        
        for b in range(B):
            # (num_patches, C) x (C, num_patches) -> (num_patches, num_patches)
            logits = torch.mm(tgt_patches[b], src_patches[b].t()) / self.temperature
            
            # Clamp logits to prevent overflow in exp() (numerical stability)
            logits = torch.clamp(logits, min=-50.0, max=50.0)
            
            # Diagonal elements are positive pairs
            labels = torch.arange(num_patches, device=logits.device)
            
            # Cross entropy loss
            batch_loss = self.cross_entropy(logits, labels)
            
            # Safety check for NaN
            if not torch.isfinite(batch_loss):
                print(f"Warning: NaN in PatchNCE loss. Logit stats: min={logits.min().item():.2f}, max={logits.max().item():.2f}, mean={logits.mean().item():.2f}")
                batch_loss = torch.tensor(0.0, device=batch_loss.device)
            
            loss += batch_loss
        
        total_loss = loss / B
        
        # Final NaN check
        if not torch.isfinite(total_loss):
            print(f"Warning: Final PatchNCE loss is NaN. Returning 0.")
            return torch.tensor(0.0, device=total_loss.device, requires_grad=True)
        
        return total_loss


def compute_patchnce_loss(
    generator,
    src_images,
    tgt_images,
    nce_layers,
    temperature=0.07,
    num_patches=256
):
    """
    Compute PatchNCE loss between source and target images.
    
    Args:
        generator: generator model with get_feature_layers method
        src_images: (B, 3, H, W) source images
        tgt_images: (B, 3, H, W) target images (generated)
        nce_layers: list of layer indices
        temperature: NCE temperature
        num_patches: number of patches to sample
    
    Returns:
        loss: PatchNCE loss
    """
    nce_loss_fn = PatchNCELoss(temperature, num_patches, nce_layers)
    
    # Extract features
    with torch.no_grad():
        src_feats = generator.get_feature_layers(src_images, nce_layers)
    
    # Detach to prevent gradients flowing back through encoder
    src_feats = [feat.detach() for feat in src_feats]
    
    # Target features (with gradients)
    tgt_feats = generator.get_feature_layers(tgt_images, nce_layers)
    
    loss = nce_loss_fn(src_feats, tgt_feats)
    
    return loss


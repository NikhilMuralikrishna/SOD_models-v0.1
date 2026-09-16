

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BASE_FILTERS, SAR_CHANNELS, OPTICAL_CHANNELS, THERMAL_CHANNELS


# ── Building blocks ───────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Double conv block: Conv → BN → ReLU → Conv → BN → ReLU
    Standard U-Net encoder/decoder unit.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ── Encoder ───────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    U-Net encoder path — 4 levels + bottleneck.
    Returns (bottleneck, [skip1, skip2, skip3, skip4])
    where skip1 is the highest resolution (128x128) and
    skip4 is the lowest (16x16).

    Channel progression with base_filters=32:
        Input  → 64  ch  128x128  (skip1 / s1)
        64     → 128 ch  64x64    (skip2 / s2)
        128    → 256 ch  32x32    (skip3 / s3)
        256    → 512 ch  16x16    (skip4 / s4)
        512    → 512 ch  16x16    (bottleneck)
    """
    def __init__(self, in_channels, base_filters=32):
        super().__init__()
        f = base_filters
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock(in_channels, f * 2)    # → 64 ch
        self.enc2 = ConvBlock(f * 2,      f * 4)     # → 128 ch
        self.enc3 = ConvBlock(f * 4,      f * 8)     # → 256 ch
        self.enc4 = ConvBlock(f * 8,      f * 16)    # → 512 ch
        self.bot  = ConvBlock(f * 16,     f * 16)    # → 512 ch (bottleneck)

    def forward(self, x):
        s1 = self.enc1(x)                   # 256x256
        s2 = self.enc2(self.pool(s1))       # 128x128
        s3 = self.enc3(self.pool(s2))       # 64x64
        s4 = self.enc4(self.pool(s3))       # 32x32
        b  = self.bot(self.pool(s4))        # 16x16
        return b, [s1, s2, s3, s4]


# ── Fusion bottleneck ─────────────────────────────────────────

class FusionBottleneck(nn.Module):
    """
    Fuse bottleneck tensors from all three encoders.
    cat(B_sar, B_optical, B_thermal) → 1536 ch
    1x1 conv                         → 512 ch

    Always receives exactly 3 bottleneck tensors
    (missing modalities use zero tensors).
    """
    def __init__(self, single_ch, num_modalities=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(single_ch * num_modalities, single_ch,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(single_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, bottlenecks):
        """bottlenecks: list of 3 tensors (B, 512, 16, 16)"""
        # Normalise each bottleneck independently before fusion
        # This prevents scale mismatch when encoders have very different
        # activation magnitudes (e.g. overfit thermal vs stable SAR)
        normed = []
        for b in bottlenecks:
            # Per-sample, per-channel normalisation
            mean = b.mean(dim=(2, 3), keepdim=True)
            std  = b.std(dim=(2, 3), keepdim=True) + 1e-6
            normed.append((b - mean) / std)
        x = torch.cat(normed, dim=1)    # (B, 1536, 16, 16)
        return self.conv(x)             # (B, 512,  16, 16)


# ── Decoder ───────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Shared decoder — 4 upsample levels.
    At each level, concatenates the fused skip from all 3 encoders
    alongside the upsampled feature map.

    Skip fusion at each level:
        cat(s_i_sar, s_i_optical, s_i_thermal)
        Concatenated skip channels = f * 2  +  f * 4  +  f * 8  +  f * 16
        (for levels 1..4 with base_filters=32: 64 + 128 + 256 + 512)

    After each upsample, in_channels = upsample_ch + skip_ch_at_that_level
    """
    def __init__(self, base_filters=32, num_modalities=3):
        super().__init__()
        f  = base_filters
        nm = num_modalities

        # Level 4: upsample 512 → 256, skip = 512*3 = 1536
        self.up4   = nn.ConvTranspose2d(f*16, f*8, kernel_size=2, stride=2)
        self.dec4  = ConvBlock(f*8 + f*16*nm, f*8)   # 256 + 1536 = 1792

        # Level 3: upsample 256 → 128, skip = 256*3 = 768
        self.up3   = nn.ConvTranspose2d(f*8,  f*4, kernel_size=2, stride=2)
        self.dec3  = ConvBlock(f*4 + f*8*nm,  f*4)   # 128 + 768  = 896

        # Level 2: upsample 128 → 64, skip = 128*3 = 384
        self.up2   = nn.ConvTranspose2d(f*4,  f*2, kernel_size=2, stride=2)
        self.dec2  = ConvBlock(f*2 + f*4*nm,  f*2)   # 64  + 384  = 448

        # Level 1: upsample 64 → 64, skip = 64*3 = 192
        self.up1   = nn.ConvTranspose2d(f*2,  f*1, kernel_size=2, stride=2)
        self.dec1  = ConvBlock(f*1 + f*2*nm,  f*1)   # 32  + 192  = 224

        # Final output: 1 channel binary mask
        self.out   = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, bottleneck, skips_list):
        """
        Parameters
        ----------
        bottleneck : Tensor (B, 512, 16, 16)
            Fused bottleneck from FusionBottleneck.
        skips_list : list of 3 lists
            Each inner list = [s1, s2, s3, s4] from one encoder.
            s1 = (B, 64, 256, 256) ... s4 = (B, 512, 32, 32)
        """
        def norm(t):
            """Per-sample per-channel normalisation to equalise scales."""
            mean = t.mean(dim=(2, 3), keepdim=True)
            std  = t.std(dim=(2, 3), keepdim=True) + 1e-6
            return (t - mean) / std

        def fuse(level):
            """Normalise then concatenate skips from all encoders."""
            return torch.cat([norm(s[level]) for s in skips_list], dim=1)

        # Decoder level 4: 16x16 → 32x32
        x = self.up4(bottleneck)
        x = self.dec4(torch.cat([x, fuse(3)], dim=1))

        # Decoder level 3: 32x32 → 64x64
        x = self.up3(x)
        x = self.dec3(torch.cat([x, fuse(2)], dim=1))

        # Decoder level 2: 64x64 → 128x128
        x = self.up2(x)
        x = self.dec2(torch.cat([x, fuse(1)], dim=1))

        # Decoder level 1: 128x128 → 256x256
        x = self.up1(x)
        x = self.dec1(torch.cat([x, fuse(0)], dim=1))

        return torch.sigmoid(self.out(x))   # (B, 1, 256, 256)




class MultiModalUNet(nn.Module):
    """
    Multi-encoder U-Net for binary oil spill detection.

    Any modality can be None at inference — missing modalities
    are replaced with zero tensors of the correct shape so the
    decoder always receives a consistent input dimensionality.

    Usage:
        model = MultiModalUNet()

        # All three modalities
        pred = model(sar=sar_t, optical=opt_t, thermal=thm_t)

        # SAR only (optical and thermal → zeros)
        pred = model(sar=sar_t)

        # SAR + Optical
        pred = model(sar=sar_t, optical=opt_t)
    """

    def __init__(self, base_filters=BASE_FILTERS):
        super().__init__()
        f = base_filters

        self.enc_sar     = Encoder(SAR_CHANNELS,     base_filters=f)
        self.enc_optical = Encoder(OPTICAL_CHANNELS, base_filters=f)
        self.enc_thermal = Encoder(THERMAL_CHANNELS, base_filters=f)

        self.fusion  = FusionBottleneck(single_ch=f*16, num_modalities=3)
        self.decoder = Decoder(base_filters=f, num_modalities=3)

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialisation for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def _zero_bottleneck(self, ref_tensor, base_filters):
        """Create zero bottleneck tensor matching spatial dims of ref."""
        f  = base_filters
        B  = ref_tensor.shape[0]
        # Bottleneck spatial size = IMAGE_SIZE / 16
        H  = ref_tensor.shape[2] // 16
        W  = ref_tensor.shape[3] // 16
        return torch.zeros(B, f*16, H, W, device=ref_tensor.device)

    def _zero_skips(self, ref_tensor, base_filters):
        """Create zero skip tensors matching spatial dims of ref."""
        f  = base_filters
        B  = ref_tensor.shape[0]
        H  = ref_tensor.shape[2]
        W  = ref_tensor.shape[3]
        return [
            torch.zeros(B, f*2,  H,    W,    device=ref_tensor.device),
            torch.zeros(B, f*4,  H//2, W//2, device=ref_tensor.device),
            torch.zeros(B, f*8,  H//4, W//4, device=ref_tensor.device),
            torch.zeros(B, f*16, H//8, W//8, device=ref_tensor.device),
        ]

    def forward(self, sar=None, optical=None, thermal=None):
        """
        Parameters
        ----------
        sar     : Tensor | None  (B, 2, 256, 256)
        optical : Tensor | None  (B, 4, 256, 256)
        thermal : Tensor | None  (B, 2, 256, 256)

        Returns
        -------
        Tensor (B, 1, 256, 256) — oil probability in [0, 1]
        """
        # At least one modality must be provided
        ref = sar if sar is not None else (
              optical if optical is not None else thermal)
        if ref is None:
            raise ValueError(
                "At least one modality (sar, optical, thermal) must be provided."
            )

        f = BASE_FILTERS
        bottlenecks = []
        skips_list  = []

        # SAR branch
        if sar is not None:
            b, s = self.enc_sar(sar)
        else:
            b = self._zero_bottleneck(ref, f)
            s = self._zero_skips(ref, f)
        bottlenecks.append(b)
        skips_list.append(s)

        # Optical branch
        if optical is not None:
            b, s = self.enc_optical(optical)
        else:
            b = self._zero_bottleneck(ref, f)
            s = self._zero_skips(ref, f)
        bottlenecks.append(b)
        skips_list.append(s)

        # Thermal branch
        if thermal is not None:
            b, s = self.enc_thermal(thermal)
        else:
            b = self._zero_bottleneck(ref, f)
            s = self._zero_skips(ref, f)
        bottlenecks.append(b)
        skips_list.append(s)

        fused = self.fusion(bottlenecks)
        return self.decoder(fused, skips_list)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_encoder(self, modality):
        """
        Freeze one encoder branch by name.
        modality: 'sar' | 'optical' | 'thermal'
        """
        enc = getattr(self, f"enc_{modality}")
        for p in enc.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters for end-to-end fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True


# ── Loss function ─────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice loss.

    BCE handles pixel-level class separation.
    Dice handles class imbalance (oil pixels are minority).

    Parameters
    ----------
    bce_weight  : float  weight for BCE term (default 0.5)
    dice_weight : float  weight for Dice term (default 0.5)
    pos_weight  : float  BCE positive class weight to upweight oil
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=3.0):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight  = pos_weight

    def forward(self, pred, target):
        """
        pred   : (B, 1, H, W) sigmoid output in [0,1]
        target : (B, 1, H, W) binary mask  in {0,1}

        Uses F.binary_cross_entropy directly on sigmoid output
        (not logit transform) to avoid NaN from extreme predictions.
        pos_weight is applied manually per-pixel.
        """
        # Clamp predictions to avoid log(0)
        pred_safe = pred.clamp(1e-7, 1.0 - 1e-7)

        # Weighted BCE: manually apply pos_weight to positive pixels
        bce = -(
            self.pos_weight * target * torch.log(pred_safe) +
            (1.0 - target) * torch.log(1.0 - pred_safe)
        ).mean()

        # Dice loss — stable formulation
        smooth = 1.0
        inter  = (pred_safe * target).sum(dim=(1, 2, 3))
        dice   = 1.0 - (2.0 * inter + smooth) / (
            pred_safe.sum(dim=(1, 2, 3)) +
            target.sum(dim=(1, 2, 3)) + smooth
        )
        dice = dice.mean()

        loss = self.bce_weight * bce + self.dice_weight * dice
        return loss

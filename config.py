

BASE_FILTERS     = 32   # bottleneck = BASE_FILTERS * 16 = 512 channels
SAR_CHANNELS     = 2    # ch1 = VH, ch2 = VV  (Sigma0 in dB)
OPTICAL_CHANNELS = 4    # B2, B3, B4, B8
THERMAL_CHANNELS = 2    # reserved; no thermal weights are shipped

IMAGE_SIZE = 256
THRESHOLD  = 0.5

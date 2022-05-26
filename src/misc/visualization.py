import torch

def normalize(
        x: torch.Tensor,
        allow_negative: bool=True,
        q: float=0.95) -> torch.Tensor:

    if allow_negative:
        ref = x.abs()
    else:
        ref = x.clamp(min=0)

    x_max = ref.quantile(q)

    if allow_negative:
        x = x.clamp(min=-x_max, max=x_max)
        x = (x + x_max) / (2 * x_max)
    else:
        x = x.clamp(min=0, max=x_max)
        x = x / x_max

    return x

def to_hsl(
        x: torch.Tensor,
        allow_negative: bool=True,
        q: float=0.95) -> torch.Tensor:

    # Normalize to 0 ~ 1
    if x.ndim == 3:
        xs = []
        for _x in x:
            xs.append(normalize(_x, allow_negative=allow_negative, q=q))

        x = torch.stack(xs, dim=0)
    else:
        x = normalize(x, allow_negative=allow_negative, q=q)

    h = (1 - x) * 240

    s = 1
    l = 0.5

    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - ((h / 60) % 2 - 1).abs())

    h_0to60 = (h < 60).float()
    h_60to120 = torch.logical_and(h >= 60, h < 120).float()
    h_120to180 = torch.logical_and(h >= 120, h < 180).float()
    h_180to240 = torch.logical_and(h >= 180, h < 240).float()
    h_240to300 = torch.logical_and(h >= 240, h < 300).float()
    h_300to360 = torch.logical_and(h >= 300, h < 360).float()

    r = c * (h_0to60 + h_300to360) + x * (h_60to120 + h_240to300)
    g = c * (h_60to120 + h_120to180) + x * (h_0to60 + h_180to240)
    b = c * (h_180to240 + h_240to300) + x * (h_120to180 + h_300to360)

    if x.ndim == 3:
        rgb = torch.stack((r, g, b), dim=1)
    else:
        rgb = torch.stack((r, g, b), dim=0)

    # Normalize to -1 ~ 1
    rgb = 2 * rgb - 1
    return rgb

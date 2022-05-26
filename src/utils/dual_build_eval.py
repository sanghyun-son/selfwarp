import random

from srwarp import transform

import random_cut
import torch

def main() -> None:
    patch_sizes = [96, 128, 144, 192]
    zs = {
        'small': 1,
        'medium': 2,
        'large': 3,
    }
    n = 100

    m_dict = {}
    for size in patch_sizes:
        m_dict[size] = {}
        p = random_cut.Pyramid(size, size)
        for k, v in zs.items():
            ms = []
            for _ in range(n):
                m = p.get_random_m(
                    z_min=(0.9 * v),
                    z_max=(1.1 * v),
                    phi_min=-0.2,
                    phi_max=0.2,
                )
                if random.random() < 0.5:
                    m = transform.inverse_3x3(m)

                ms.append(m)

            m_dict[size][k] = ms

    torch.save(m_dict, 'm_dual_eval.pth')
    return

if __name__ == '__main__':
    main()

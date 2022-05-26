'''
This is a standalone file for converting existing SR networks.
'''
import argparse
import torch


def to_rrdb(from_state, to_state):
    for kf, kt in zip(from_state.keys(), to_state.keys()):
        to_state[kt] = from_state[kf]

    keys = list(to_state.keys())
    # Input layer weight and bias scaling
    # 64 x 3 x 3 x 3
    to_state[keys[0]] /= 2
    to_state[keys[1]] += to_state[keys[0]].sum((1, 2, 3))
    # Output layer weight and bias scaling
    to_state[keys[-2]] *= 2
    to_state[keys[-1]] = 2 * to_state[keys[-1]] - 1
    state = {
        'model': to_state,
        'metadata': 'RRDB implementation from the authors',
    }
    return state

def to_ddbpn(from_state, to_state):
    for kf, kt in zip(from_state.keys(), to_state.keys()):
        if 'up' in kf or 'down' in kf:
            v = from_state[kf]
            if v.ndim > 1 and v.size(1) > 64:
                print('Weights are reordered:', kf, v.size())
                split = v.split(64, dim=1)
                from_state[kf] = torch.cat(split[::-1], dim=1)

        to_state[kt] = from_state[kf]

    keys = list(to_state.keys())
    # Input layer weight and bias scaling
    # 64 x 3 x 3 x 3
    print('Input layer is renormalized:', keys[0], keys[1])
    to_state[keys[0]] /= 2
    to_state[keys[1]] += to_state[keys[0]].sum((1, 2, 3))
    # Output layer weight scaling
    print('Output layer is renormalized:', keys[-2])
    to_state[keys[-2]] *= 2
    state = {
        'model': to_state,
        'metadata': 'DDBPN implementation from the authors',
    }
    return state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--from_state', type=str)
    parser.add_argument('-t', '--to_state', type=str)
    parser.add_argument('-s', '--save_as', type=str)
    parser.add_argument('-m', '--model', type=str)
    cfg = parser.parse_args()

    from_state = torch.load(cfg.from_state)
    to_state = torch.load(cfg.to_state)
    if cfg.model in ('RRDB', 'ESRGAN'):
        state = to_rrdb(from_state, to_state)
    elif cfg.model in ('DDBPN',):
        state = to_ddbpn(from_state, to_state)

    torch.save(state, cfg.save_as)

if __name__ == '__main__':
    main()

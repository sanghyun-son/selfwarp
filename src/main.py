from os import path

from trainer import base_trainer
from config import get_config
from misc import logger

import torch
import tqdm
from speaking_gpu import bot

def main() -> None:
    cfg = get_config.parse()
    sg_bot = bot.SpeakingGPU(prefix=path.join(cfg.save, cfg.ablation))

    try:
        with logger.Logger(cfg) as l:
            if not cfg.test_only:
                sg_bot.send('Start the training.')

            t = base_trainer.get_trainer(cfg, l)
            if cfg.test_only:
                t.evaluation()
            else:
                tq = tqdm.trange(t.begin_epoch, cfg.epochs, ncols=80)
                for epoch in tq:
                    tq.set_description(
                        'Epoch {}/{}'.format(epoch + 1, cfg.epochs),
                    )
                    t.fit()
                    if (epoch + 1) % cfg.test_period == 0:
                        t.evaluation()

                    t.at_epoch_end()

                    if (epoch + 1) % 30 == 0:
                        sg_bot.send(f'Epoch {epoch + 1} / {cfg.epochs} finished.')

            t.misc.join_background()

    except KeyboardInterrupt:
        print('!!!Terminate the program!!!')
        exit()

    if not cfg.test_only:
        sg_bot.send('Finished!', force_notification=True)

    return

if __name__ == '__main__':
    main()

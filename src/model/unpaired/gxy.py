from model import common
from model.sr import rcan


class GXY(rcan.RCAN):

    def __init__(self) -> None:
        super().__init__(
            depth=10,
            n_resgroups=5,
            upsample=False,
        )
        return

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv) -> dict:
        return {}

from model import common
from model.sr import rcan


class UYY(rcan.RCAN):

    def __init__(self) -> None:
        super().__init__(
            depth=20,
            n_resgroups=5,
            upsample=True,
        )
        return

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv) -> dict:
        return {}

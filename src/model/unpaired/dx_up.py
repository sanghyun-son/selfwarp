from model.unpaired import d_shared


class DXUp(d_shared.DShared):

    def __init__(self) -> None:
        super().__init__(scale=4)
        print('dx_up discriminator')
        return
from model.unpaired import d_shared


class DY(d_shared.DShared):

    def __init__(self) -> None:
        super().__init__(scale=1)
        print('dy discriminator')
        return
from os import path

from data.sr.div2k import patch


class Flickr2KPatch(patch.DIV2KPatch):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_path(self, dpath, train):
        dpath = path.join(dpath, 'Flickr2K', 'patch_importance')
        return dpath


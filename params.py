import numpy as np



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self
    

params = AttrDict(
    # Training params
    batch_size=16,
    learning_rate=0.00001,
    max_grad_norm=None,

    # Data params
    sample_rate=22050,
    n_fft=1024,
    n_win=1024,
    n_hop=256,

    unconditional=True,
    iters=8,  ## SANTI: move to 12? 20? let's discuss

    # unconditional sample len
    audio_len = 128 * 256, # unconditional_synthesis_samples
)

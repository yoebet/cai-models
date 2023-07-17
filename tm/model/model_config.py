class ModelConfig:
    def __init__(self,
                 d_input=34,
                 d_model=1024,
                 heads=8,
                 blocks=6,
                 d_ff=1024,
                 dropout=0.1,
                 max_len=128,
                 d_gen_ff=16
                 ):
        self.d_input = d_input
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.blocks = blocks
        self.dropout = dropout
        self.max_len = max_len
        self.d_gen_ff = d_gen_ff

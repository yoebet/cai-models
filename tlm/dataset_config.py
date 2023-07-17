class KlineDatasetConfig:
    def __init__(self,
                 kl_dir,
                 symbol='BTCUSDT',
                 interval='1h',
                 ):
        self.kl_dir = kl_dir
        self.symbol = symbol
        self.interval = interval


class DatasetConfig:
    def __init__(self,
                 kl_config: KlineDatasetConfig,
                 news_meta_dir,
                 news_vec_dir,
                 shuffle_buffer=20,
                 batch_size=2,
                 train_batches_per_step=30,
                 val_batches_per_step=5,
                 ):
        self.kl_config = kl_config
        self.news_meta_dir = news_meta_dir
        self.news_vec_dir = news_vec_dir
        self.shuffle_buffer = shuffle_buffer
        self.batch_size = batch_size
        self.train_batches_per_step = train_batches_per_step
        self.val_batches_per_step = val_batches_per_step

    def __str__(self):
        return f'''
        DatasetConfig({self.news_meta_dir = }, {self.batch_size = })
        '''

    def __repr__(self):
        return f'DatasetConfig(\'{self.news_meta_dir}\', {self.batch_size})'

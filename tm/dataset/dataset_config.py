SPLIT_NAME_TRAIN = 'train'
SPLIT_NAME_VALIDATION = 'valid'


class DatasetConfig:
    def __init__(self,
                 base_dir,
                 symbol='BTCUSDT',
                 interval='15m',
                 extra_intervals=None,
                 shuffle_buffer=20,
                 seq_len=4,
                 batch_size=2,
                 train_batches_per_step=30,
                 val_batches_per_step=5,
                 cached_batch_base_dir=None,
                 use_cached_batch=True,
                 ds_split_method_name=None,
                 ):
        self.base_dir = base_dir
        self.symbol = symbol
        self.interval = interval
        self.extra_intervals = extra_intervals
        self.shuffle_buffer = shuffle_buffer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.train_batches_per_step = train_batches_per_step
        self.val_batches_per_step = val_batches_per_step
        self.cached_batch_base_dir = cached_batch_base_dir
        self.use_cached_batch = use_cached_batch
        self.ds_split_method_name = ds_split_method_name


intervals_meta = {
    '5s': {
        'seconds': 1,
    },
    '1m': {
        'seconds': 60,
    },
    '15m': {
        'seconds': 15 * 60,
        'one_file': True,
    },
    '1h': {
        'seconds': 60 * 60,
        'one_file': True,
        'iterate_one_file': True,
    },
    '4h': {
        'seconds': 4 * 60 * 60,
        'one_file': True,
        'rolling': '1h',
        'iterate_one_file': True,
    },
    '1d': {
        'seconds': 24 * 60 * 60,
        'one_file': True,
        'rolling': '1h',
        'iterate_one_file': True,
    }
}

extra_intervals_map = {'1m': ('5s', '15m', '4h'),
                       '15m': ('1m', '4h', '1d'),
                       '1h': ('15m', '4h', '1d'),
                       '4h': ('1h', '1d'),
                       '1d': ('1h', '4h')
                       }


def get_d_input_by_interval(interval):
    vl = 10
    d_input = 4 + vl
    eis = extra_intervals_map.get(interval)
    if eis is not None:
        d_input += vl * len(eis)
    return d_input

from torch.utils.data import Dataset, DataLoader
from dataset_config import KlineDatasetConfig


class KlMapDataset(Dataset):
    def __init__(self,
                 kl_dir,
                 symbol='BTCUSDT',
                 interval='1h',
                 ):
        self.kl_dir = kl_dir
        self.symbol = symbol
        self.interval = interval
        self.month_cache = {}

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return 10

    def get_by_time(self, ts):
        return idx


if __name__ == '__main__':
    base_dir = '../data/tm'
    klds = KlMapDataset(f'{base_dir}/market/basic/spot-kline',
                        interval='4h',
                        )
    print(klds['abc'])

    dl = DataLoader(dataset=klds,
                    # batch_size=ds_config.batch_size,
                    drop_last=True,
                    num_workers=0
                    )

    for idx in dl:
        print(idx)

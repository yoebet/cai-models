import os
import numpy as np
import pandas as pd
import torchdata.datapipes as dp
from torch.utils.data import IterDataPipe
import jsonlines
from dataset_config import DatasetConfig, KlineDatasetConfig


@dp.functional_datapipe("load_news_vec")
class NewsVecIterDataPipe(IterDataPipe):

    def __init__(self, datapipe, ds_config: DatasetConfig, name='-'):
        self.datapipe = datapipe
        self.ds_config = ds_config
        self.name = name

    def __iter__(self):
        news_meta_dir = self.ds_config.news_meta_dir
        news_vec_dir = self.ds_config.news_vec_dir
        for filename in self.datapipe:
            print(filename)
            vec_dir_the_day = os.path.dirname(filename)
            # TODO
            vec_dir_the_day = vec_dir_the_day.replace(news_meta_dir, news_vec_dir)
            if not os.path.exists(vec_dir_the_day):
                continue
            with jsonlines.open(filename) as reader:
                for row in reader:
                    # 'id': 'e8ejSyX2qRDnGaPyoHk9wq',
                    # 'source': 'blockchain',
                    # 'url': 'https://blockchainreporter.net/...',
                    # 'title': 'Nike Successful in Gathering $185M Proceeds by Selling NFTs',
                    # 'text_word_count': 530,
                    # 'article_time': '2022-08-23T09:57:00'
                    news_id = row['id']
                    vec_filename = f'{vec_dir_the_day}/{news_id}.npy'
                    if not os.path.exists(vec_filename):
                        continue
                    vec_data = np.load(vec_filename)
                    yield {'news_id': news_id, 'vec_data': vec_data.shape}

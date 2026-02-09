
import os
import json
from typing import Union, Dict, Optional
import ast

import pandas as pd
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle


def _default_data_dir(task_subdir: str) -> str:
    env_root = os.getenv('SOFTCOT_DATA_DIR')
    if env_root:
        return os.path.join(env_root, task_subdir)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(repo_root, 'data', task_subdir)


def _validate_paths(paths: Dict[str, str], loader_name: str) -> None:
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        missing_list = ', '.join(missing)
        raise FileNotFoundError(
            f'{loader_name} missing dataset file(s): {missing_list}. '
            'Set SOFTCOT_DATA_DIR to your dataset root, '
            'or pass explicit paths to loader.load(paths=...).'
        )


class GSM8KLoader(Loader):

    def _load(self, path: str) -> DataSet:
        ds = DataSet()

        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    data = json.loads(line)
                    instance = Instance(**data)
                    ds.append(instance)

        return ds

    def load(self, paths: Optional[Union[str, Dict[str, str]]] = None) -> DataBundle:
        if paths is None:
            paths = _default_data_dir('gsm8k')
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'train_socratic.jsonl'),
                'dev': os.path.join(paths, 'test_socratic.jsonl'),
                'test': os.path.join(paths, 'test_socratic.jsonl')
            }

        _validate_paths(paths, self.__class__.__name__)
        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})


class AQuALoader(Loader):

    def _load(self, path: str) -> DataSet:
        ds = DataSet()
        with open(path) as f:
            ins_list = json.load(f)

        for ins in ins_list:
            instance = Instance(**ins)
            ds.append(instance)

        return ds

    def load(self, paths: Optional[Union[str, Dict[str, str]]] = None) -> DataBundle:
        if paths is None:
            paths = _default_data_dir('aqua')
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'gsm_style_train.jsonl'),
                'dev': os.path.join(paths, 'gsm_style_dev.jsonl'),
                'test': os.path.join(paths, 'gsm_style_test.jsonl')
            }

        _validate_paths(paths, self.__class__.__name__)
        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})


class DULoader(Loader):

    def _load(self, path: str) -> DataSet:
        ds = DataSet()
        with open(path) as f:
            ins_list = json.load(f)

        for ins in ins_list:
            instance = Instance(**ins)
            ds.append(instance)

        return ds

    def load(self, paths: Optional[Union[str, Dict[str, str]]] = None) -> DataBundle:
        if paths is None:
            paths = _default_data_dir('du')
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'date_understanding_gsm_style.json'),
                'dev': os.path.join(paths, 'date_understanding_gsm_style.json'),
                'test': os.path.join(paths, 'date_understanding_gsm_style.json')
            }

        _validate_paths(paths, self.__class__.__name__)
        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})


class StrategyQALoader(Loader):

    def __init__(self, train_split=0.8):
        super().__init__()
        self.train_split = train_split

    def _load(self, path: str, is_train=True) -> DataSet:
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        num_train = int(len(dataset) * self.train_split)
        if is_train:
            for ins in dataset[:num_train]:
                ds.append(Instance(**ins))
        else:
            for ins in dataset[num_train:]:
                ds.append(Instance(**ins))
        return ds

    def load(self, paths: Optional[Union[str, Dict[str, str]]] = None) -> DataBundle:
        if paths is None:
            paths = _default_data_dir('strategyqa')
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'strategyqa_train.json'),
                'dev': os.path.join(paths, 'strategyqa_train.json'),
                'test': os.path.join(paths, 'strategyqa_train.json')
            }

        _validate_paths(paths, self.__class__.__name__)
        return DataBundle(datasets={k: self._load(v, is_train=('train' in [k])) for k, v in paths.items()})


class AugASDivLoader(GSM8KLoader):
    def load(self, paths: Optional[Union[str, Dict[str, str]]] = None) -> DataBundle:
        if paths is None:
            paths = _default_data_dir('asdiv-aug')
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'aug-train.jsonl'),
                'dev': os.path.join(paths, 'aug-dev.jsonl'),
                'test': os.path.join(paths, 'aug-dev.jsonl')
            }

        _validate_paths(paths, self.__class__.__name__)
        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})



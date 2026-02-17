import os
import json
import re
from typing import Any, Dict, List, Optional, Union

from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle


def _dedup(items: List[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output


def _task_dir_aliases(task_subdir: str) -> List[str]:
    task = task_subdir.strip("/\\")
    aliases = [task, task.lower(), task.upper(), task.capitalize()]
    if task.lower() == "strategyqa":
        aliases.extend(["StrategyQA"])
    elif task.lower() == "aqua":
        aliases.extend(["AQUA"])
    elif task.lower() == "gsm8k":
        aliases.extend(["GSM8K"])
    return _dedup(aliases)


def _default_data_dir(task_subdir: str) -> str:
    env_root = os.getenv('SOFTCOT_DATA_DIR')
    task_aliases = _task_dir_aliases(task_subdir)
    if env_root:
        for alias in task_aliases:
            candidate = os.path.join(env_root, alias)
            if os.path.isdir(candidate):
                return candidate
        return os.path.join(env_root, task_subdir)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.join(repo_root, 'data')
    for alias in task_aliases:
        candidate = os.path.join(default_root, alias)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(default_root, task_subdir)


def _validate_paths(paths: Dict[str, str], loader_name: str) -> None:
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        missing_list = ', '.join(missing)
        raise FileNotFoundError(
            f'{loader_name} missing dataset file(s): {missing_list}. '
            'Set SOFTCOT_DATA_DIR to your dataset root, '
            'or pass explicit paths to loader.load(paths=...).'
        )


def _paths_from_env(prefix: str) -> Optional[Dict[str, str]]:
    train = os.getenv(f"SOFTCOT_{prefix}_TRAIN")
    dev = os.getenv(f"SOFTCOT_{prefix}_DEV")
    test = os.getenv(f"SOFTCOT_{prefix}_TEST")
    if not any([train, dev, test]):
        return None
    if not train or not test:
        raise ValueError(
            f"SOFTCOT_{prefix}_TRAIN and SOFTCOT_{prefix}_TEST must both be set when overriding dataset paths."
        )
    if not dev:
        dev = test
    return {"train": train, "dev": dev, "test": test}


def _resolve_default_split_paths(
    root: str,
    candidates: List[Dict[str, str]],
) -> Dict[str, str]:
    """
    Pick the first all-existing split layout from candidates.
    If none match, return the first candidate resolved to full paths
    so _validate_paths can raise a clear error.
    """
    resolved_candidates = []
    for candidate in candidates:
        resolved = {k: os.path.join(root, v) for k, v in candidate.items()}
        resolved_candidates.append(resolved)
        if all(os.path.exists(path) for path in resolved.values()):
            return resolved
    return resolved_candidates[0]


def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    text = raw.strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]
    except json.JSONDecodeError:
        pass

    rows: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _to_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "yes", "1"}:
            return True
        if lower in {"false", "no", "0"}:
            return False
    return value


def _extract_choice_letter(value: Any, valid_letters: str = "ABCDEF") -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        if text.startswith("####"):
            text = text[4:].strip()

        if len(text) == 1 and text.upper() in valid_letters:
            return text.upper()

        patterns = [
            rf"(?i)####\s*([{re.escape(valid_letters)}])\b",
            rf"(?i)(?:final\s+answer|answer)\s*(?:is|:)?\s*(?:\\boxed\{{)?\s*([{re.escape(valid_letters)}])\s*(?:\}})?",
            rf"(?i)\b(?:option|choice)\s*([{re.escape(valid_letters)}])\b",
            rf"(?i)\b([{re.escape(valid_letters)}])\b(?=[^A-Za-z]*$)",
        ]
        for pattern in patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                return matches[-1].group(1).upper()
        return None

    return None


class GSM8KLoader(Loader):

    def _normalize_instance(self, ins: Dict[str, Any], idx: int) -> Dict[str, Any]:
        normalized = dict(ins)
        normalized["question"] = str(normalized.get("question", "")).strip()

        answer_value = normalized.get("answer")
        if answer_value is None:
            answer_value = (
                normalized.get("gold")
                or normalized.get("answer_from_dataset")
                or normalized.get("response_ans")
            )

        if answer_value is None:
            normalized_answer = ""
        else:
            answer_text = str(answer_value).strip()
            answer_lines = answer_text.splitlines() if answer_text else []
            if answer_text.startswith("####"):
                normalized_answer = answer_text
            elif answer_lines and answer_lines[-1].strip().startswith("####"):
                normalized_answer = answer_text
            elif answer_text:
                normalized_answer = f"#### {answer_text}"
            else:
                normalized_answer = ""

        normalized["answer"] = normalized_answer
        normalized.setdefault("index", normalized.get("id", f"gsm8k_{idx:06d}"))
        return normalized

    def _load(self, path: str) -> DataSet:
        ds = DataSet()
        ins_list = _read_json_or_jsonl(path)
        for idx, ins in enumerate(ins_list):
            normalized = self._normalize_instance(ins, idx)
            instance = Instance(**normalized)
            ds.append(instance)

        return ds

    def load(self, paths: Optional[Union[str, Dict[str, str]]] = None) -> DataBundle:
        if paths is None:
            env_paths = _paths_from_env("GSM8K")
            if env_paths is not None:
                paths = env_paths
            else:
                paths = _default_data_dir('gsm8k')
        if isinstance(paths, str):
            paths = _resolve_default_split_paths(paths, [
                {
                    'train': 'train_socratic.jsonl',
                    'dev': 'test_socratic.jsonl',
                    'test': 'test_socratic.jsonl',
                },
                {
                    'train': 'train.jsonl',
                    'dev': 'test.jsonl',
                    'test': 'test.jsonl',
                },
                {
                    'train': 'train.json',
                    'dev': 'test.json',
                    'test': 'test.json',
                },
            ])

        _validate_paths(paths, self.__class__.__name__)
        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})


class AQuALoader(Loader):

    def _normalize_instance(self, ins: Dict[str, Any], idx: int) -> Dict[str, Any]:
        normalized = dict(ins)

        question = str(normalized.get("question", "")).strip()
        options = normalized.get("options")
        if isinstance(options, str) and options.strip() and options.strip() not in question:
            question = f"{question}\n{options.strip()}"
        normalized["question"] = question

        answer_letter = (
            _extract_choice_letter(normalized.get("answer"), valid_letters="ABCDE")
            or _extract_choice_letter(normalized.get("gold"), valid_letters="ABCDE")
            or _extract_choice_letter(normalized.get("answer_from_dataset"), valid_letters="ABCDE")
        )

        answer_value = normalized.get("answer")
        if isinstance(answer_value, str) and answer_value.strip().startswith("####"):
            normalized_answer = answer_value.strip()
        elif answer_letter is not None:
            rationale = normalized.get("response_rationale") or normalized.get("response_payload")
            if isinstance(rationale, str) and rationale.strip():
                normalized_answer = f"{rationale.strip()}\n#### {answer_letter}"
            else:
                normalized_answer = f"#### {answer_letter}"
        elif isinstance(answer_value, str):
            normalized_answer = answer_value
        elif answer_value is not None:
            normalized_answer = str(answer_value)
        else:
            normalized_answer = ""

        normalized["answer"] = normalized_answer
        normalized.setdefault("index", normalized.get("id", f"aqua_{idx:06d}"))
        return normalized

    def _load(self, path: str) -> DataSet:
        ds = DataSet()
        ins_list = _read_json_or_jsonl(path)

        for idx, ins in enumerate(ins_list):
            normalized = self._normalize_instance(ins, idx)
            instance = Instance(**normalized)
            ds.append(instance)

        return ds

    def load(self, paths: Optional[Union[str, Dict[str, str]]] = None) -> DataBundle:
        if paths is None:
            env_paths = _paths_from_env("AQUA")
            if env_paths is not None:
                paths = env_paths
            else:
                paths = _default_data_dir('aqua')
        if isinstance(paths, str):
            paths = _resolve_default_split_paths(paths, [
                {
                    'train': 'gsm_style_train.jsonl',
                    'dev': 'gsm_style_dev.jsonl',
                    'test': 'gsm_style_test.jsonl',
                },
                {
                    'train': 'train.jsonl',
                    'dev': 'test.jsonl',
                    'test': 'test.jsonl',
                },
                {
                    'train': 'train.json',
                    'dev': 'test.json',
                    'test': 'test.json',
                },
            ])

        _validate_paths(paths, self.__class__.__name__)
        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})


class DULoader(Loader):

    def _load(self, path: str) -> DataSet:
        ds = DataSet()
        ins_list = _read_json_or_jsonl(path)

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

    def _normalize_instance(self, ins: Dict[str, Any], idx: int) -> Dict[str, Any]:
        normalized = dict(ins)

        answer_value = (
            normalized.get("answer")
            if "answer" in normalized
            else normalized.get("gold", normalized.get("answer_from_dataset"))
        )
        normalized["answer"] = _to_bool(answer_value)

        facts = normalized.get("facts")
        if isinstance(facts, list):
            normalized["facts"] = [str(item).strip() for item in facts if str(item).strip()]
        elif isinstance(facts, str) and facts.strip():
            normalized["facts"] = [facts.strip()]
        else:
            normalized["facts"] = []

        normalized["question"] = str(normalized.get("question", "")).strip()
        normalized.setdefault("index", normalized.get("id", f"strategyqa_{idx:06d}"))
        return normalized

    def _load(self, path: str, is_train: bool = True, split_single_file: bool = False) -> DataSet:
        ds = DataSet()

        dataset = _read_json_or_jsonl(path)
        if split_single_file:
            num_train = int(len(dataset) * self.train_split)
            if is_train:
                dataset = dataset[:num_train]
            else:
                dataset = dataset[num_train:]

        for idx, ins in enumerate(dataset):
            normalized = self._normalize_instance(ins, idx)
            ds.append(Instance(**normalized))

        return ds

    def load(self, paths: Optional[Union[str, Dict[str, str]]] = None) -> DataBundle:
        if paths is None:
            env_paths = _paths_from_env("STRATEGYQA")
            if env_paths is not None:
                paths = env_paths
            else:
                paths = _default_data_dir('strategyqa')
        if isinstance(paths, str):
            paths = _resolve_default_split_paths(paths, [
                {
                    'train': 'strategyqa_train.json',
                    'dev': 'strategyqa_train.json',
                    'test': 'strategyqa_train.json',
                },
                {
                    'train': 'train.jsonl',
                    'dev': 'test.jsonl',
                    'test': 'test.jsonl',
                },
                {
                    'train': 'train.json',
                    'dev': 'test.json',
                    'test': 'test.json',
                },
            ])

        _validate_paths(paths, self.__class__.__name__)
        split_single_file = len({os.path.abspath(v) for v in paths.values()}) == 1
        return DataBundle(datasets={
            k: self._load(v, is_train=(k == 'train'), split_single_file=split_single_file)
            for k, v in paths.items()
        })


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



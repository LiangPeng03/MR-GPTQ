import random
import gzip
import json
from itertools import islice
from pathlib import Path
from typing import Optional, List

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer


def load_c4_shard(split):
    """Read the original C4 shard, including caches with older config hashes."""
    if split not in {"train", "validation"}:
        raise ValueError(f"Unsupported C4 split: {split}")
    shard = (
        "en/c4-train.00000-of-01024.json.gz" if split == "train"
        else "en/c4-validation.00000-of-00008.json.gz"
    )
    try:
        return load_dataset("allenai/c4", data_files={split: shard}, split=split)
    except (ValueError, FileNotFoundError, ConnectionError) as original_error:
        from datasets import config

        cache = Path(config.HF_DATASETS_CACHE) / "allenai___c4"
        candidates = []
        for info_path in cache.glob("*/*/*/dataset_info.json"):
            info = json.loads(info_path.read_text(encoding="utf-8"))
            sources = info.get("download_checksums", {})
            # Require this exact single source shard, preserving document order.
            if len(sources) != 1 or not any(
                source.endswith("/" + shard) for source in sources
            ):
                continue
            split_info = info.get("splits", {}).get(split)
            if split_info is None:
                continue
            files = sorted(info_path.parent.glob(f"c4-{split}*.arrow"))
            if files:
                candidates.append((info_path, split_info["num_examples"], files))
        if not candidates:
            if split == "validation":
                from huggingface_hub import hf_hub_download

                raw_path = hf_hub_download(
                    repo_id="allenai/c4", repo_type="dataset", filename=shard,
                    revision="1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
                    local_files_only=True,
                )
                # The PPL protocol uses exactly the first 1,100 documents.
                with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
                    rows = [{"text": json.loads(line)["text"]}
                            for line in islice(handle, 1100)]
                if len(rows) != 1100:
                    raise ValueError(f"Incomplete C4 validation prefix: {raw_path}")
                print(f"C4 validation: local gzip, first 1100 documents ({raw_path})")
                return Dataset.from_list(rows)
            raise FileNotFoundError(
                f"No verified local Arrow cache for {shard} under {cache}"
            ) from original_error
        # Multiple caches for the same source are equivalent. Prefer the pinned
        # source revision already used in this experiment.
        candidates.sort(key=lambda candidate: (
            candidate[0].parent.name != "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
            str(candidate[0]),
        ))
        info_path, expected_rows, files = candidates[0]
        parts = [Dataset.from_file(str(file)) for file in files]
        dataset = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        if len(dataset) != expected_rows:
            raise ValueError(
                f"Incomplete C4 cache {info_path.parent}: "
                f"{len(dataset)} rows, expected {expected_rows}"
            )
        print(f"C4 {split}: local Arrow cache {info_path.parent} ({len(dataset)} rows)")
        return dataset


# Only for evaluation
def get_wikitext2(tokenizer: AutoTokenizer,  sequence_length: int):
    test_dataset_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_dataset_tok = tokenizer("\n\n".join(test_dataset_raw["text"]), return_tensors="pt").input_ids
    num_test_sequences = test_dataset_tok.numel() // sequence_length
    test_loader = []
    for i in range(num_test_sequences):
        test_loader.append(test_dataset_tok[:, i * sequence_length : (i + 1) * sequence_length])
    return test_loader


def get_c4_eval(tokenizer: AutoTokenizer,  sequence_length: int):
    val_datasetraw = load_c4_shard("validation")
    valenc = tokenizer(' '.join(val_datasetraw[:1100]['text']), return_tensors='pt').input_ids
    num_seqs = min(valenc.numel() // sequence_length, 256)
    valenc = valenc[:, :(num_seqs * sequence_length)]
    test_loader = []
    for i in range(num_seqs):
        test_loader.append(valenc[:, i * sequence_length : (i + 1) * sequence_length])
    return test_loader

def get_c4(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
):
    train_datasetraw = load_c4_shard("train")
    all_id =[]
    trainloader = []
    for _ in range(num_calibration_samples):
        while True:
            i = random.randint(0, len(train_datasetraw) - 1)
            if i in all_id:
                continue
            trainenc = tokenizer(train_datasetraw[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= max_sequence_length:
                break
        all_id.append(i)
        i = random.randint(0, trainenc.input_ids.shape[1] - max_sequence_length)
        tokenized_sample = trainenc.input_ids[:, i:i + max_sequence_length]
        trainloader.append(tokenized_sample)
    return trainloader


def get_wikitext2_calib(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
):
    """Load WikiText-2 train split for calibration."""
    random.seed(seed)
    train_dataset_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # Filter out empty texts (WikiText uses empty lines as paragraph separators)
    train_dataset_raw = train_dataset_raw.filter(lambda x: len(x["text"].strip()) > 0)
    
    # Concatenate all texts into one long sequence (same approach as get_wikitext2 eval)
    full_text = "\n\n".join(train_dataset_raw["text"])
    full_enc = tokenizer(full_text, return_tensors="pt").input_ids
    
    # Ensure we have enough tokens
    total_tokens = full_enc.shape[1]
    if total_tokens < max_sequence_length:
        raise ValueError(f"WikiText-2 train has only {total_tokens} tokens, less than max_sequence_length={max_sequence_length}")
    
    trainloader = []
    for _ in range(num_calibration_samples):
        i = random.randint(0, total_tokens - max_sequence_length)
        tokenized_sample = full_enc[:, i:i + max_sequence_length]
        trainloader.append(tokenized_sample)
    return trainloader


def get_open_thoughts(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    train_dataset_raw = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
    if num_calibration_samples:
        train_dataset_raw = train_dataset_raw.shuffle(seed=seed).select(range(num_calibration_samples))
    # Preprocess the data into the format the model is trained with.
    def preprocess(example):
        messages = []
        # add system prompt
        messages.append({"role": "system", "content": example['system']})
        # add dialogue
        for message in example['conversations']:
            messages.append({"role": message["from"], "content": message["value"]})
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
    train_dataset_raw = train_dataset_raw.map(preprocess)
    # Tokenize the data
    def tokenize(sample):
        return tokenizer(
            sample["text"], 
            padding=False, 
            max_length=max_sequence_length, 
            truncation=True, 
            add_special_tokens=False,
        )
    train_dataset = train_dataset_raw.map(tokenize, remove_columns=train_dataset_raw.column_names)
    train_dataset = [torch.tensor(sample['input_ids']).unsqueeze(0) for sample in train_dataset]
    return train_dataset


def get_open_platypus(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    train_dataset_raw = load_dataset("garage-bAInd/Open-Platypus", split="train")
    if num_calibration_samples:
        train_dataset_raw = train_dataset_raw.shuffle(seed=seed).select(range(num_calibration_samples))
    # Preprocess the data into the format the model is trained with.
    def preprocess(example):
        messages = [
            {"role": "user", "content": example["instruction"]}, 
            {"role": "assistant", "content":  example["output"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
    train_dataset_raw = train_dataset_raw.map(preprocess)
    # Tokenize the data
    def tokenize(sample):
        return tokenizer(
            sample["text"], 
            padding=False, 
            max_length=max_sequence_length, 
            truncation=True, 
            add_special_tokens=False,
        )
    train_dataset = train_dataset_raw.map(tokenize, remove_columns=train_dataset_raw.column_names)
    train_dataset = [torch.tensor(sample['input_ids']).unsqueeze(0) for sample in train_dataset]
    return train_dataset

def get_fineweb_edu(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    train_dataset_raw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    train_dataset_raw = train_dataset_raw.shuffle(seed=seed, buffer_size=1_000)
    trainloader = []
    for j, sample in enumerate(train_dataset_raw):
        trainenc = tokenizer(
            sample['text'],
            return_tensors="pt"
        )
        if trainenc.input_ids.shape[1] < max_sequence_length:
            continue
        i = random.randint(0, trainenc.input_ids.shape[1] - max_sequence_length)
        tokenized_sample = trainenc.input_ids[:, i:i + max_sequence_length]
        trainloader.append(tokenized_sample)
        if len(trainloader)>=num_calibration_samples:
            break
    return trainloader

def get_ultrachat_200k(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    train_dataset_raw = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    if num_calibration_samples:
        train_dataset_raw = train_dataset_raw.shuffle(seed=seed).select(range(num_calibration_samples))
    # Preprocess the data into the format the model is trained with.
    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    train_dataset_raw = train_dataset_raw.map(preprocess)
    # Tokenize the data
    def tokenize(sample):
        return tokenizer(
            sample["text"], 
            padding=False, 
            max_length=max_sequence_length, 
            truncation=True, 
            add_special_tokens=False,
        )
    train_dataset = train_dataset_raw.map(tokenize, remove_columns=train_dataset_raw.column_names)
    train_dataset = [torch.tensor(sample['input_ids']).unsqueeze(0) for sample in train_dataset]
    return train_dataset

def get_tulu3_sft_mixture(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    # Load raw dataset
    train_dataset_raw = load_dataset("allenai/tulu-3-sft-mixture", split="train")

    # Optionally subsample early for efficiency
    if num_calibration_samples:
        train_dataset_raw = train_dataset_raw.shuffle(seed=seed).select(range(num_calibration_samples))

    # Preprocess into chat text
    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False
            )
        }

    train_dataset_raw = train_dataset_raw.map(preprocess)

    # Tokenize into input_ids
    def tokenize(sample):
        tokenized = tokenizer(
            sample["text"],
            padding=False,
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=False,
        )
        return tokenized

    tokenized_dataset = train_dataset_raw.map(tokenize, remove_columns=train_dataset_raw.column_names)

    # Convert to list of tensors
    train_dataset = [torch.tensor(sample['input_ids']).unsqueeze(0) for sample in tokenized_dataset]

    return train_dataset


def get_data(
    dataset_name: str, 
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    if dataset_name == "open-thoughts":
        return get_open_thoughts(tokenizer, max_sequence_length, num_calibration_samples, seed)
    if dataset_name == "open-platypus":
        return get_open_platypus(tokenizer, max_sequence_length, num_calibration_samples, seed)
    if dataset_name == "ultrachat-200k":
        return get_ultrachat_200k(tokenizer, max_sequence_length, num_calibration_samples, seed)
    if dataset_name == "fineweb-edu":
        return get_fineweb_edu(tokenizer, max_sequence_length, num_calibration_samples, seed)
    if dataset_name == "c4":
        return get_c4(tokenizer, max_sequence_length, num_calibration_samples, seed)
    if dataset_name == "wikitext2":
        return get_wikitext2_calib(tokenizer, max_sequence_length, num_calibration_samples, seed)
    if dataset_name == "tulu":
        return get_tulu3_sft_mixture(tokenizer, max_sequence_length, num_calibration_samples, seed)
    else:
        raise ValueError("Unknown dataset")

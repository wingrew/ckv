import json
from argparse import Namespace
from collections.abc import Mapping
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow


@dataclass
class OpenAIDataset(BaseDataset):
    dataset_path: str
    num_requests: int
    fixed_output_len: Optional[int]
    request_order: str
    pretokenize: bool

    @classmethod
    def from_args(cls, args: Namespace) -> "OpenAIDataset":
        return cls(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            fixed_output_len=args.sharegpt_output_len,
            request_order=getattr(args, "openai_request_order", "dataset"),
            pretokenize=getattr(args, "openai_pretokenize", False),
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        return sample_openai_requests(
            dataset_path=self.dataset_path,
            num_requests=self.num_requests,
            tokenizer=tokenizer,
            fixed_output_len=self.fixed_output_len,
            request_order=self.request_order,
            pretokenize=self.pretokenize,
        )


def sample_openai_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    request_order: str = "dataset",
    pretokenize: bool = False,
) -> List[DatasetRow]:
    """
    Load OpenAI-compatible chat completion requests from a JSONL file.

    Each line should be a JSON object with:
    - "messages": list of {"role": str, "content": str}
    - "max_tokens": int (used as output_len if fixed_output_len not set)
    - "tools": optional list of tool definitions
    - "temperature": optional temperature value
    - "top_p": optional top_p value
    - Other OpenAI API parameters are also extracted and passed through
    """
    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            if num_requests > 0 and len(dataset) >= num_requests:
                break
            if line.strip():
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue

    # Fields that should NOT be passed through extra_request_body
    # These are either handled separately or are metadata
    # max_tokens is excluded because it's handled via output_len -> max_completion_tokens
    # max_completion_tokens is also excluded to avoid conflicts
    EXCLUDED_FIELDS = {"messages", "max_tokens", "max_completion_tokens", "model"}

    filtered_dataset: List[DatasetRow] = []
    for data in dataset:
        messages = data.get("messages", [])
        if not messages:
            continue

        # Use max_tokens from the request, or fall back to fixed_output_len
        output_len = fixed_output_len or data.get("max_tokens", 256)

        # Extract extra request body parameters (tools, temperature, top_p, etc.)
        extra_body = {k: v for k, v in data.items() if k not in EXCLUDED_FIELDS}

        # Match the payload sent by the benchmark, including tool definitions.
        # Some tokenizers return a BatchEncoding without return_dict=True.
        encoded_prompt = tokenizer.apply_chat_template(
            messages,
            tools=extra_body.get("tools"),
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(encoded_prompt, Mapping):
            encoded_prompt = encoded_prompt["input_ids"]
        prompt_len = len(encoded_prompt)

        if pretokenize:
            # Dataset loading already tokenized the exact chat payload for
            # prompt_len and ordering. Reuse it on SGLang's timed request path.
            extra_body["input_ids"] = list(encoded_prompt)

        # Pass messages list directly - the serving benchmark handles List[Dict] prompts
        filtered_dataset.append(
            DatasetRow(
                prompt=messages,
                prompt_len=prompt_len,
                output_len=output_len,
                extra_request_body=extra_body,  # Store per-request parameters
            )
        )

    if request_order == "shortest-input":
        # The client still maintains the requested concurrency. Front-loading
        # short prefills reduces the cold-wave completion-time penalty and lets
        # the server's SJF policy see similarly sized requests together.
        filtered_dataset.sort(key=lambda row: row.prompt_len)
    elif request_order in ("ttft-balanced", "ttft-short-first"):
        ordered = sorted(filtered_dataset, key=lambda row: row.prompt_len)
        cold_wave = ordered[:12]
        remaining = ordered[12:]
        balanced = []
        lo, hi = 0, len(remaining) - 1
        while lo <= hi:
            if request_order == "ttft-short-first":
                balanced.append(remaining[lo])
                lo += 1
                if lo <= hi:
                    balanced.append(remaining[hi])
                    hi -= 1
            else:
                balanced.append(remaining[hi])
                hi -= 1
                if lo <= hi:
                    balanced.append(remaining[lo])
                    lo += 1
        filtered_dataset = cold_wave + balanced

    print(f"Loaded {len(filtered_dataset)} OpenAI-format requests")
    print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in filtered_dataset])}")
    return filtered_dataset

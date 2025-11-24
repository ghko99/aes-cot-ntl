from transformers import DataCollatorForLanguageModeling
import unicodedata
from typing import List, Dict, Union


# =========================
# 기타 유틸/설정
# =========================
def normalize_text(text):
    return unicodedata.normalize("NFC", text)

# =========================
# Collator (동적 패딩 + 질문 마스킹)
# =========================
class LlamaAESCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, max_seq_length=2048):
        super().__init__(tokenizer, mlm=False)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eot_token = tokenizer.eos_token
        self.max_seq_length = max_seq_length

    def __call__(self, examples: List[Dict[str, Union[str, List[int]]]]):
        merged_sequences = [
            f"{normalize_text(ex['instruction'])}{normalize_text(ex['output'])}{self.eot_token}"
            for ex in examples
        ]
        batch = self.tokenizer(
            merged_sequences,
            padding="longest",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        labels = batch["input_ids"].clone()

        for i, ex in enumerate(examples):
            q_ids = self.tokenizer(
                normalize_text(ex["instruction"]),
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )["input_ids"][0]
            q_len = q_ids.size(0)
            labels[i, :q_len] = -100

        labels[labels == self.pad_token_id] = -100
        return {"input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": labels}
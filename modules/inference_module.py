from pathlib import Path
import csv
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from .number_tokenizer import AutoNumberTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

# ======================================================
# 숫자 토큰 ID 매핑 함수
# ======================================================
def build_digit_token_id_map(tokenizer):
    """
    1~9에 해당하는 가장 '간단한' 토큰 id를 선택.
    우선순위: '1'~'9' -> '▁1'~'▁9' -> decode 가능한 첫 토큰
    """
    cand_by_num = {d: [] for d in range(1, 10)}
    vocab = tokenizer.get_vocab()
    for tok, tid in vocab.items():
        try:
            val = tokenizer.decode_number_token(tok)
        except ValueError:
            continue
        if val in range(1, 10) and float(val).is_integer():
            cand_by_num[int(val)].append(tok)

    digit_map = {}
    for d, toks in cand_by_num.items():
        if not toks:
            for t in [str(d), f"▁{d}"]:
                if t in vocab:
                    digit_map[d] = vocab[t]
                    break
            if d not in digit_map:
                raise ValueError(f"숫자 {d} 에 해당하는 토큰을 찾지 못했습니다.")
            continue

        if str(d) in toks:
            chosen = str(d)
        elif f"▁{d}" in toks:
            chosen = f"▁{d}"
        else:
            chosen = toks[0]
        digit_map[d] = vocab[chosen]
    return digit_map  # {1: id, ..., 9: id}


# ======================================================
# 모델 로딩
# ======================================================
def load_inference_model(adapter_dir, base_model_name="meta-llama/Llama-3.1-8B-Instruct"):
    try:
        use_bf16 = torch.cuda.is_bf16_supported()
    except Exception:
        use_bf16 = False

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model_lora = PeftModel.from_pretrained(base, adapter_dir)
    model_lora.eval()
    return model_lora


# ======================================================
# 추론 및 CSV 저장
# ======================================================
@torch.inference_mode()
def run_test_and_save_csv(
    test_file: str,
    out_dir: str,
    adapter_dir: str,
    max_seq_length: int = 1456,
    max_new_tokens: int = 16,
):
    """
    test_file: jsonl 형식의 테스트 데이터
    out_dir: 결과 저장 디렉토리
    adapter_dir: 학습된 모델 경로 (LoRA)
    """

    # 모델 / 토크나이저 로드
    model = load_inference_model(adapter_dir)
    tokenizer = AutoNumberTokenizer.from_pretrained(adapter_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_length

    # 데이터셋 로드
    ds = load_dataset("json", data_files=test_file)["train"]
    digit_id_map = build_digit_token_id_map(tokenizer)  # {1..9: token_id}

    # 출력 파일 설정
    adapter_name = Path(adapter_dir).name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{adapter_name}_inference.csv"

    fieldnames = [
        "sample_idx",
        "gen_pos",
        "label",
        "pred_even_tokens",
        "chosen_token",
        "chosen_token_id",
        "prob_1","prob_2","prob_3","prob_4","prob_5","prob_6","prob_7","prob_8","prob_9",
    ]

    # CSV 저장
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, ex in enumerate(tqdm(ds, desc="Running inference")):
            instruction = ex.get("instruction", "")
            label = ex.get("label", ex.get("output", ""))

            enc = tokenizer(
                instruction,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
                padding=False,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(model.device)
            attn_mask = enc["attention_mask"].to(model.device)

            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy
                temperature=0.0,
                top_p=1.0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            gen_ids = gen_out.sequences[:, input_ids.size(1):]  # [1, gen_len]
            gen_len = gen_ids.size(1)
            scores = gen_out.scores

            # 짝수번째(1-indexed: 2,4,6,...) 토큰만 결합
            even_tokens = []
            for p in range(gen_len):
                if (p + 1) % 2 == 0:
                    even_tokens.append(tokenizer.decode([int(gen_ids[0, p])], skip_special_tokens=True))
            pred_even_tokens = "".join(even_tokens).strip()

            # 각 스텝별 확률 계산 및 CSV 기록
            for p in range(gen_len):
                logits = scores[p].squeeze(0)            # [vocab]
                probs = torch.softmax(logits, dim=-1)    # [vocab]

                chosen_id = int(gen_ids[0, p].item())
                chosen_tok = tokenizer.decode([chosen_id], skip_special_tokens=True)

                row = {
                    "sample_idx": idx,
                    "gen_pos": p + 1,
                    "label": label,
                    "pred_even_tokens": pred_even_tokens,
                    "chosen_token": chosen_tok,
                    "chosen_token_id": chosen_id,
                }
                # 숫자 1~9의 확률
                for d in range(1, 10):
                    tid = digit_id_map[d]
                    row[f"prob_{d}"] = float(probs[tid].item())
                writer.writerow(row)

    print(f"\n추론 완료 및 CSV 저장: {out_path}")





@torch.inference_mode()
def run_inference(model, tokenizer, test_dataset, out_dir: str):
    """
    학습 완료된 model 그대로 사용
    """
    out_dir = Path(out_dir) / "inference_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inference_results.csv"

    digit_id_map = build_digit_token_id_map(tokenizer)
    max_seq_length = tokenizer.model_max_length

    fieldnames = [
        "sample_idx", "gen_pos", "label", "gen_label",
        "chosen_token", "chosen_token_id",
    ] + [f"prob_{i}" for i in range(1, 10)]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, ex in enumerate(tqdm(test_dataset, desc="Running inference")):
            instruction = ex.get("instruction", "")
            label = ex.get("label", ex.get("output", ""))

            enc = tokenizer(
                instruction,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
                padding=False,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(model.device)
            attn_mask = enc["attention_mask"].to(model.device)

            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=16,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            gen_ids = gen_out.sequences[:, input_ids.size(1):]
            gen_len = gen_ids.size(1)
            scores = gen_out.scores

            even_tokens = [
                tokenizer.decode([int(gen_ids[0, p])], skip_special_tokens=True)
                for p in range(gen_len) if (p + 1) % 2 == 0
            ]
            pred_even_tokens = "".join(even_tokens).strip()

            for p in range(gen_len):
                logits = scores[p].squeeze(0)
                probs = torch.softmax(logits, dim=-1)
                chosen_id = int(gen_ids[0, p].item())
                chosen_tok = tokenizer.decode([chosen_id], skip_special_tokens=True)
                row = {
                    "sample_idx": idx,
                    "gen_pos": p + 1,
                    "label": label,
                    "pred_even_tokens": pred_even_tokens,
                    "chosen_token": chosen_tok,
                    "chosen_token_id": chosen_id,
                }
                for d in range(1, 10):
                    row[f"prob_{d}"] = float(probs[digit_id_map[d]].item())
                writer.writerow(row)
    print(f"Inference complete. Results saved to {out_path}")
    return out_path


class ScoreDelayStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length, target_sequence="### Score:\n", delay_tokens=16):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.target_sequence = target_sequence
        self.delay_tokens = delay_tokens
        self.found_step = None  # 타겟 문구가 발견된 시점의 인덱스 저장

    def __call__(self, input_ids, scores, **kwargs):
        # 프롬프트 부분을 제외하고 순수 생성된 토큰만 가져옴
        generated_ids = input_ids[0, self.prompt_length:]
        current_gen_len = len(generated_ids)

        # A. 이미 타겟을 찾은 상태라면? -> delay_tokens 만큼 지났는지 확인
        if self.found_step is not None:
            if (current_gen_len - self.found_step) >= self.delay_tokens:
                return True # 정지
            return False # 계속 생성

        # B. 아직 못 찾았다면? -> 텍스트 디코딩하여 검색
        decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if self.target_sequence in decoded_text:
            self.found_step = current_gen_len # 발견 시점 기록
        
        return False


@torch.inference_mode()
def run_inference_mtl(model, tokenizer, test_dataset, out_dir: str):
    """
    Feedback -> Score 구조 추론 함수
    """
    out_dir = Path(out_dir) / "inference_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inference_results.csv"

    digit_id_map = build_digit_token_id_map(tokenizer)
    max_seq_length = tokenizer.model_max_length

    # [수정] 헤더 변경: gen_feedback, label_feedback 추가
    # pred_even_tokens는 피드백 구조에서 의미가 모호하므로 제거하거나 token_text로 대체
    fieldnames = [
        "sample_idx", "gen_pos", "gen_label", "label", "label_feedback", 
        "gen_feedback", "chosen_token", "chosen_token_id"
    ] + [f"prob_{i}" for i in range(1, 10)]

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, ex in enumerate(tqdm(test_dataset, desc="Running inference")):
            instruction = ex.get("instruction", "")
            label = ex.get('score_output', "").replace("\n### Score:\n", "")
            
            # [수정] 정답 피드백 데이터 로드 (데이터셋의 키 이름이 'feedback'이라고 가정)
            label_feedback = ex.get("feedback_output", "").replace("\n### Feedback:\n", "").strip() 

            enc = tokenizer(
                instruction,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
                padding=False,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(model.device)
            attn_mask = enc["attention_mask"].to(model.device)

            # [수정] Stopping Criteria 설정
            prompt_len = input_ids.shape[1]
            criteria_instance = ScoreDelayStoppingCriteria(
                tokenizer=tokenizer, 
                prompt_length=prompt_len, 
                target_sequence="### Score:\n", 
                delay_tokens=16
            )
            stopping_criteria = StoppingCriteriaList([criteria_instance])

            # [수정] max_new_tokens 대폭 증가 (피드백 생성을 위해)
            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=1342,  # 피드백 길이에 맞춰 넉넉하게 설정
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria # Criteria 적용
            )

            gen_ids = gen_out.sequences[:, prompt_len:]
            gen_len = gen_ids.size(1)
            scores = gen_out.scores

            full_gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            # [핵심 로직] Score 부분 인덱싱 범위 설정 및 피드백 분리
            target_seq = "### Score:\n"
            
            if criteria_instance.found_step is not None:
                # Case 1: 구분자가 발견됨 -> 발견된 지점 바로 뒤 16개 토큰 보기
                start_pos = criteria_instance.found_step
                end_pos = min(gen_len, start_pos + 16)
                
                # 생성된 텍스트에서 피드백 부분만 추출
                parts = full_gen_text.split(target_seq)
                gen_feedback = parts[0].strip()
            else:
                # Case 2: 구분자가 없음 -> 마지막 16개 토큰 보기
                start_pos = max(0, gen_len - 16)
                end_pos = gen_len
                
                # 전체를 피드백으로 간주
                gen_feedback = full_gen_text.strip()
            
            gen_label = tokenizer.decode(gen_ids[0, start_pos:end_pos], skip_special_tokens=True).strip()

            # [수정] 선택된 범위(Score 부분)에 대해서만 CSV 기록
            for p in range(start_pos, end_pos):
                logits = scores[p].squeeze(0)
                probs = torch.softmax(logits, dim=-1)
                chosen_id = int(gen_ids[0, p].item())
                chosen_tok = tokenizer.decode([chosen_id], skip_special_tokens=True)
                
                row = {
                    "sample_idx": idx,
                    "gen_pos": p + 1, # 전체 생성 길이 기준 인덱스
                    "label": label,
                    "gen_label": gen_label,
                    "label_feedback": label_feedback,
                    "gen_feedback": gen_feedback,
                    "chosen_token": chosen_tok,
                    "chosen_token_id": chosen_id,
                }
                
                for d in range(1, 10):
                    row[f"prob_{d}"] = float(probs[digit_id_map[d]].item())
                
                writer.writerow(row)

    print(f"Inference complete. Results saved to {out_path}")
    return out_path
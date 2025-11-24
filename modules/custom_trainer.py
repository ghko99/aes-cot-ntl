
from transformers import Trainer
from typing import Any, Dict, Union
from .wasserstein_number_token_loss import WassersteinNumberTokenLoss
from .number_token_loss import NumberTokenLoss
import torch

class CustomTrainer(Trainer):
    """
    Hugging Face Trainer 확장 버전:
    CE loss + ntl_weight * Wasserstein number token loss 결합
    """
    def __init__(
        self,
        *args,
        ntl_weight: float = 0.3,
        num_tokenizer=None,
        order_numbers=None,
        loss_type: str = "mse",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ntl_weight = float(ntl_weight)
        self.num_tokenizer = num_tokenizer
        self.order_numbers = order_numbers

        # device 지정 (Trainer가 관리하는 device로 통일)
        device = self.args.device
        
        vocab_size = self.model.config.vocab_size
        # Wasserstein 손실 로드 (지연 import)
        if loss_type == "was":
            self.ntl_criterion = WassersteinNumberTokenLoss(
                vocab_size=vocab_size,
                device=device,
                order_numbers=self.order_numbers,
                tokenizer=self.num_tokenizer
            )
        elif loss_type == "mse":
            self.ntl_criterion = NumberTokenLoss(
                tokenizer=self.num_tokenizer,
                vocab_size=vocab_size,
                device=device,
                loss_function=torch.nn.functional.mse_loss
            )
        else:
            raise ValueError(f"지원하지 않는 loss_type: {loss_type}")
    @staticmethod
    def _to_serializable(v: Any) -> Any:
        # 텐서 -> float/int, dict/list도 재귀 처리
        if isinstance(v, torch.Tensor):
            # 스칼라 텐서만 로그해야 함
            if v.numel() == 1:
                return v.detach().to("cpu", non_blocking=True).item()
            # 스칼라가 아니면 평균 등으로 스칼라화해서 기록(선택)
            return v.detach().to("cpu", non_blocking=True).float().mean().item()
        if isinstance(v, dict):
            return {k: CustomTrainer._to_serializable(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [CustomTrainer._to_serializable(x) for x in v]
        # numpy도 처리(혹시 있을 경우)
        try:
            import numpy as np
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
        except Exception:
            pass
        return v
    
    def _safe_log(self, metrics: Dict[str, Any]) -> None:
        self.log(self._to_serializable(metrics))

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, tuple]:
        """
        total_loss = CE loss (HF 기본) + ntl_weight * NTL loss
        """
        # ⚙️ 1. 기본 CE 로스 계산 (HF Trainer 내부 로직 사용)
        base_loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, **kwargs
        )

        # ⚙️ 2. logits 추출
        logits = outputs.get("logits", None)
        if logits is None:
            raise ValueError("Model outputs must include 'logits' to compute NTL loss.")

        
        # ⚙️ 3. Wasserstein NTL 손실 계산
        ntl_loss = self.ntl_criterion(logits, inputs["labels"])
        # ⚙️ 4. 총합 로스 계산
        total_loss = base_loss + self.ntl_weight * ntl_loss

        # ⚙️ 5. (선택) 로깅용 정보 저장
        if isinstance(outputs, dict):
            outputs["ce_loss"] = base_loss.detach()
            outputs["ntl_loss"] = ntl_loss.detach()
        # ---- 여기서 텐서 그대로 넣지 말고 float 변환해서 로깅! ----
        
        self._safe_log({
            "loss_total": total_loss,
            "loss_ce": base_loss if base_loss is not None else 0.0,
            "loss_ntl": ntl_loss if ntl_loss is not None else 0.0,
            "ntl_weight": float(self.ntl_weight),            # <- 반드시 float!
            "epoch": float(self.state.epoch or 0), # <- float로
        })

        # ⚙️ 6. 반환
        return (total_loss, outputs) if return_outputs else total_loss
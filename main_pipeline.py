import argparse
from modules.train_module import train_model
from modules.inference_module import run_inference_mtl
from modules.evaluate_module import evaluate_results
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Baseline 학습 (NTL 미적용)")
    parser.add_argument("--ratio", type=float, default=1.0, help="데이터 비율 (0.1=10%)")
    parser.add_argument("--ntl_weights", type=float, default=2.0, help="NTL loss 가중치")
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "was"], help="NTL 손실 함수 유형")
    parser.add_argument("--device_id", type=int, help="사용할 GPU 장치 ID")
    


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    # 1️⃣ Training (Trainer, test_ds, tokenizer, output_dir)
    trainer, test_ds, tokenizer, output_dir = train_model(
        baseline=args.baseline,
        ntl_weight=args.ntl_weights,
        ratio=args.ratio,
        loss_type = args.loss_type
    )

    # 2️⃣ Inference (best model already loaded)
    csv_path = run_inference_mtl(
        model=trainer.model,
        tokenizer=tokenizer,
        test_dataset=test_ds,
        out_dir=output_dir
    )

    # 3️⃣ Evaluation
    evaluate_results(str(csv_path), save_dir=output_dir)

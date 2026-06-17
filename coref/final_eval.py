import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoTokenizer

from coref.model import CorefModel
from coref.dev_eval import evaluate_docs
from coref.scorer import bootstrap_ci

HOLDOUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test_holdout_preco.json")
BASELINES = ["sentence", "window5", "fulldoc"]


class ListSplit:
    def __init__(self, docs):
        self.docs = docs

    def __getitem__(self, i):
        return self.docs[i]

    def __len__(self):
        return len(self.docs)


def main():
    with open(HOLDOUT_PATH) as f:
        payload = json.load(f)
    docs = payload["docs"]
    assert len(docs) == 500, f"sealed holdout must have 500 docs, got {len(docs)}"
    split = ListSplit(docs)
    doc_indices = list(range(len(docs)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    report = {}
    for baseline in BASELINES:
        ckpt_path = f"checkpoints/coref_{baseline}_best.pt"
        if not os.path.exists(ckpt_path):
            print(f"[skip] no checkpoint for {baseline} at {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = CorefModel().to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        result = evaluate_docs(model, split, doc_indices, baseline, tokenizer, device, with_buckets=True)
        ci = bootstrap_ci(result["doc_scores"], n_boot=1000, seed=0)

        report[baseline] = {
            "dev_selected_step": ckpt.get("step"),
            "dev_conll_f1_at_selection": ckpt.get("dev_conll_f1"),
            "sealed_test_agg": result["agg"],
            "sealed_test_ci_95": ci,
            "cross_sentence_buckets": result["buckets"],
            "n_docs": len(docs),
        }
        print(f"\n=== {baseline} (sealed test, n=500) ===")
        print(f"  CoNLL F1: {result['agg']['conll_f1']:.4f}  (95% CI {ci['conll_f1'][0]:.4f}-{ci['conll_f1'][1]:.4f})")
        for m in ["muc", "b3", "ceafe"]:
            print(f"  {m}: P={result['agg'][m]['precision']:.4f} R={result['agg'][m]['recall']:.4f} "
                  f"F1={result['agg'][m]['f1']:.4f} (95% CI {ci[m][0]:.4f}-{ci[m][1]:.4f})")
        print("  cross-sentence pairwise link F1 by distance bucket:")
        for name, bm in result["buckets"].items():
            print(f"    {name} sents: P={bm['precision']:.4f} R={bm['recall']:.4f} F1={bm['f1']:.4f} "
                  f"(n_gold_pairs={bm['n_gold_pairs']}, n_pred_pairs={bm['n_pred_pairs']})")
        del model
        torch.cuda.empty_cache()

    with open("results/coref_baselines_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote results/coref_baselines_report.json")


if __name__ == "__main__":
    main()

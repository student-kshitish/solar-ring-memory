import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
from datasets import load_dataset

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test_holdout_preco.json")


def main():
    ds = load_dataset("coref-data/preco")
    val = ds["validation"]
    assert len(val) == 500, f"expected 500 validation docs, got {len(val)}"
    docs = [val[i] for i in range(len(val))]
    payload = {
        "meta": {
            "source": "coref-data/preco validation split",
            "n_docs": len(docs),
            "warning": "SEALED TEST SET. Never read during training, dev evaluation, or checkpoint selection.",
        },
        "docs": docs,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f)
    print(f"Wrote {len(docs)} sealed docs to {OUT_PATH}")


if __name__ == "__main__":
    main()

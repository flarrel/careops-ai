import importlib.util
import unittest
from collections import Counter

from careops.corpus import DEV_CLAIMS, read_jsonl

HAS_DATA = all(importlib.util.find_spec(name) for name in ("numpy", "PIL", "pydantic"))
HAS_MODEL = HAS_DATA and all(
    importlib.util.find_spec(name) for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_DATA, "Install the finetune extra")
class RoutingDataTests(unittest.TestCase):
    def test_split_groups_and_labels_are_disjoint_and_balanced(self):
        from careops.finetune.fixtures import LABELS, examples

        splits = examples(read_jsonl(DEV_CLAIMS))
        claimed = set()
        for name, rows in splits.items():
            claim_ids = {row.claim_id for row in rows}
            self.assertTrue(claimed.isdisjoint(claim_ids))
            claimed.update(claim_ids)
            self.assertEqual(len({row.sample_id for row in rows}), len(rows))
            self.assertEqual(Counter(row.label for row in rows),
                             dict.fromkeys(LABELS, len(rows) // len(LABELS)), name)
            self.assertTrue(all(row.claim_id not in row.prompt for row in rows))
            self.assertTrue(all(row.prompt != row.alternate for row in rows))
        self.assertEqual([len(rows) for rows in splits.values()], [144, 48, 48])

    def test_records_require_an_explicit_synthetic_label(self):
        from careops.finetune.fixtures import examples

        rows = read_jsonl(DEV_CLAIMS)
        rows[0]["synthetic"] = False
        with self.assertRaisesRegex(ValueError, "only synthetic"):
            examples(rows)


@unittest.skipUnless(HAS_MODEL, "Install the finetune extra")
class RoutingMetricsTests(unittest.TestCase):
    def test_unknown_output_counts_as_incorrect_with_invalid_column(self):
        from careops.finetune.fixtures import variants
        from careops.finetune.training import evaluate

        rows = list(variants(read_jsonl(DEV_CLAIMS)[0], 0))
        outputs = [row.label for row in rows]
        outputs[0] = "request payment approval"
        result = evaluate(rows, outputs)
        self.assertEqual(result["exact_match"], 0.75)
        self.assertEqual(result["macro_f1"], 0.75)
        self.assertEqual(result["invalid_outputs"], 1)
        self.assertEqual(result["confusion_with_invalid_column"][0][-1], 1)

# Policy corpus and evaluation inputs

The [source manifest](../data/manifests/policy-sources.json) pins ten Indonesian regulations from official catalogs and PDF archives. Each entry records the exact edition, catalog status at collection, byte count, retrieval date, and SHA-256. The JKN base regulation and its three amendments are one document family. Cite the edition named by the question. The historical INA-CBG edition is marked revoked and is included for version comparisons. The collection is a snapshot of the selected documents.

The raw PDFs and extracted text stay local. The pinned files total 34,490,885 bytes. From the repository root with Python 3.12, fetch the PDFs and reproduce the audit.

```sh
python3 -m pip install -e '.[data]'
PYTHONPATH=src python3 -m careops.corpus fetch
PYTHONPATH=src python3 -m careops.corpus audit > /tmp/careops-corpus-audit.json
diff -u data/corpus-audit.json /tmp/careops-corpus-audit.json
```

Fetch validates the PDF signature, byte count, and checksum before saving under `data/raw/`. It reuses matching cached files and reports errors when upstream bytes or cached files differ. Audit can run offline on that cache. Use `--raw DIR` with either command to select a different location.

## Extraction audit

The [audit](../data/corpus-audit.json) measures 340 physical PDF pages across four document families. It uses native text extraction with pypdf 7.1.1, Unicode NFKC normalization, and whitespace folding. Citations use one-based physical PDF page numbers plus an article or section. The audit covers pages 1–35 of the 2023 tariff PDF and pages 1–6 of the historical INA-CBG PDF. The large tariff schedule beyond the inspected pages is outside the benchmark.

No selected page extracted to an empty string, but 88 pages yielded fewer than 100 characters. The quality regulation's pages 7–93 are dominated by footer and page-number text in this extractor. The tariff PDF's page 31 also yielded only 17 characters. These pages do not supply any evaluation citation. The 2024 amendment has detectable OCR errors even on longer pages. Use page-image checks for any later cases relying on that material. Native extraction does not preserve table structure, so the corpus should abstain on unsupported table lookups. Exact normalized duplicate-page checks found zero duplicates within each file. The split treats amendments as related even when the pages differ byte for byte.

## Synthetic claims and evaluation split

The [development fixtures](../data/samples/claims-development.jsonl) contain 20 synthetic claims across four templates. Their amounts range from Rp160,000 to Rp2,630,000, with a median of Rp249,500. Five examples per template allow predictable fixture coverage. Counts and amounts reflect the authored examples. The `template` field is for grouping and must be omitted from tool responses and model inputs. All service codes are fictitious.

Each split contains 25 authored cases with 15 source-backed questions, three unanswerable questions, two instruction-boundary cases, and five synthetic claim tool cases. Gold expectations include references for policy answers and specific lookup fields for tool cases. Each cited passage was checked during authoring. The labels have no independent clinical review. Measure results on these fixtures as engineering tests.

Development cases cite payment and facility-quality sources. The held-out cases use insurance-governance and medical-records sources and disjoint claim templates. Develop against the development cases. Held-out source documents can be searched when running the final retrieval evaluation. Keep held-out questions, labels, and claims out of the index and of development tuning. The public audit reads only the public development fixtures and source PDFs.

The private final set is pinned by [the manifest digest](../evals/sealed.sha256). The sealed manifest pins the corpus manifest and the exact case and claim fixture bytes. When the private directory is explicitly available for final evaluation, verify it with:

```sh
PYTHONPATH=src python3 -m careops.corpus verify-seal /path/to/sealed-eval
```

Verification checks file integrity, fixture references, and overlap across document families, claim templates, IDs, and exact normalized questions. Freeze prompts, thresholds, and routing before opening the final labels.

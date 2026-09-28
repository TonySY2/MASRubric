The criterion-bank JSON is bundled here. The embedding cache is generated
separately because precomputed vectors can be large:

```bash
python test/metrics_pool/two_pool/embed_metrics-trigger.py \
  --input_file test/metrics_pool/two_pool/deduped-mixed_metrics_two_pool.json \
  --output_cache_file test/metrics_pool/two_pool/deduped-mixed_two_pool-trigger.jsonl
```

For the non-deduplicated ablation pool:

```bash
python test/metrics_pool/two_pool/embed_metrics-trigger.py \
  --input_file test/metrics_pool/two_pool/mixed_metrics_two_pool.json \
  --output_cache_file test/metrics_pool/two_pool/mixed_embeddings_cache_two_pool.jsonl
```

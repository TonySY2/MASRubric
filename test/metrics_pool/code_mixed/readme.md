The mixed code criterion-bank JSON is bundled here. The embedding cache is
generated separately because precomputed vectors can exceed GitHub's single-file
size limit:

```bash
python test/metrics_pool/two_pool/embed_metrics-trigger.py \
  --input_file test/metrics_pool/code_mixed/deduplicated_metrics_pool.json \
  --output_cache_file test/metrics_pool/code_mixed/deduplicated_embeddings-trigger.jsonl
```

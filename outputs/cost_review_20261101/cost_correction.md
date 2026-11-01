# Source-based cost correction

Reanalysis date: 1 November 2026 (Australia/Sydney). No model inference or paid API requests were made. The original results remain unchanged.

## Cost basis

The primary local-model estimate prices the recorded successful generation elapsed time at Lambda’s published one-GPU H100 PCIe 80GB rate of US$3.29/hour. Source: https://lambda.ai/pricing (accessed 1 November 2026). The quoted configuration includes 26 vCPUs, 225 GiB RAM and 1 TiB SSD; applicable taxes are additional. This is a reference rental-equivalent, generation-only estimate, not a measurement of the university server bill or a benchmark run on Lambda hardware.

The sensitivity case uses US$2.89/hour from Runpod’s central GPU Pods pricing table, https://www.runpod.io/pricing (accessed 1 November 2026; page updated 28 October). Its separate GPU-model FAQ contains a conflicting older number; the central table is used consistently.

GPT token rates are independently verified against https://developers.openai.com/api/docs/models/gpt-5.6-sol : US$4 per million input tokens, US$0.40 cached input, US$20 output. The page also specifies the long-context and cache-write multipliers used by the existing manifest. Repricing all 552 saved calls reproduces US$11.4481088, including validation and test, at those rates. This is a usage-priced estimate, not an invoice.

## Method and scope

Local cost = saved successful generation elapsed seconds / 3600 × the sourced hourly reference rate. GPT cost = returned token usage priced per request at the verified rates. A cascade pays local generation cost for every cheap-model case plus the strong-model cost for each escalated case. Three responses per model and case remain included.

Both the original fixed policies and a fresh validation-only policy selection were evaluated. Re-selection uses the same three confidence signals, threshold grid, calibration procedure, 95% strong-only validation accuracy requirement and tie-breakers. All eight primary selections remain unchanged. The 74-case test set has already been inspected, so this is a disclosed exploratory cost-basis reanalysis, not a new confirmatory experiment.

The time records exclude model loading, downloads, idle allocation, storage, failed attempts and taxes. They measure generation elapsed time, not GPU-kernel active time or energy. Real deployment scheduling, caching and overhead can change the total. For local-to-local pairs the hourly rate cancels from relative savings because all models use the same GPU-hour rate. For GPT cascades the local rental reference changes the trade-off against API charges.

## Corrected headline values

| Cascade | Previous assumed-token saving | Sourced generation-cost saving | Correct cases, cascade / strong |
|---|---:|---:|---:|
| Llama 8B to Llama 70B | 16.3% | 2.5% | 21 / 21 |
| Llama 8B to Qwen 72B | 16.5% | 10.1% | 18 / 22 |
| Qwen 7B to Llama 70B | 0.8% | -4.4% | 18 / 21 |
| Qwen 7B to Qwen 72B | 18.8% | 18.4% | 20 / 22 |
| Llama 8B to GPT | 32.5% | 15.5% | 27 / 31 |
| Qwen 7B to GPT | 10.1% | -1.5% | 27 / 31 |
| Llama 8B to Qwen 27B | -4.0% | -11.2% | 25 / 27 |
| Qwen 7B to Qwen 27B | 2.6% | 9.0% | 22 / 27 |

Under the primary reference rate, Llama 8B/Llama 70B matches the strong-only total of 21 correct cases with a 2.5% cost reduction. Llama 8B/GPT gets 27 correct cases versus 31 for GPT alone, with a 15.5% cost reduction.

## Files

[The cost reanalysis](sourced_cost_reanalysis.json) records the sources, price assumptions and results for both reference rates.

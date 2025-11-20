# Progress Report

So far, I have implemented a compact decoder-only transformer model based on our Assignment 4 architecture. The primary modification was adapting the data pipeline to train on the WikiText-2 corpus. After confirming that the model trains reliably on this dataset, I designed and ran two controlled experiments in Google Colab to investigate how architectural and optimization choices influence performance.

## Experiment 1: Model Size and Performance
I trained three variants of the model with increasing depth and dimensionality:

- **2 layers, 64-dim hidden size**
- **4 layers, 128-dim hidden size**
- **6 layers, 256-dim hidden size**

The results show a clear relationship between model size and performance. Larger models achieved lower training loss and higher next-token accuracy, though at the cost of longer training time and increased GPU memory usage. These findings are consistent with expected scaling behavior and provide a baseline for further experimentation.

## Experiment 2: Optimizer Comparison
To evaluate how optimizer choice affects training efficiency and stability, I compared **SGD**, **Adam**, and **AdamW** under identical training conditions. As expected, SGD trained the fastest per step but produced significantly worse accuracy and higher loss. Adam and AdamW performed nearly identically, both converging faster and to better final values than SGD. This suggests that adaptive optimizers remain preferable for small transformer training in constrained settings.

## Next Steps
Before completing Part 1, I may run one additional experiment—for example testing a different positional-encoding scheme or introducing dropout—to deepen the analysis. After that, I will move on to Part 2, where I plan to select a compact pretrained model from Hugging Face (likely DistilGPT-2 or T5-small) and apply it to a downstream task, then compare its performance and behavior to my mini transformer. I will conclude with evaluation metrics, interpretability visualizations, and a discussion linking results to course concepts.

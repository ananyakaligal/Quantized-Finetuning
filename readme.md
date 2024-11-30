# Efficient Fine-tuning of Quantized Language Models for Medical Domain Adaptation

This repository contains the implementation and results of a study on efficient fine-tuning of large language models (LLMs) for the medical domain. By combining 4-bit quantization and Parameter-Efficient Fine-Tuning (PEFT) techniques, such as LoRA (Low-Rank Adaptation), we significantly reduce computational requirements while maintaining high model performance. The study uses GEMMA-2B, a large-scale model, and adapts it to the medical domain using the MedMCQA dataset. This work achieves impressive results with reduced resource consumption while ensuring model accuracy and efficiency.

## 🚀 Key Features
- **4-bit Quantization**: Implemented using the `bitsandbytes` library, enabling a 75% reduction in model size while maintaining performance.
- **Low-Rank Adaptation (LoRA)**: Optimized fine-tuning targeting key model layers for efficiency.
- **Integrated Pipeline**: A cohesive workflow combining quantization and fine-tuning for medical domain adaptation.

## 📊 Results Summary
| Metric                 | Before Quantization | After Quantization | Improvement (%) |
|------------------------|---------------------|--------------------|------------------|
| Model Size (MB)        | 2009.06            | 398.72            | -80.14%         |
| Medical Accuracy (%)   | 22.11              | 33.35             | +50.84%         |
| Memory Usage (GB)      | 40                 | 10                | -75%            |
| Inference Speed (ms)   | 200                | 150               | +25%            |

## 📚 Dataset: MedMCQA
The study uses the **MedMCQA** dataset, which consists of:
- **187,125 multiple-choice medical questions** spanning 14 major subjects.
- Training, validation, and test splits of 177,125, 5,000, and 5,000 questions, respectively.
- Rich metadata for analysis and evaluation.

## 🛠 Techniques
### Quantization
- **4-bit Quantization**:
  - Reduces the memory footprint while preserving accuracy.
  - Implements advanced memory management strategies such as double quantization and `bfloat16` compute types.

### Fine-tuning
- **LoRA (Low-Rank Adaptation)**:
  - Decomposes weight updates into low-rank matrices.
  - Optimizes only essential parameters, reducing training overhead.

### Training
- Gradient checkpointing, dynamic batch sizing, and cosine learning rate schedules.
- Hardware: NVIDIA A100 GPUs with an optimized PyTorch and Transformers stack.

## 📈 Performance
- Achieved a **final validation loss of 1.36**.
- Improved performance across question types, including basic, intermediate, and advanced medical queries.
- Robust ablation studies comparing quantization and PEFT methods.

## 🏗 System Architecture
1. **Data Preprocessing**: Tokenization and preparation for medical QA.
2. **Quantization Module**: Compresses model weights for efficiency.
3. **Fine-tuning Module**: Adaptation using LoRA.
4. **Evaluation Module**: Monitors performance metrics and logs results.

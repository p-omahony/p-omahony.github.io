---
title: "Deepseek-R1: Parameter-Efficient Fine-Tuning on AWS"
date: 2025-01-02
categories: [llm, deep learning]
tags: [llm, deep learning]     # TAG names should always be lowercase
---

## Introduction   

The goal of this project is to fine-tune the DeepSeek Distilled Llama R1 model using 4-bit quantization provided by Unsloth. This will also give us the opportunity to thoroughly revisit key concepts such as LoRA, QLoRA, and quantization. For training and model serving, we will leverage AWS services to ensure scalability and efficiency.

## A bit of background
### 1. Deepseek-R1

DeepSeek-R1 was developed by the Chinese AI startup DeepSeek to enhance performance in complex reasoning tasks, including mathematics, coding, and logical inference. The model's training involved a multi-stage process:

TO_CHECK
- Supervised Fine-Tuning (SFT): Initially, the model was fine-tuned on a "cold-start" dataset comprising thousands of examples formatted to improve output readability and coherence.
- Reinforcement Learning (RL): Following SFT, the model underwent reinforcement learning using rule-based rewards to further enhance its reasoning capabilities.

/TO_CHECK

This training methodology enabled DeepSeek-R1 to achieve performance comparable to OpenAI's o1 model across various benchmarks. Notably, DeepSeek-R1 is open-source under the MIT License, allowing for widespread use and adaptation.

It is important to note that the objective of R1 was to enhance the reasoning capabilities of their previous models.

### 2. Distilled Models

Distilled models are compressed versions of larger machine learning models, created through a process called knowledge distillation. This technique involves training a smaller model (the student model) to mimic the performance of a larger, more complex model (the teacher model) while maintaining high accuracy and efficiency.

![LLM Distillation](../assets/images/fm-finetuning-distill.drawio.png)

In our case, we will use the `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` model. **Llama-8B** is one of Meta's smaller models, which has been further distilled to replicate the performance of the original **DeepSeek-R1** model. This distilled version retains much of the original model’s capabilities while being more efficient in terms of computational resources and memory usage.

## LoRA: Low-Rank Adaptation

Low-Rank Adaptation (LoRA) is a technique in machine learning designed to efficiently fine-tune large models for specific tasks without the need to retrain the entire model. This approach is particularly beneficial when working with large language models (LLMs) that have billions of parameters, as it significantly reduces computational costs and training time.

Traditional fine-tuning methods involve adjusting all parameters of a pre-trained model, which can be resource-intensive. LoRA addresses this by introducing low-rank matrices into the model's architecture. These matrices are smaller and require fewer parameters to train, allowing for efficient adaptation to new tasks. The original model's parameters remain unchanged, and the low-rank matrices are added to the existing weights during inference, enabling the model to perform specific tasks effectively.

If we have a look at regular finetuning:

![Regular Finetuning](../assets/images/fm-finetuning-regular-finetuning.drawio.png)

The weight update is obtained during regular backpropagation and is typically calculated this way:

$$\Delta W = \alpha \left(- \nabla L_W \right)$$

Where $$\alpha$$ is the learning rate and $$\nabla L_W$$ is the gradient of the loss with respect to $$W$$.

When training fully connected (dense) layers in neural networks, weight matrices typically have full rank, meaning no redundant rows or columns.
In contrast, low rank means the matrix has redundant components.
For example, let be a matrix $$A \in \mathbf{R}^{4 \times 4}$$ :

![A](../assets/images/fm-finetuning-A.drawio.png)

Here we can see that $$a_1 = a_2 + \frac{1}{2}a_4$$. It means that $$A$$ is not full rank and that information is stored redundantly. So we could drop one of these rows and so reduce the dimensionality of $$A$$.

While pretrained model weights have full rank for their original tasks, the LoRA authors note that large language models exhibit a low "intrinsic dimension" when adapted
 to new tasks (Aghajanyan et al., 2020). This means the data can be approximated by a lower-dimensional space, allowing the new weight matrix to be decomposed into smaller
 matrices without significant loss of information.

![LoRA Finetuning](../assets/images/fm-finetuning-lora-finetuning.drawio.png)

So now our weight update is $$\Delta W = W_AW_B$$ with $$W \in \mathbf{R}^{m \times p}$$, $$W_A \in \mathbf{R}^{m \times r}$$ and $$W_B \in \mathbf{R}^{r \times p}$$; $$r$$ being the new rank.
Typically, $$r$$ is predetermined before training and remains fixed, while the decomposition is learned during training.

## Metrics

## Training

## Evaluation

## References

<link rel="icon" type="image/png" href="../../assets/img/favicons/favicon-96x96.png" sizes="96x96" />
<link rel="icon" type="image/svg+xml" href="../../assets/img/favicons/favicon.svg" />
<link rel="shortcut icon" href="../../assets/img/favicons/favicon.ico" />
<link rel="apple-touch-icon" sizes="180x180" href="../../assets/img/favicons/apple-touch-icon.png" />

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

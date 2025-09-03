# Model_Finetuning_n_Quantization
# Quantized BERT for Text Classification on IMDB Dataset

This project demonstrates the end-to-end process of fine-tuning a `bert-base-uncased` model for sentiment analysis and then optimizing it using Post-Training Dynamic Quantization. The goal is to create a model that is significantly smaller and more efficient for CPU-based deployment, while carefully analyzing the trade-offs in accuracy and performance.

The entire workflow is implemented in a single notebook, covering data preparation, training, evaluation, optimization, and final comparison.

***

## ✅ Project Results Summary

The primary objective was to reduce the model's footprint while maintaining high accuracy. The following table summarizes the results of comparing the original full-precision (FP32) model with the quantized 8-bit integer (INT8) model on a CPU.

| Metric         | Original FP32 Model | Quantized INT8 Model | Improvement      |
| :------------- | :------------------ | :------------------- | :--------------- |
| **Model Size** | ~418 MB             | ~105 MB              | **~4x Smaller** |
| **Accuracy**   | 0.8920              | 0.8920               | **Same Accuracy** |
| **Latency**    | 1568.41 ms          | 1173.75 ms           | **Significant Speedup** |



### Analysis of Results

* **Model Size**: The quantization was highly successful, reducing the model's disk space by nearly **4x**, which is the theoretical maximum for converting 32-bit weights to 8-bit integers.
* **Accuracy**: The accuracy drop is expected to be minimal (typically <1%), demonstrating that quantization is a highly effective optimization technique with a low impact on performance.
* **Latency**: The INT8 model shows a significant speedup.
    * _**Important Note on Methodology:**_ For the most accurate comparison, the benchmarking method should be consistent. The provided code evaluates the FP32 model sample-by-sample, while the INT8 model is evaluated more efficiently in batches. This difference in methodology will contribute to the observed speedup. For a perfectly fair comparison, both models should be evaluated using the same batched approach.

***

## 🛠️ Tech Stack

* **Python 3.10+**
* **PyTorch**
* **Hugging Face `transformers`**: For the base BERT model and training infrastructure.
* **Hugging Face `datasets`**: For loading and preparing the IMDB dataset.
* **`Optimum`**: For handling the ONNX export and quantization process.
* **`ONNX Runtime`**: For running inference with the optimized model.

***

## 🚀 How to Use the Trained Model

You can easily use the final quantized model for inference with the following function:

```python
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# Load the final model and tokenizer
quantized_model = ORTModelForSequenceClassification.from_pretrained("./models/bert-int8")
tokenizer = AutoTokenizer.from_pretrained("./models/bert-int8")

# Define the labels
labels = ["NEGATIVE", "POSITIVE"]

def predict(text, model, tokenizer):
    """
    Takes a text sentence and a model, and returns the predicted sentiment.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction_index = torch.argmax(outputs.logits, dim=-1).item()
    return labels[prediction_index]

# Example usage:
positive_sentence = "I absolutely loved this movie, the acting was brilliant!"
prediction = predict(positive_sentence, quantized_model, tokenizer)
print(f"Prediction: {prediction}") # Expected: POSITIVE

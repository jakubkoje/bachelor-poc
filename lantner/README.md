# Lantner

Lantner is a comprehensive Named Entity Recognition (NER) pipeline that combines the power of Large Language Models (LLMs) with fine-tuned GLiNER models for accurate entity extraction. It provides a complete workflow from data preprocessing to model evaluation.

## Features

- **Data Preprocessing**: Clean and prepare text data for entity extraction
- **LLM-based Annotation**: Use powerful language models to annotate text with entities
- **GLiNER Fine-tuning**: Fine-tune GLiNER models on your annotated data
- **Entity Extraction**: Extract entities from text using the fine-tuned model
- **Model Evaluation**: Evaluate model performance on test data
- **Long Document Support**: Automatic handling of long documents through chunking
- **Hardware Optimization**: Automatic device selection (CUDA, MPS, or CPU)

## Installation

```bash
pip install lantner
```

## Usage

```python
from lantner import Lantner

# Initialize the pipeline
pipeline = Lantner(
    gliner_model="gliner_small-v2.1",  # GLiNER model to use
    llm_model="gpt-4",                 # LLM model for annotation
    llm_api_key="your-api-key",        # Optional: API key for LLM
    llm_api_base="your-api-base"       # Optional: API base for LLM
)

# 1. Preprocess your data
texts = pipeline.preprocess(data_source="path/to/your/data.json")

# 2. Annotate the data with entities
annotated_data = pipeline.annotate(
    preprocessed_data=texts,
    labels=["person", "organization", "location"]
)

# 3. Fine-tune the GLiNER model
model = pipeline.fine_tune(annotated_data=annotated_data)

# 4. Use the model for inference
entities = pipeline.inference(
    model=model,
    text="Your text to analyze",
    labels=["person", "organization", "location"],
    threshold=0.7
)

# 5. Evaluate the model
metrics = pipeline.evaluate(model=model, test_data=test_data)
```

## Advanced Configuration

The `Lantner` class supports various configuration options:

```python
pipeline = Lantner(
    gliner_model="gliner_small-v2.1",  # GLiNER model name
    llm_model="gpt-4",                 # LLM model name
    llm_api_key="your-api-key",        # Optional: API key
    llm_api_base="your-api-base",      # Optional: API base
    # Additional GLiNER training parameters
    learning_rate=5e-6,
    weight_decay=0.01,
    num_train_epochs=3,
    # ... other training arguments
)
```

## Data Format

### Input Data
- JSON file containing an array of strings
- List of text strings

### Annotated Data Format
```python
[
    {
        "tokenized_text": ["token1", "token2", ...],
        "ner": [(start_idx, end_idx, "entity_type"), ...],
        "chunk_info": {
            "original_length": total_tokens,
            "chunk_start": start_position,
            "chunk_end": end_position
        }
    }
]
```

### Entity Extraction Output
```python
[
    {
        "entity": "extracted text",
        "types": ["entity_type1", "entity_type2", ...]
    }
]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

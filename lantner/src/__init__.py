import html
import json
import random
import re
from typing import Any, Optional, Union

import dspy
import markdownify
import pandas as pd
import torch
from bs4 import BeautifulSoup
from gliner import GLiNER
from gliner.data_processing.collator import DataCollator, DataCollatorWithPadding
from gliner.training import Trainer, TrainingArguments
from transformers import AutoTokenizer, EarlyStoppingCallback

# Maximum window size for tokenization and processing
MAX_WINDOW_SIZE = 384


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize the input text into a list of tokens.

    Args:
        text (str): The input text to tokenize

    Returns:
        list[str]: list of tokens extracted from the text
    """
    return re.findall(r"\w+(?:[-_]\w+)*|\S", text)


def extract_entities(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract entities from the input data and prepare it for model training.

    Args:
        data (list[dict[str, Any]]): list of dictionaries containing text and entity annotations

    Returns:
        list[dict[str, Any]]: list of processed examples with tokenized text and entity spans
    """
    all_examples = []

    for dt in data:
        # Attempt to extract entities; skip current record on failure
        try:
            tokens = tokenize_text(dt["text"])
            ents = [(k["entity"], k["types"]) for k in dt["entities"]]
        except Exception:
            continue

        # If tokens exceed max length, process in chunks
        if len(tokens) > MAX_WINDOW_SIZE:
            chunk_start = 0
            overlap = 50  # Number of tokens to overlap between chunks

            while chunk_start < len(tokens):
                chunk_end = min(chunk_start + MAX_WINDOW_SIZE, len(tokens))
                chunk_tokens = tokens[chunk_start:chunk_end]

                # Find entities in this chunk
                spans = []
                for entity in ents:
                    entity_tokens = tokenize_text(str(entity[0]))

                    # Skip entities longer than our chunk
                    if len(entity_tokens) > len(chunk_tokens):
                        continue

                    # Search for entity in this chunk
                    for i in range(len(chunk_tokens) - len(entity_tokens) + 1):
                        if (
                            " ".join(chunk_tokens[i : i + len(entity_tokens)]).lower()
                            == " ".join(entity_tokens).lower()
                        ):
                            for el in entity[1]:
                                if isinstance(el, str):
                                    spans.append(
                                        (
                                            i,
                                            i + len(entity_tokens) - 1,
                                            el.lower().replace("_", " "),
                                        )
                                    )

                # Only add chunks that contain entities
                if spans:
                    all_examples.append(
                        {
                            "tokenized_text": chunk_tokens,
                            "ner": spans,
                            "chunk_info": {
                                "original_length": len(tokens),
                                "chunk_start": chunk_start,
                                "chunk_end": chunk_end,
                            },
                        }
                    )

                # Move to next chunk with overlap
                chunk_start += MAX_WINDOW_SIZE - overlap
        else:
            # Original processing for sequences within limit
            spans = []
            for entity in ents:
                entity_tokens = tokenize_text(str(entity[0]))

                # Find the start and end indices of each entity in the tokenized text
                for i in range(len(tokens) - len(entity_tokens) + 1):
                    if (
                        " ".join(tokens[i : i + len(entity_tokens)]).lower()
                        == " ".join(entity_tokens).lower()
                    ):
                        for el in entity[1]:
                            spans.append(
                                (
                                    i,
                                    i + len(entity_tokens) - 1,
                                    el.lower().replace("_", " "),
                                )
                            )

            # Append the tokenized text and its corresponding named entity recognition data
            all_examples.append(
                {
                    "tokenized_text": tokens,
                    "ner": spans,
                    "chunk_info": {
                        "original_length": len(tokens),
                        "is_complete": True,
                    },
                }
            )

    return all_examples


class EntityExtractionSignature(dspy.Signature):
    """Extract named entities from the given passage. Look for the following entity types: {entities}.
    For each entity found, return its text and the type(s) it belongs to.
    Only extract entities that are explicitly mentioned in the text.
    Do not make assumptions or infer entities that are not directly stated."""

    passage: str = dspy.InputField()
    entities: list[dict[str, Any]] = dspy.OutputField(
        desc="list of entities with their types, formatted as [{'entity': 'entity name', 'types': ['type1', 'type2', ...]}]"
    )


def clean_html_text(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    extracted_text = soup.getText(separator=" ")
    extracted_text = html.unescape(extracted_text)
    cleaned_text = re.sub(r"\s+", " ", extracted_text)
    cleaned_text = re.sub(r"[^a-zA-Z\u00C0-\u017F0-9\s.,!?-]", "", cleaned_text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text


class Lantner:
    """
    A comprehensive Named Entity Recognition (NER) pipeline that combines LLM-based annotation
    with fine-tuned GLiNER models for accurate entity extraction.

    This class provides a complete workflow from data preprocessing to model evaluation,
    with support for long documents and automatic hardware optimization.
    """

    def __init__(
        self,
        gliner_model: str = "gliner_small-v2.1",
        llm_model: str = "gpt-4",
        llm_api_key: Optional[str] = None,
        llm_api_base: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Lantner instance for Named Entity Recognition (NER) pipeline.

        Args:
            gliner_model (str): Name of the GLiNER model to use for fine-tuning.
                Defaults to "gliner_small-v2.1".
                See https://github.com/urchade/GLiNER/blob/main/README_Extended.md for all supported models.
            llm_model (str): Name of the LLM model from litellm's supported models.
                Defaults to "gpt-4".
                See https://docs.litellm.ai/docs/providers for all supported models.
            llm_api_key (str, optional): API key for the LLM provider.
                Required if using a paid LLM service.
            llm_api_base (str, optional): API base URL for the LLM provider.
                Required for custom LLM deployments.
            **kwargs: Additional parameters for GLiNER fine-tuning.
                These will be passed to the TrainingArguments class.
        """
        self.gliner_model = gliner_model
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_api_base = llm_api_base
        self.training_args = kwargs

    def preprocess(self, data_source: Union[str, list[str]], **kwargs) -> list[str]:
        """
        Phase 1: Preprocess input data for entity extraction.

        This method handles data cleaning and preparation, including HTML content cleaning
        and file validation. It supports both JSON files and direct text input.

        Args:
            data_source (Union[str, list[str]]): Input data source. Can be either:
                - A path to a JSON file containing an array of strings
                - A list of text strings
            **kwargs: Additional preprocessing parameters (currently unused)

        Returns:
            list[str]: A list of preprocessed text strings ready for annotation

        Raises:
            ValueError: If the JSON file doesn't contain an array of strings
                       or if data_source is invalid
            FileNotFoundError: If the specified JSON file doesn't exist
            json.JSONDecodeError: If the JSON file is malformed
        """
        if isinstance(data_source, str):
            if not data_source.endswith(".json"):
                raise ValueError("File must be a JSON file")
            try:
                with open(data_source, "r") as f:
                    data = json.load(f)
                if not isinstance(data, list) or not all(
                    isinstance(item, str) for item in data
                ):
                    raise ValueError("JSON file must contain an array of strings")
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON file format")
            except FileNotFoundError:
                raise FileNotFoundError(f"File {data_source} not found")
        elif isinstance(data_source, list):
            if not all(isinstance(item, str) for item in data_source):
                raise ValueError("list must contain only strings")
            data = data_source

        texts = []
        for text in data:
            soup = BeautifulSoup(text, "html.parser")
            extracted_text = soup.getText(separator=" ")

            # Decode HTML entities
            extracted_text = html.unescape(extracted_text)

            # Clean up the text by removing multiple spaces, newlines, tabs, and special characters
            cleaned_text = re.sub(
                r"\s+", " ", extracted_text
            )  # Replace multiple whitespace with single space

            # Remove special symbols but keep accented characters
            # Keep letters (including accented), numbers, basic punctuation and spaces
            cleaned_text = re.sub(
                r"[^a-zA-Z\u00C0-\u017F0-9\s.,!?-]", "", cleaned_text
            )  # Keep letters (including accented), numbers, spaces, and basic punctuation

            cleaned_text = cleaned_text.strip()  # Remove leading/trailing whitespace
            texts.append(cleaned_text)

        return texts

    def annotate(
        self, preprocessed_data: list[str], labels: list[str], **kwargs
    ) -> list[dict[str, Any]]:
        """
        Phase 2: Annotate text using Large Language Model (LLM) with DSPy.

        This method uses the specified LLM model to extract named entities from the
        preprocessed text. It handles long documents through automatic chunking and
        supports parallel processing.

        Args:
            preprocessed_data (list[str]): list of preprocessed text strings.
            labels (list[str]): list of entity types to extract.
            **kwargs: Additional annotation parameters:
                - num_threads (int): Number of parallel threads for processing.
                  Defaults to 4.

        Returns:
            list[dict[str, Any]]: A list of annotated data, where each item contains:
                - tokenized_text: list of tokens
                - ner: list of entity spans (start, end, type)
                - chunk_info: Information about text chunking

        Raises:
            ValueError: If preprocessed_data or labels are not provided
        """
        if preprocessed_data is None:
            raise ValueError("No training data available. Call preprocess() first.")

        entities = labels
        if entities is None:
            raise ValueError("No entities provided for annotation.")

        # Import litellm here to avoid circular imports
        import litellm

        # Set up the API key if provided
        if self.llm_api_key:
            litellm.api_key = self.llm_api_key

        # Configure DSPy with the LLM
        lm = dspy.LM(
            self.llm_model,
            api_key=self.llm_api_key,
            api_base=self.llm_api_base,
            max_tokens=4096,
        )
        dspy.configure(lm=lm)

        # Create the entity extraction predictor with the entities in the docstring
        EntityExtractionSignature.__doc__ = EntityExtractionSignature.__doc__.format(
            entities=", ".join(entities)
        )
        extractor = dspy.Predict(EntityExtractionSignature)

        # Process each text, splitting into chunks if needed
        annotated_data = []
        for i, text in enumerate(preprocessed_data):
            # Split text into chunks if it's too long
            chunks = [text]

            # Process each chunk
            chunk_results = []
            for chunk in chunks:
                # Create example for this chunk
                example = dspy.Example(passage=chunk).with_inputs("passage")

                # Process the chunk
                output = dspy.Evaluate(
                    devset=[example],
                    metric=lambda x, y: True,  # Simple metric that always returns True
                    num_threads=kwargs.get("num_threads", 4),
                    display_progress=True,
                    return_outputs=True,
                )(extractor)

                # Extract the results
                result = dict(output[1][0][1])
                chunk_results.extend(result.get("entities", []))

            # Combine results from all chunks
            annotated_data.append({"text": text, "entities": chunk_results})

            print(f"Iteration {i} completed")
            print(chunk_results)

        # Tokenize the annotated data
        return extract_entities(annotated_data)

    def fine_tune(self, annotated_data: list[dict[str, Any]]) -> GLiNER:
        """
        Phase 3: Fine-tune the GLiNER model on annotated data.

        This method trains the GLiNER model on the annotated data, automatically
        handling device selection and training configuration. It includes validation
        split and early stopping.

        Args:
            annotated_data (list[dict[str, Any]]): list of annotated data.
                Each item should contain:
                - tokenized_text: list of tokens
                - ner: list of entity spans (start, end, type)

        Returns:
            GLiNER: The fine-tuned GLiNER model

        Note:
            The training process automatically:
            - Splits data into train/test sets (90/10)
            - Shuffles the dataset
            - Adjusts batch size based on dataset size
            - Uses available hardware (CUDA, MPS, or CPU)
        """
        device = None

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model = GLiNER.from_pretrained(self.gliner_model)
        model.data_processor.transformer_tokenizer.model_max_length = MAX_WINDOW_SIZE

        data_collator = DataCollator(
            model.config, data_processor=model.data_processor, prepare_labels=True
        )

        model.to(device)

        print("Dataset size:", len(annotated_data))

        random.shuffle(annotated_data)
        print("Dataset is shuffled...")

        train_dataset = annotated_data[: int(len(annotated_data) * 0.9)]
        test_dataset = annotated_data[int(len(annotated_data) * 0.9) :]

        num_steps = 500
        batch_size = 8
        data_size = len(train_dataset)

        # Adjust batch size if it's larger than the dataset size
        if batch_size > data_size:
            batch_size = max(1, data_size)
            print(f"Batch size adjusted to {batch_size} to match dataset size")

        num_batches = max(1, data_size // batch_size)
        num_epochs = max(1, num_steps // num_batches)

        # Default training arguments
        default_args = {
            "output_dir": "models",
            "learning_rate": 5e-6,
            "weight_decay": 0.01,
            "others_lr": 1e-5,
            "others_weight_decay": 0.01,
            "lr_scheduler_type": "linear",  # cosine
            "warmup_ratio": 0.1,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "focal_loss_alpha": 0.75,
            "focal_loss_gamma": 2,
            "num_train_epochs": num_epochs,
            "evaluation_strategy": "steps",
            "save_steps": 100,
            "save_total_limit": 10,
            "dataloader_num_workers": 0,
            "use_cpu": False,
            "report_to": "none",
        }

        # Update default args with any provided kwargs
        default_args.update(self.training_args)

        training_args = TrainingArguments(**default_args)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=model.data_processor.transformer_tokenizer,
            data_collator=data_collator,
        )

        trainer.train()

        return trainer.model

    def inference(
        self, model: GLiNER, text: str, labels: list[str], threshold: float = 0.7
    ) -> list[dict[str, Any]]:
        """
        Extract entities from text using the fine-tuned model.

        This method processes input text through the model, handling long texts
        through automatic chunking and deduplicating results.

        Args:
            model (GLiNER): The fine-tuned GLiNER model to use
            text (str): The text to extract entities from
            labels (list[str]): list of entity types to extract
            threshold (float): Confidence threshold for entity extraction.
                Defaults to 0.7.

        Returns:
            list[dict[str, Any]]: list of extracted entities, where each item contains:
                - entity: The extracted entity text
                - types: list of entity types assigned to the entity

        Note:
            For texts longer than the model's maximum window size (384 tokens),
            the text is automatically split into overlapping chunks for processing.
        """
        if model is None:
            model = GLiNER.from_pretrained(self.gliner_model)

        model.data_processor.transformer_tokenizer.model_max_length = MAX_WINDOW_SIZE

        # Set device
        device = torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "mps" if torch.mps.is_available() else "cpu"
        )
        model = model.to(device)
        model.eval()

        # Tokenize and chunk text if needed
        tokenizer = model.data_processor.transformer_tokenizer

        # Tokenize the text
        tokens = tokenizer(
            text, return_tensors="pt", truncation=False, padding=False
        ).input_ids[0]

        # If text is short enough, process it directly
        if len(tokens) <= MAX_WINDOW_SIZE:
            entities = model.predict_entities(text, labels, threshold=threshold)
            return [
                {
                    "entity": e["text"],
                    "types": (
                        [e["label"]] if isinstance(e["label"], str) else e["label"]
                    ),
                }
                for e in entities
            ]

        # Otherwise, chunk the text
        chunks = [
            tokens[i : i + MAX_WINDOW_SIZE]
            for i in range(0, len(tokens), MAX_WINDOW_SIZE)
        ]
        text_chunks = [
            tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks
        ]

        # Process each chunk and merge results
        from collections import defaultdict

        entity_dict = defaultdict(list)

        for chunk in text_chunks:
            entities = model.predict_entities(chunk, labels)
            for e in entities:
                label = e["label"] if isinstance(e["label"], str) else e["label"]
                entity_dict[e["text"]].extend(
                    label if isinstance(label, list) else [label]
                )

        # Deduplicate entity types
        return [
            {"entity": entity, "types": list(set(types))}
            for entity, types in entity_dict.items()
        ]

    def evaluate(
        self, model: GLiNER, test_data: list[dict[str, Any]], **kwargs
    ) -> dict[str, float]:
        """
        Evaluate the model's performance on test data.

        This method assesses the model's ability to extract entities by comparing
        predictions against the test data.

        Args:
            model (GLiNER): The fine-tuned GLiNER model to evaluate
            test_data (list[dict[str, Any]]): list of test data containing ground truth annotations
            **kwargs: Additional evaluation parameters passed to the model's evaluate method

        Returns:
            dict[str, float]: Evaluation metrics including precision, recall, and F1 score
        """
        if model is None:
            model = GLiNER.from_pretrained(self.gliner_model)

        model.data_processor.transformer_tokenizer.model_max_length = MAX_WINDOW_SIZE
        model.eval()
        return model.evaluate(test_data=test_data, **kwargs)

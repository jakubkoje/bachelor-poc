import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "lantner", "src"))

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from typing import Optional
import json
import requests
from urllib.parse import urlparse
from gliner import GLiNER

# Import lantner
from lantner.src import Lantner

app = FastAPI(title="Lantner NER App")

# Optimized training arguments for the multilang model
training_args = {
    "learning_rate": 3e-5,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "num_train_epochs": 10,
    "dataloader_num_workers": 8,
    "bf16": True,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
}

# Initialize lantner pipeline with multilang model (based on test.py)
ner_pipeline = Lantner(
    gliner_model="jakubkoje/gliner-multi-movies",
    llm_model="openai/gpt-4o-mini",  # Using a more accessible model
    llm_api_key="",  # Add your API key here if needed
    # llm_api_base="http://localhost:11434",  # Uncomment for local LLM
    **training_args,
)

model = GLiNER.from_pretrained("jakubkoje/gliner-multi-movies")


# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lantner NER</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 20px;
        }
        .description {
            background-color: #fff5f0;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #ff8000;
        }
        .description p {
            margin: 0 0 10px 0;
            color: #555;
            line-height: 1.5;
        }
        .description p:last-child {
            margin-bottom: 0;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #555;
            font-weight: 500;
        }
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            transition: border-color 0.3s;
            font-family: inherit;
            min-height: 100px;
            resize: vertical;
            box-sizing: border-box;
        }
        textarea:focus, input[type="url"]:focus {
            outline: none;
            border-color: #ff8000;
        }
        .entities-section {
            margin-bottom: 20px;
            padding: 15px;
            background-color: #fff8f0;
            border-radius: 5px;
        }
        button {
            background-color: #ff8000;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s;
            width: 100%;
        }
        button:hover {
            background-color: #e6700d;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-left: 4px solid #ff8000;
            border-radius: 5px;
            display: none;
        }
        .result h3 {
            margin-top: 0;
            color: #333;
        }
        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 10px;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #ff8000;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Lantner Named Entity Recognition</h1>
        <div class="description">
            <p><strong>Movie Data Extraction from HTML</strong></p>
            <p>This tool specializes in extracting movie-related entities from HTML content. It uses a fine-tuned GLiNER 2.1 multilang model that was trained using automated labeling with GPT-4.1-nano to identify and extract movie information such as titles, years, runtime, ratings, genres, directors, cast, and plot details.</p>
            <p><strong>How to use:</strong> Enter a URL to a webpage containing movie information (e.g., IMDB, CSFD, etc.) and the tool will fetch the HTML and extract movie entities.</p>
        </div>
        <form id="inputForm">
            <div class="form-group">
                <label for="urlInput">Enter URL for movie data extraction:</label>
                <input type="url" id="urlInput" name="urlInput" placeholder="https://example.com/movie-page" required style="width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 5px; font-size: 16px; transition: border-color 0.3s; font-family: inherit; box-sizing: border-box;">
            </div>
            
            <div class="entities-section">
                <label>Entity Types to Extract (one per line):</label>
                <textarea id="entityLabels" name="entityLabels" placeholder="title&#10;year&#10;runtime&#10;rating&#10;genre&#10;director&#10;cast&#10;plot" rows="5"></textarea>
                <small style="color: #666;">Examples: title, year, runtime, rating, genre, director, cast, plot</small>
            </div>
            
            <button type="submit" id="submitBtn">Extract Entities</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Processing...</p>
        </div>
        
        <div class="result" id="result">
            <h3>Response:</h3>
            <pre id="responseData"></pre>
        </div>
    </div>

    <script>
        // Set default entity labels
        document.getElementById('entityLabels').value = 'title\\nyear\\nruntime\\nrating\\ngenre\\ndirector\\ncast\\nplot';
        
        document.getElementById('inputForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const urlInput = document.getElementById('urlInput').value;
            const entityLabels = document.getElementById('entityLabels').value;
            const submitBtn = document.getElementById('submitBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const responseData = document.getElementById('responseData');
            
            // Show loading state
            submitBtn.disabled = true;
            loading.style.display = 'block';
            result.style.display = 'none';
            
            try {
                const formData = new FormData();
                formData.append('url', urlInput);
                formData.append('entity_labels', entityLabels);
                
                const response = await fetch('/extract_entities', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                // Display the JSON response
                responseData.textContent = JSON.stringify(data, null, 2);
                result.style.display = 'block';
                
            } catch (error) {
                responseData.textContent = JSON.stringify({
                    "error": "Failed to process request",
                    "details": error.message
                }, null, 2);
                result.style.display = 'block';
            } finally {
                // Hide loading state
                submitBtn.disabled = false;
                loading.style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the HTML page with input form"""
    return HTMLResponse(content=HTML_TEMPLATE)


def fetch_html_from_url(url: str) -> str:
    """Fetch HTML content from a URL"""
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")
        
        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the HTML content
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        return response.text
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch URL: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing URL: {str(e)}")


@app.post("/extract_entities")
async def extract_entities(url: str = Form(...), entity_labels: str = Form(...)):
    """Extract named entities from HTML content fetched from URL using lantner"""
    try:
        # Parse entity labels from textarea input
        labels = [
            label.strip().lower()
            for label in entity_labels.split("\n")
            if label.strip()
        ]

        if not labels:
            labels = ["person", "organization", "location"]  # Default labels

        try:
            # Fetch HTML content from URL
            print(f"Fetching HTML from URL: {url}")
            html_content = fetch_html_from_url(url)
            print(f"Fetched HTML content length: {len(html_content)}")
            
            # Preprocess the HTML content
            print(f"Preprocessing HTML content: {html_content[:100]}...")
            preprocessed_texts = ner_pipeline.preprocess(data_source=[html_content])
            preprocessed_text = (
                preprocessed_texts[0] if preprocessed_texts else html_content
            )

            # Use the pre-trained model for inference (no retraining needed)
            print(f"Running inference on: {preprocessed_text[:100]}...")
            entities_result = ner_pipeline.inference(
                model=model, text=preprocessed_text, labels=labels, threshold=0.5
            )

            # Convert lantner output format to our expected format
            extracted_entities = []
            if entities_result:
                for entity in entities_result:
                    print(entity)
                    extracted_entities.append(
                        {
                            "entity": entity.get("entity", ""),
                            "types": entity.get("types", []),
                        }
                    )

            response_data = {
                "url": url,
                "requested_labels": labels,
                "extracted_entities": extracted_entities,
                "entity_count": len(extracted_entities),
                "text_stats": {
                    "html_length": len(html_content),
                    "preprocessed_length": len(preprocessed_text),
                    "html_word_count": len(html_content.split()),
                    "preprocessed_word_count": len(preprocessed_text.split()),
                    "sentence_count": len(
                        [s for s in preprocessed_text.split(".") if s.strip()]
                    ),
                },
                "status": "success",
                "message": f"Extracted {len(extracted_entities)} entities from {url} using lantner multilang model with preprocessing",
                "ner_method": "lantner_multilang_with_preprocessing",
            }

        except Exception as e:
            response_data = {
                "error": "Entity extraction failed",
                "url": url,
                "requested_labels": labels,
                "details": str(e),
                "status": "extraction_failed",
            }

        return JSONResponse(content=response_data)

    except Exception as e:
        return JSONResponse(
            content={
                "error": "Request processing failed",
                "details": str(e),
                "status": "failed",
            }
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

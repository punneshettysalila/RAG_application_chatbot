# PDF RAG Chatbot

A Streamlit-based **Retrieval-Augmented Generation (RAG)** chatbot that lets you:
- Upload one or more PDF files
- Build a FAISS vector index from document chunks
- Ask questions grounded in your uploaded documents
- Get answers from **TinyLlama** via Hugging Face Transformers

## Features

- Multi-PDF upload and indexing
- Semantic retrieval with `all-MiniLM-L6-v2` embeddings
- Chat experience with short conversation memory
- Dark theme selector in UI
- Quick starter prompts
- Guardrails for missing index/model
- LLM timeout protection (`LLM_TIMEOUT_SECONDS`)

## Project Structure

```text
app.py            # Streamlit UI and chat flow
rag_pipeline.py   # PDF loading, chunking, embedding, retrieval
llm_handler.py    # Hugging Face model loading, prompting, response handling
```

## Requirements

- Python 3.9+
- Hugging Face account + access token
- (Recommended) GPU for faster inference

### Python packages used

- `streamlit`
- `langchain-community`
- `langchain-text-splitters`
- `langchain-huggingface`
- `transformers`
- `torch`
- `python-dotenv`
- `faiss-cpu` (or FAISS variant for your environment)
- `pypdf`

## Setup

1. **Clone the repo** and move into the project folder.
2. **Create/activate a virtual environment** (recommended).
3. **Install dependencies**:

```bash
pip install streamlit langchain-community langchain-text-splitters langchain-huggingface transformers torch python-dotenv faiss-cpu pypdf
```

4. **Create a `.env` file** in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
# Optional
LLM_TIMEOUT_SECONDS=90
```

The app accepts any of these token variable names:
- `HUGGINGFACEHUB_API_TOKEN` (preferred)
- `HUGGINGFACE_API_TOKEN`
- `HF_TOKEN`

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit in your browser.

## How to Use

1. Upload PDF files from the sidebar.
2. Click **Build Vector Index**.
3. Click **Load LLaMA Model**.
4. Ask questions in the chat input.
5. Type `exit` to close the current chat session.
6. Use **Reset Conversation** to start over.


## OUTPUT:

<img width="1920" height="1020" alt="Image" src="https://github.com/user-attachments/assets/89f93c80-c68b-408e-889e-d3e2b32e4235" />



<img width="1920" height="1020" alt="Image" src="https://github.com/user-attachments/assets/e7bc7427-ad42-402d-bbfa-30d1ecd08471" />

## Notes

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- On CUDA-capable systems, the app uses GPU and may auto-enable 4-bit quantization when supported.
- On CPU-only machines, responses may be slower.

## Troubleshooting

- **“Missing Hugging Face token”**  
  Ensure `.env` exists in the project root and contains a valid token.

- **PDF indexing fails**  
  Verify PDFs contain extractable text (not scanned images only).

- **Model load is slow**  
  First run downloads model files; later runs are faster due to caching.

- **LLM timeout errors**  
  Increase `LLM_TIMEOUT_SECONDS` in `.env` or ask shorter questions.


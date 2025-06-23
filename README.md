# 🧠 AI-Powered Health Recommendation System

This project is an **AI-driven preventive health assistant** that provides **personalized medical recommendations** based on user responses and relevant scientific literature (PDFs).  
It uses **OpenAI GPT-4**, **Pinecone vector database**, and **OCR techniques** to process medical PDFs and intelligently respond to user health forms.

Ideal for:
- Digital health platforms
- Personalized wellness assistants
- Research-backed health advisory tools

---

## 🧠 Tech Stack

| Technology         | Purpose                                                   |
|--------------------|-----------------------------------------------------------|
| **Python 3.10+**   | Core programming language                                  |
| **OpenAI GPT-4**   | Generate personalized health recommendations               |
| **Pinecone**       | Vector database to store and query document embeddings     |
| **LangChain**      | Chunk and format long text documents                       |
| **PDFPlumber**     | Extract text from native PDFs                              |
| **PDF2Image + EasyOCR** | Extract text from scanned PDFs using OCR             |
| **dotenv**         | Securely load API keys from `.env`                         |
| **Pickle + JSON**  | Local cache and metadata storage                           |
| **uuid, glob, os** | File handling and cleanup utilities                        |

---

## 🚀 Features

- 📄 Reads & processes medical PDFs (even scanned ones)
- 🧠 Extracts text with OCR and PDF parsing
- 🧬 Creates embeddings using OpenAI
- 🔍 Stores and queries content using Pinecone vector DB
- 🤖 Uses GPT-4 to generate tailored, reference-backed recommendations
- ⚡ Caches OCR results and embeddings for performance
- 📚 Matches user health form data with PubMed literature

---

## 📁 Project Structure

```
├── main.py # Main app logic
├── uploads/ # Input medical PDFs
├── cache/
│ ├── ocr/ # OCR text cache
│ └── embeddings/ # Embedding cache
├── temp_uploads/ # Temporary file store
├── processed_data.pkl # Local vector storage
├── pdf_pmid_mapping.json # Maps PDFs to PMIDs
├── requirements.txt # Dependency list
├── .env # Your API keys (not pushed)
├── .gitignore # Ignore secrets & venv

```


## 🧩 Dependencies

| Library               | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `openai`              | GPT-4 API for recommendation generation                                 |
| `dotenv`              | Loads `.env` file containing API keys                                   |
| `pdfplumber`          | Extracts text from PDFs                                                  |
| `pdf2image`           | Converts PDF pages to images (for OCR)                                  |
| `easyocr`             | Performs OCR on scanned PDF pages                                       |
| `pinecone-client`     | Interfaces with Pinecone vector DB                                      |
| `langchain`           | Splits long documents into chunks for embedding                         |
| `uuid`, `pickle`, `os`, `json`, `glob`, `shutil`, `time` | File caching, metadata, and utilities |

Install all dependencies:

```bash
pip install -r requirements.txt
```
---

🔐 Environment Setup
Create a .env file with your API keys:

```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key

```

---


## 🧹 Caching Strategy
OCR results stored in cache/ocr/

Embeddings stored in cache/embeddings/

Old cache files (older than 30 days) are auto-deleted on next run



---

## 👨‍💻 Author
Ashraful Hossain
AI Developer | ML Engineer

---

## 📄 License
MIT License — free for personal and commercial use



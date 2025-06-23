# # import os
# # import openai
# # import pdfplumber
# # from pdf2image import convert_from_path
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.schema import Document
# # from langchain_openai import OpenAIEmbeddings
# # from dotenv import load_dotenv
# # import easyocr
# # import glob
# # from pinecone import Pinecone, ServerlessSpec
# # import pickle
# # import json
# # import time

# # # Load environment variables from .env file
# # load_dotenv()
# # openai_api_key = os.getenv("OPENAI_API_KEY")
# # pinecone_api_key = os.getenv("PINECONE_API_KEY")

# # # Validate API keys
# # if not openai_api_key or not pinecone_api_key:
# #     print("❌ Error: Missing API keys. Please set OPENAI_API_KEY and PINECONE_API_KEY in .env file.")
# #     exit()

# # # Initialize Pinecone client
# # pc = Pinecone(api_key=pinecone_api_key)
# # PINECONE_INDEX_NAME = "medical-chatbot-index"

# # # Define folder and file paths
# # PDF_FOLDER = "pdf_files/"
# # PROCESSED_DATA_FILE = "processed_data.pkl"
# # PDF_PMID_MAPPING_FILE = "pdf_pmid_mapping.json"
# # OCR_CACHE_DIR = "cache/ocr/"
# # EMBEDDING_CACHE_DIR = "cache/embeddings/"

# # # Create cache directories if they don't exist
# # os.makedirs(OCR_CACHE_DIR, exist_ok=True)
# # os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

# # # Initialize OpenAI client
# # client = openai.OpenAI(api_key=openai_api_key)

# # # Function: Clean old cache files older than 30 days
# # def clean_cache_files(directory):
# #     """Remove .txt and .pkl files older than 30 days from the specified directory."""
# #     deleted_count = 0
# #     for file in glob.glob(os.path.join(directory, "*.txt")) + glob.glob(os.path.join(directory, "*.pkl")):
# #         if os.path.getmtime(file) < time.time() - (30 * 24 * 3600):
# #             os.remove(file)
# #             print(f"🗑️ Removed old cache file: {file}")
# #             deleted_count += 1
# #     print(f"✅ Total {deleted_count} cache files deleted from {directory}")

# # # Function: Load PDF-PMID mapping from JSON file
# # def load_pdf_pmid_mapping():
# #     """Load the mapping of PDF file paths to PMIDs from a JSON file, creating an empty file if it doesn't exist."""
# #     if os.path.exists(PDF_PMID_MAPPING_FILE):
# #         with open(PDF_PMID_MAPPING_FILE, 'r', encoding='utf-8') as f:
# #             return json.load(f)
# #     else:
# #         empty_mapping = {}
# #         with open(PDF_PMID_MAPPING_FILE, 'w', encoding='utf-8') as f:
# #             json.dump(empty_mapping, f, indent=2)
# #         return empty_mapping

# # # Function: Save PDF-PMID mapping to JSON file
# # def save_pdf_pmid_mapping(mapping):
# #     """Save the PDF-PMID mapping to a JSON file."""
# #     with open(PDF_PMID_MAPPING_FILE, 'w', encoding='utf-8') as f:
# #         json.dump(mapping, f, indent=2)

# # # Load PDF files from the PDF folder
# # pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
# # if not pdf_files:
# #     print("❌ No PDF files found. Please add PDFs to 'pdf_files/' directory.")
# #     exit()

# # # Function: Convert PDF pages to images for OCR
# # def convert_pdf_to_images(pdf_path):
# #     """Convert each page of a PDF to a PNG image and save in the OCR cache directory."""
# #     image_paths = []
# #     try:
# #         images = convert_from_path(pdf_path, poppler_path=r"C:\Users\STA\Downloads\Release-24.08.0-0 (3)\poppler-24.08.0\Library\bin")
# #         for i, image in enumerate(images):
# #             image_path = os.path.join(OCR_CACHE_DIR, f"page_{i+1}.png")
# #             image.save(image_path, 'PNG')
# #             image_paths.append(image_path)
# #     except Exception as e:
# #         print(f"⚠️ Error converting {pdf_path} to images: {e}")
# #     return image_paths

# # # Function: Extract text from an image using OCR
# # def extract_text_from_image(image_path):
# #     """Extract text from an image using EasyOCR, loading from cache if available or saving to cache if not."""
# #     cache_file = os.path.join(OCR_CACHE_DIR, f"{os.path.basename(image_path)}.txt")
# #     if os.path.exists(cache_file):
# #         with open(cache_file, 'r', encoding='utf-8') as f:
# #             print(f"✅ Using cached OCR text for {image_path}")
# #             return f.read()
# #     try:
# #         reader = easyocr.Reader(['en'], gpu=True)
# #         result = reader.readtext(image_path)
# #         extracted_text = " ".join([detection[1] for detection in result])
# #         with open(cache_file, 'w', encoding='utf-8') as f:
# #             f.write(extracted_text)
# #         print(f"✅ OCR completed and cached for {image_path}")
# #         return extracted_text
# #     except Exception as e:
# #         print(f"⚠️ Error extracting text from image {image_path}: {e}")
# #         return ""

# # # Function: Extract text directly from PDF
# # def extract_text_from_pdf(pdf_path):
# #     """Extract text from all pages of a PDF using pdfplumber."""
# #     text = ""
# #     try:
# #         with pdfplumber.open(pdf_path) as pdf:
# #             for page in pdf.pages:
# #                 page_text = page.extract_text()
# #                 if page_text:
# #                     text += page_text + "\n"
# #     except Exception as e:
# #         print(f"⚠️ Error extracting text from {pdf_path}: {e}")
# #     return text

# # # Function: Process PDFs and save data
# # def process_and_save_pdfs(pdf_files, pdf_pmid_mapping):
# #     """Process PDFs to extract text (direct and OCR), combine it, and save with metadata to a pickle file."""
# #     processed_data = {}
# #     print("📑 Processing PDFs...")
# #     for pdf_file in pdf_files:
# #         pdf_name = os.path.normpath(pdf_file)
# #         pmid = pdf_pmid_mapping.get(pdf_name, "PMID-UNKNOWN")
# #         text = extract_text_from_pdf(pdf_file)
# #         image_paths = convert_pdf_to_images(pdf_file)
# #         ocr_text = ""
# #         for image_path in image_paths:
# #             ocr_result = extract_text_from_image(image_path)
# #             ocr_text += ocr_result + " "
# #         combined_text = text + " " + ocr_text
# #         metadata = {"pmid": pmid}
# #         processed_data[pdf_name] = {"text": combined_text, "metadata": metadata}
# #     with open(PROCESSED_DATA_FILE, "wb") as f:
# #         pickle.dump(processed_data, f)
# #     print("✅ PDF processing complete. Data saved.")
# #     return processed_data

# # # Function: Upsert PDF data to Pinecone
# # def upsert_pdf_to_pinecone(pdf_path, pmid, index):
# #     """Extract text from a PDF, generate embeddings, cache them, and upsert to Pinecone."""
# #     pdf_name = os.path.normpath(pdf_path)
# #     cache_file = os.path.join(EMBEDDING_CACHE_DIR, f"embeddings_{pmid}.pkl")
# #     if os.path.exists(cache_file):
# #         with open(cache_file, "rb") as f:
# #             vectors = pickle.load(f)
# #         print(f"✅ Using cached embeddings for {pdf_name}")
# #     else:
# #         text = extract_text_from_pdf(pdf_path)
# #         image_paths = convert_pdf_to_images(pdf_path)
# #         ocr_text = ""
# #         for image_path in image_paths:
# #             ocr_result = extract_text_from_image(image_path)
# #             ocr_text += ocr_result + " "
# #         combined_text = text + " " + ocr_text
# #         if combined_text.strip():
# #             text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
# #             texts = text_splitter.split_text(combined_text)
# #             documents = [Document(page_content=text) for text in texts]
# #             embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# #             vectors = []
# #             for i, doc in enumerate(documents):
# #                 embedding = embeddings.embed_query(doc.page_content)
# #                 vectors.append((
# #                     f"vec_{i}_{pmid}",
# #                     embedding,
# #                     {"text": doc.page_content, "pmid": pmid}
# #                 ))
# #             with open(cache_file, "wb") as f:
# #                 pickle.dump(vectors, f)
# #             print(f"✅ Embeddings created and cached for {pdf_name}")
    
# #     batch_size = 100
# #     for i in range(0, len(vectors), batch_size):
# #         batch = vectors[i:i + batch_size]
# #         try:
# #             index.upsert(vectors=batch)
# #         except Exception as e:
# #             print(f"⚠️ Error upserting batch to Pinecone: {e}")
# #     print(f"✅ PDF {pdf_name} successfully added to Pinecone index.")

# # # Function: Load and upsert data to Pinecone
# # def load_and_upsert_data(index, pdf_pmid_mapping):
# #     """Load processed data from pickle file and upsert to Pinecone, skipping existing entries."""
# #     processed_data = {}
# #     if os.path.exists(PROCESSED_DATA_FILE):
# #         with open(PROCESSED_DATA_FILE, "rb") as f:
# #             processed_data = pickle.load(f)
        
# #         print("📤 Checking Pinecone for existing data...")
# #         for pdf_name, data in processed_data.items():
# #             pmid = data["metadata"]["pmid"]
# #             sample_vector = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query("test")
# #             query_results = index.query(
# #                 vector=sample_vector,
# #                 top_k=1,
# #                 filter={"pmid": pmid}
# #             )
# #             if query_results["matches"]:
# #                 print(f"✅ Data for PMID {pmid} already exists in Pinecone. Skipping upload.")
# #                 continue
            
# #             combined_text = data["text"]
# #             metadata = data["metadata"]
# #             if combined_text.strip():
# #                 text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
# #                 texts = text_splitter.split_text(combined_text)
# #                 documents = [Document(page_content=text) for text in texts]
# #                 embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# #                 vectors = []
# #                 for i, doc in enumerate(documents):
# #                     embedding = embeddings.embed_query(doc.page_content)
# #                     vectors.append((
# #                         f"vec_{i}_{metadata['pmid']}",
# #                         embedding,
# #                         {
# #                             "text": doc.page_content,
# #                             "pmid": metadata["pmid"]
# #                         }
# #                     ))
# #                 batch_size = 50
# #                 for i in range(0, len(vectors), batch_size):
# #                     batch = vectors[i:i + batch_size]
# #                     try:
# #                         index.upsert(vectors=batch)
# #                     except Exception as e:
# #                         print(f"⚠️ Error upserting batch to Pinecone: {e}")
# #         print("✅ Data upload to Pinecone completed.")
# #     else:
# #         processed_data = process_and_save_pdfs(pdf_files, pdf_pmid_mapping)
# #         load_and_upsert_data(index, pdf_pmid_mapping)

# # # Function: Add a new PDF to the system
# # def add_new_pdf(pdf_path, pmid, index, pdf_pmid_mapping):
# #     """Add a new PDF by processing it, updating the mapping, upserting to Pinecone, and saving to processed_data.pkl."""
# #     pdf_name = os.path.normpath(pdf_path)
# #     pdf_pmid_mapping[pdf_name] = pmid
# #     save_pdf_pmid_mapping(pdf_pmid_mapping)
# #     upsert_pdf_to_pinecone(pdf_path, pmid, index)
    
# #     # Update processed_data.pkl
# #     processed_data = {}
# #     if os.path.exists(PROCESSED_DATA_FILE):
# #         with open(PROCESSED_DATA_FILE, "rb") as f:
# #             processed_data = pickle.load(f)
# #     text = extract_text_from_pdf(pdf_path)
# #     image_paths = convert_pdf_to_images(pdf_path)
# #     ocr_text = ""
# #     for image_path in image_paths:
# #         ocr_result = extract_text_from_image(image_path)
# #         ocr_text += ocr_result + " "
# #     combined_text = text + " " + ocr_text
# #     metadata = {"pmid": pmid}
# #     processed_data[pdf_name] = {"text": combined_text, "metadata": metadata}
# #     with open(PROCESSED_DATA_FILE, "wb") as f:
# #         pickle.dump(processed_data, f)
# #     print(f"✅ New PDF {pdf_name} and PMID {pmid} added and saved to processed_data.pkl.")

# # # Function: Add multiple PDFs to the system
# # def add_multiple_pdfs(pdf_pmid_list, index, pdf_pmid_mapping):
# #     """Process and add multiple PDFs with their PMIDs, skipping duplicates."""
# #     processed_pdfs = set()
# #     for pdf_path, pmid in pdf_pmid_list:
# #         pdf_path = os.path.normpath(pdf_path)
# #         if pdf_path in processed_pdfs:
# #             print(f"⚠️ Skipping duplicate PDF: {pdf_path}")
# #             continue
# #         if os.path.exists(pdf_path):
# #             print(f"📄 Processing new PDF: {pdf_path} (PMID: {pmid})")
# #             add_new_pdf(pdf_path, pmid, index, pdf_pmid_mapping)
# #             processed_pdfs.add(pdf_path)
# #         else:
# #             print(f"❌ PDF file {pdf_path} not found. Please add it to 'pdf_files/' directory.")

# # # Function: Generate health recommendations
# # def generate_fast_recommendations(json_data, index, user_name="User"):
# #     """Generate preventive health recommendations based on user data and Pinecone data."""
# #     try:
# #         form_data = json_data.get("response_form_data", [])
# #         context = ""
# #         pmids = set()

# #         for item in form_data:
# #             user_input = f"{item.get('question', '')}: {item.get('response_text', '')}"
# #             query_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query(user_input)
# #             query_results = index.query(
# #                 vector=query_embedding,
# #                 top_k=5,
# #                 include_metadata=True
# #             )
# #             for match in query_results["matches"]:
# #                 metadata = match.get("metadata", {})
# #                 text = metadata.get("text", "")
# #                 pmid = metadata.get("pmid", "PMID-UNKNOWN")
# #                 context += text + " "
# #                 if pmid != "PMID-UNKNOWN":
# #                     pmids.add(pmid)

# #         context = context.strip()
# #         pmids = list(pmids)

# #         if not context:
# #             return {
# #                 "message": f"Sorry, {user_name}, I don't have specific information on this topic. Please consult a healthcare professional for accurate advice.",
# #                 "sources": "No source information found."
# #             }

# #         messages = [
# #             {
# #                 "role": "system",
# #                 "content": f"""
# #     You are a professional medical AI assistant. Based on the user's health data and the following medical literature references, provide clear, concise preventive health recommendations. The user data includes responses to the following questions: Name, Age, Gender, Zip Code, Smoking Status, Exercise Frequency, Sleep Duration, Daily Fruit/Vegetable Intake, Weekly Physical Activity, Last Doctor Visit, Blood Pressure, Do you wear seatbelts?, Do you feel safe at home?, Last Dental Checkup, BMI, Cholesterol Level.

# #     - Format your response as a numbered list.
# #     - After each recommendation, include a source link in this exact format:  
# #     Source: https://pubmed.ncbi.nlm.nih.gov/{{pmid}}/
# #     - Use these PubMed IDs as references for your response: {', '.join(pmids)}.
# #     - If no PubMed ID is relevant for a recommendation, write:  
# #     Source: No source information found.
# #     - Use simple, actionable language tailored to the user's specific health data (e.g., address high blood pressure if reported, recommend quitting smoking if the user smokes).
# #     - Focus on preventive health measures based on the provided questions and user responses.
# #     """
# #             },
# #             {
# #                 "role": "user",
# #                 "content": f"User data: {json.dumps(form_data, ensure_ascii=False)}\nContext: {context}"
# #             }
# #         ]

# #         response = client.chat.completions.create(
# #             model="gpt-4-turbo",
# #             messages=messages,
# #             max_tokens=500,
# #             temperature=0.2
# #         )

# #         ai_response = response.choices[0].message.content.strip()
# #         sources_text = f"Sources: {', '.join(pmids)}" if pmids else "Sources: No source information found."

# #         return {"message": ai_response, "sources": sources_text}

# #     except Exception as e:
# #         return {"error": f"Error: {e}"}

# # # Function: Process backend JSON data
# # def process_backend_data(json_input_file, index):
# #     """Read a JSON file with user data and generate health recommendations."""
# #     try:
# #         with open(json_input_file, 'r', encoding='utf-8') as f:
# #             json_data = json.load(f)
# #         user_name = json_data.get("user_name", "User")
# #         recommendations = generate_fast_recommendations(json_data, index, user_name)
# #         recommendations['user_name'] = user_name
# #         return recommendations
# #     except Exception as e:
# #         return {"error": f"Error reading JSON file: {e}"}

# # # Main execution block
# # if __name__ == "__main__":
# #     # Clean old cache files at the start
# #     print("🧹 Cleaning old cache files...")
# #     clean_cache_files(OCR_CACHE_DIR)
# #     clean_cache_files(EMBEDDING_CACHE_DIR)

# #     # Load PDF-PMID mapping
# #     pdf_pmid_mapping = load_pdf_pmid_mapping()

# #     # Create Pinecone index if it doesn't exist
# #     if PINECONE_INDEX_NAME not in pc.list_indexes().names():
# #         print("🛠️ Creating Pinecone index...")
# #         pc.create_index(
# #             name=PINECONE_INDEX_NAME,
# #             dimension=1536,
# #             metric="cosine",
# #             spec=ServerlessSpec(cloud="aws", region="us-east-1")
# #         )
# #     index = pc.Index(PINECONE_INDEX_NAME)

# #     # Initial data load and upload to Pinecone
# #     load_and_upsert_data(index, pdf_pmid_mapping)

# #     # Add new PDFs and PMIDs
# #     new_pdfs = [
# #         (os.path.join(PDF_FOLDER, "ABPM vs office in HTN_NEJM.pdf"), "PMID-12802026"),
# #         (os.path.join(PDF_FOLDER, "OptiBP app.pdf"), "PMID-35727426")
# #     ]
# #     add_multiple_pdfs(new_pdfs, index, pdf_pmid_mapping)

# #     # Process JSON file for health recommendations
# #     json_input_file = "input.json"
# #     results = process_backend_data(json_input_file, index)

# #     # Print results
# #     if "error" in results:
# #         print(f"❌ {results['error']}")
# #     else:
# #         print(f"Hello {results.get('user_name', 'User')},")
# #         print("Thank you for using our service.")
# #         print("Based on your responses, here are preventive health recommendations just for you.")
# #         print("Please take a moment to email these to yourself, a loved one, or your medical provider.")
# #         print(results['message'])
# #         print(f"Sources: {results['sources']}")

















# import os
# import openai
# import pdfplumber
# from pdf2image import convert_from_path
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# import easyocr
# import glob
# from pinecone import Pinecone, ServerlessSpec
# import pickle
# import json
# import time

# # Load environment variables from .env file
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")

# # Validate API keys
# if not openai_api_key or not pinecone_api_key:
#     print("❌ Error: Missing API keys. Please set OPENAI_API_KEY and PINECONE_API_KEY in .env file.")
#     exit()

# # Initialize Pinecone client
# pc = Pinecone(api_key=pinecone_api_key)
# PINECONE_INDEX_NAME = "medical-chatbot-index"

# # Define folder and file paths
# PDF_FOLDER = "pdf_files/"
# PROCESSED_DATA_FILE = "processed_data.pkl"
# PDF_PMID_MAPPING_FILE = "pdf_pmid_mapping.json"
# OCR_CACHE_DIR = "cache/ocr/"
# EMBEDDING_CACHE_DIR = "cache/embeddings/"

# # Create cache directories if they don't exist
# os.makedirs(OCR_CACHE_DIR, exist_ok=True)
# os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

# # Initialize OpenAI client
# client = openai.OpenAI(api_key=openai_api_key)

# # Function: Clean old cache files older than 30 days
# def clean_cache_files(directory):
#     """Remove .txt and .pkl files older than 30 days from the specified directory."""
#     deleted_count = 0
#     for file in glob.glob(os.path.join(directory, "*.txt")) + glob.glob(os.path.join(directory, "*.pkl")):
#         if os.path.getmtime(file) < time.time() - (30 * 24 * 3600):
#             os.remove(file)
#             print(f"🗑️ Removed old cache file: {file}")
#             deleted_count += 1
#     print(f"✅ Total {deleted_count} cache files deleted from {directory}")

# # Function: Load PDF-PMID mapping from JSON file
# def load_pdf_pmid_mapping():
#     """Load the mapping of PDF file paths to PMIDs from a JSON file, creating an empty file if it doesn't exist."""
#     if os.path.exists(PDF_PMID_MAPPING_FILE):
#         with open(PDF_PMID_MAPPING_FILE, 'r', encoding='utf-8') as f:
#             return json.load(f)
#     else:
#         empty_mapping = {}
#         with open(PDF_PMID_MAPPING_FILE, 'w', encoding='utf-8') as f:
#             json.dump(empty_mapping, f, indent=2)
#         return empty_mapping

# # Function: Save PDF-PMID mapping to JSON file
# def save_pdf_pmid_mapping(mapping):
#     """Save the PDF-PMID mapping to a JSON file."""
#     with open(PDF_PMID_MAPPING_FILE, 'w', encoding='utf-8') as f:
#         json.dump(mapping, f, indent=2)

# # Load PDF files from the PDF folder
# pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
# if not pdf_files:
#     print("❌ No PDF files found. Please add PDFs to 'pdf_files/' directory.")
#     exit()

# # Function: Convert PDF pages to images for OCR
# def convert_pdf_to_images(pdf_path):
#     """Convert each page of a PDF to a PNG image and save in the OCR cache directory."""
#     image_paths = []
#     try:
#         images = convert_from_path(pdf_path, poppler_path=r"C:\Users\STA\Downloads\Release-24.08.0-0 (3)\poppler-24.08.0\Library\bin")
#         for i, image in enumerate(images):
#             image_path = os.path.join(OCR_CACHE_DIR, f"page_{i+1}.png")
#             image.save(image_path, 'PNG')
#             image_paths.append(image_path)
#     except Exception as e:
#         print(f"⚠️ Error converting {pdf_path} to images: {e}")
#     return image_paths

# # Function: Extract text from an image using OCR
# def extract_text_from_image(image_path):
#     """Extract text from an image using EasyOCR, loading from cache if available or saving to cache if not."""
#     cache_file = os.path.join(OCR_CACHE_DIR, f"{os.path.basename(image_path)}.txt")
#     if os.path.exists(cache_file):
#         with open(cache_file, 'r', encoding='utf-8') as f:
#             print(f"✅ Using cached OCR text for {image_path}")
#             return f.read()
#     try:
#         reader = easyocr.Reader(['en'], gpu=True)
#         result = reader.readtext(image_path)
#         extracted_text = " ".join([detection[1] for detection in result])
#         with open(cache_file, 'w', encoding='utf-8') as f:
#             f.write(extracted_text)
#         print(f"✅ OCR completed and cached for {image_path}")
#         return extracted_text
#     except Exception as e:
#         print(f"⚠️ Error extracting text from image {image_path}: {e}")
#         return ""

# # Function: Extract text directly from PDF
# def extract_text_from_pdf(pdf_path):
#     """Extract text from all pages of a PDF using pdfplumber."""
#     text = ""
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text + "\n"
#     except Exception as e:
#         print(f"⚠️ Error extracting text from {pdf_path}: {e}")
#     return text

# # Function: Process PDFs and save data
# def process_and_save_pdfs(pdf_files, pdf_pmid_mapping):
#     """Process PDFs to extract text (direct and OCR), combine it, and save with metadata to a pickle file."""
#     processed_data = {}
#     print("📑 Processing PDFs...")
#     for pdf_file in pdf_files:
#         pdf_name = os.path.normpath(pdf_file)
#         pmid = pdf_pmid_mapping.get(pdf_name, "PMID-UNKNOWN")
#         text = extract_text_from_pdf(pdf_file)
#         image_paths = convert_pdf_to_images(pdf_file)
#         ocr_text = ""
#         for image_path in image_paths:
#             ocr_result = extract_text_from_image(image_path)
#             ocr_text += ocr_result + " "
#         combined_text = text + " " + ocr_text
#         metadata = {"pmid": pmid}
#         processed_data[pdf_name] = {"text": combined_text, "metadata": metadata}
#     with open(PROCESSED_DATA_FILE, "wb") as f:
#         pickle.dump(processed_data, f)
#     print("✅ PDF processing complete. Data saved.")
#     return processed_data

# # Function: Upsert PDF data to Pinecone
# def upsert_pdf_to_pinecone(pdf_path, pmid, index):
#     """Extract text from a PDF, generate embeddings, cache them, and upsert to Pinecone."""
#     pdf_name = os.path.normpath(pdf_path)
#     cache_file = os.path.join(EMBEDDING_CACHE_DIR, f"embeddings_{pmid}.pkl")
#     if os.path.exists(cache_file):
#         with open(cache_file, "rb") as f:
#             vectors = pickle.load(f)
#         print(f"✅ Using cached embeddings for {pdf_name}")
#     else:
#         text = extract_text_from_pdf(pdf_path)
#         image_paths = convert_pdf_to_images(pdf_path)
#         ocr_text = ""
#         for image_path in image_paths:
#             ocr_result = extract_text_from_image(image_path)
#             ocr_text += ocr_result + " "
#         combined_text = text + " " + ocr_text
#         if combined_text.strip():
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
#             texts = text_splitter.split_text(combined_text)
#             documents = [Document(page_content=text) for text in texts]
#             embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#             vectors = []
#             for i, doc in enumerate(documents):
#                 embedding = embeddings.embed_query(doc.page_content)
#                 vectors.append((
#                     f"vec_{i}_{pmid}",
#                     embedding,
#                     {"text": doc.page_content, "pmid": pmid}
#                 ))
#             with open(cache_file, "wb") as f:
#                 pickle.dump(vectors, f)
#             print(f"✅ Embeddings created and cached for {pdf_name}")
    
#     batch_size = 100
#     for i in range(0, len(vectors), batch_size):
#         batch = vectors[i:i + batch_size]
#         try:
#             index.upsert(vectors=batch)
#         except Exception as e:
#             print(f"⚠️ Error upserting batch to Pinecone: {e}")
#     print(f"✅ PDF {pdf_name} successfully added to Pinecone index.")

# # Function: Load and upsert data to Pinecone
# def load_and_upsert_data(index, pdf_pmid_mapping):
#     """Load processed data from pickle file and upsert to Pinecone, skipping existing entries."""
#     processed_data = {}
#     if os.path.exists(PROCESSED_DATA_FILE):
#         with open(PROCESSED_DATA_FILE, "rb") as f:
#             processed_data = pickle.load(f)
        
#         print("📤 Checking Pinecone for existing data...")
#         for pdf_name, data in processed_data.items():
#             pmid = data["metadata"]["pmid"]
#             sample_vector = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query("test")
#             query_results = index.query(
#                 vector=sample_vector,
#                 top_k=1,
#                 filter={"pmid": pmid}
#             )
#             if query_results["matches"]:
#                 print(f"✅ Data for PMID {pmid} already exists in Pinecone. Skipping upload.")
#                 continue
            
#             combined_text = data["text"]
#             metadata = data["metadata"]
#             if combined_text.strip():
#                 text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
#                 texts = text_splitter.split_text(combined_text)
#                 documents = [Document(page_content=text) for text in texts]
#                 embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#                 vectors = []
#                 for i, doc in enumerate(documents):
#                     embedding = embeddings.embed_query(doc.page_content)
#                     vectors.append((
#                         f"vec_{i}_{metadata['pmid']}",
#                         embedding,
#                         {
#                             "text": doc.page_content,
#                             "pmid": metadata["pmid"]
#                         }
#                     ))
#                 batch_size = 50
#                 for i in range(0, len(vectors), batch_size):
#                     batch = vectors[i:i + batch_size]
#                     try:
#                         index.upsert(vectors=batch)
#                     except Exception as e:
#                         print(f"⚠️ Error upserting batch to Pinecone: {e}")
#         print("✅ Data upload to Pinecone completed.")
#     else:
#         processed_data = process_and_save_pdfs(pdf_files, pdf_pmid_mapping)
#         load_and_upsert_data(index, pdf_pmid_mapping)

# # Function: Add a new PDF to the system
# def add_new_pdf(pdf_path, pmid, index, pdf_pmid_mapping):
#     """Add a new PDF by processing it, updating the mapping, upserting to Pinecone, and saving to processed_data.pkl."""
#     pdf_name = os.path.normpath(pdf_path)
#     pdf_pmid_mapping[pdf_name] = pmid
#     save_pdf_pmid_mapping(pdf_pmid_mapping)
#     upsert_pdf_to_pinecone(pdf_path, pmid, index)
    
#     # Update processed_data.pkl
#     processed_data = {}
#     if os.path.exists(PROCESSED_DATA_FILE):
#         with open(PROCESSED_DATA_FILE, "rb") as f:
#             processed_data = pickle.load(f)
#     text = extract_text_from_pdf(pdf_path)
#     image_paths = convert_pdf_to_images(pdf_path)
#     ocr_text = ""
#     for image_path in image_paths:
#         ocr_result = extract_text_from_image(image_path)
#         ocr_text += ocr_result + " "
#     combined_text = text + " " + ocr_text
#     metadata = {"pmid": pmid}
#     processed_data[pdf_name] = {"text": combined_text, "metadata": metadata}
#     with open(PROCESSED_DATA_FILE, "wb") as f:
#         pickle.dump(processed_data, f)
#     print(f"✅ New PDF {pdf_name} and PMID {pmid} added and saved to processed_data.pkl.")

# # Function: Add multiple PDFs to the system
# def add_multiple_pdfs(pdf_pmid_list, index, pdf_pmid_mapping):
#     """Process and add multiple PDFs with their PMIDs, skipping duplicates."""
#     processed_pdfs = set()
#     for pdf_path, pmid in pdf_pmid_list:
#         pdf_path = os.path.normpath(pdf_path)
#         if pdf_path in processed_pdfs:
#             print(f"⚠️ Skipping duplicate PDF: {pdf_path}")
#             continue
#         if os.path.exists(pdf_path):
#             print(f"📄 Processing new PDF: {pdf_path} (PMID: {pmid})")
#             add_new_pdf(pdf_path, pmid, index, pdf_pmid_mapping)
#             processed_pdfs.add(pdf_path)
#         else:
#             print(f"❌ PDF file {pdf_path} not found. Please add it to 'pdf_files/' directory.")

# def generate_fast_recommendations(json_data, index, user_name="User"):
#     try:
#         form_data = json_data.get("response_form_data", [])
#         context = ""
#         pmids = set()

#         for item in form_data:
#             user_input = f"{item.get('question', '')}: {item.get('response_text', '')}"
#             query_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query(user_input)
#             query_results = index.query(
#                 vector=query_embedding,
#                 top_k=5,
#                 include_metadata=True
#             )
#             for match in query_results["matches"]:
#                 metadata = match.get("metadata", {})
#                 text = metadata.get("text", "")
#                 pmid = metadata.get("pmid", "PMID-UNKNOWN")
#                 context += text + " "
#                 if pmid != "PMID-UNKNOWN":
#                     pmids.add(pmid)

#         context = context.strip()
#         pmids = list(pmids)

#         if not context:
#             return {
#                 "message": f"Sorry, {user_name}, I don't have specific information on this topic. Please consult a healthcare professional for accurate advice.",
#                 "sources": ""
#             }

#         messages = [
#             {
#                 "role": "system",
#                 "content": f"""
#     You are a professional medical AI assistant. Based on the user's health data and the following medical literature references, provide clear, concise preventive health recommendations. The user data includes responses to the following questions: Name, Age, Gender, Zip Code, Smoking Status, Exercise Frequency, Sleep Duration, Daily Fruit/Vegetable Intake, Weekly Physical Activity, Last Doctor Visit, Blood Pressure, Do you wear seatbelts?, Do you feel safe at home?, Last Dental Checkup, BMI, Cholesterol Level.

#     - Format your response as a numbered list.
#     - After each recommendation, include a source link in this exact format:  
#     Source: https://pubmed.ncbi.nlm.nih.gov/{{pmid}}/
#     - Use these PubMed IDs as references for your response: {', '.join(pmids)}.
#     - If no PubMed ID is relevant for a recommendation, do NOT include any source line.
#     - Use simple, actionable language tailored to the user's specific health data (e.g., address high blood pressure if reported, recommend quitting smoking if the user smokes).
#     - Focus on preventive health measures based on the provided questions and user responses.
#     """
#             },
#             {
#                 "role": "user",
#                 "content": f"User data: {json.dumps(form_data, ensure_ascii=False)}\nContext: {context}"
#             }
#         ]

#         response = client.chat.completions.create(
#             model="gpt-4-turbo",
#             messages=messages,
#             max_tokens=500,
#             temperature=0.2
#         )

#         ai_response = response.choices[0].message.content.strip()
#         ai_response = ai_response.replace("Source: No source information found.", "").strip()

#         sources_text = f"Sources: {', '.join(pmids)}" if pmids else ""

#         return {"message": ai_response, "sources": sources_text}

#     except Exception as e:
#         return {"error": f"Error: {e}"}


#     except Exception as e:
#         return {"error": f"Error: {e}"}

# # Function: Process backend JSON data
# def process_backend_data(json_input_file, index):
#     """Read a JSON file with user data and generate health recommendations."""
#     try:
#         with open(json_input_file, 'r', encoding='utf-8') as f:
#             json_data = json.load(f)
#         user_name = json_data.get("user_name", "User")
#         recommendations = generate_fast_recommendations(json_data, index, user_name)
#         recommendations['user_name'] = user_name
#         return recommendations
#     except Exception as e:
#         return {"error": f"Error reading JSON file: {e}"}

# # Main execution block
# if __name__ == "__main__":
#     # Clean old cache files at the start
#     print("🧹 Cleaning old cache files...")
#     clean_cache_files(OCR_CACHE_DIR)
#     clean_cache_files(EMBEDDING_CACHE_DIR)

#     # Load PDF-PMID mapping
#     pdf_pmid_mapping = load_pdf_pmid_mapping()

#     # Create Pinecone index if it doesn't exist
#     if PINECONE_INDEX_NAME not in pc.list_indexes().names():
#         print("🛠️ Creating Pinecone index...")
#         pc.create_index(
#             name=PINECONE_INDEX_NAME,
#             dimension=1536,
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1")
#         )
#     index = pc.Index(PINECONE_INDEX_NAME)

#     # Initial data load and upload to Pinecone
#     load_and_upsert_data(index, pdf_pmid_mapping)

#     # Add new PDFs and PMIDs
#     new_pdfs = [
#         (os.path.join(PDF_FOLDER, "ABPM vs office in HTN_NEJM.pdf"), "PMID-12802026"),
#         (os.path.join(PDF_FOLDER, "OptiBP app.pdf"), "PMID-35727426")
#     ]
#     add_multiple_pdfs(new_pdfs, index, pdf_pmid_mapping)

#     # Process JSON file for health recommendations
#     json_input_file = "input.json"
#     results = process_backend_data(json_input_file, index)

#     # Print results
#     if "error" in results:
#         print(f"❌ {results['error']}")
#     else:
#         print(f"Hello {results.get('user_name', 'User')},")
#         print("Thank you for using our service.")
#         print("Based on your responses, here are preventive health recommendations just for you.")
#         print("Please take a moment to email these to yourself, a loved one, or your medical provider.")
#         print(results['message'])
#     if results['sources']:
#         sources_line = results['sources']
#     if sources_line.lower().startswith("sources: sources:"):
#         sources_line = sources_line[len("sources: "):]
#     #print(f"Sources: {sources_line}")













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

# # # Load environment variables
# # load_dotenv()
# # openai_api_key = os.getenv("OPENAI_API_KEY")
# # pinecone_api_key = os.getenv("PINECONE_API_KEY")

# # if not openai_api_key or not pinecone_api_key:
# #     print("\u274C OpenAI API Key or Pinecone API Key is missing. Check your .env file.")
# #     exit()

# # # Initialize Pinecone
# # pc = Pinecone(api_key=pinecone_api_key)
# # PINECONE_INDEX_NAME = "medical-chatbot-index"

# # # Check if Pinecone index already exists
# # if PINECONE_INDEX_NAME not in pc.list_indexes().names():
# #     pc.create_index(
# #         name=PINECONE_INDEX_NAME,
# #         dimension=1536,
# #         metric="cosine",
# #         spec=ServerlessSpec(
# #             cloud="aws",
# #             region="us-east-1"
# #         )
# #     )

# # # Load existing Pinecone index
# # index = pc.Index(PINECONE_INDEX_NAME)

# # # Folder containing PDFs
# # PDF_FOLDER = "pdf_files/"
# # pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))

# # if not pdf_files:
# #     print("\u274C No PDF files found in 'pdf_files/' folder. Please add PDFs and try again.")
# #     exit()

# # # Max chat history length
# # MAX_HISTORY_LENGTH = 10

# # # Initialize OpenAI client
# # client = openai.OpenAI(api_key=openai_api_key)

# # def convert_pdf_to_images(pdf_path):
# #     """Convert PDF pages to images for OCR."""
# #     image_paths = []
# #     try:
# #         images = convert_from_path(pdf_path, poppler_path=r"C:\\Users\\STA\\Downloads\\Release-24.08.0-0\\poppler-24.08.0\\Library\\bin")
# #         for i, image in enumerate(images):
# #             image_path = f"page_{i+1}.png"
# #             image.save(image_path, 'PNG')
# #             image_paths.append(image_path)
# #     except Exception as e:
# #         return []
# #     return image_paths

# # def extract_text_from_image(image_path):
# #     """Extract text from an image using OCR."""
# #     reader = easyocr.Reader(['en'], gpu=False)
# #     result = reader.readtext(image_path)
# #     extracted_text = " ".join([detection[1] for detection in result])
# #     return extracted_text

# # def extract_text_from_pdf(pdf_path):
# #     """Extract text directly from PDF if possible."""
# #     text = ""
# #     try:
# #         with pdfplumber.open(pdf_path) as pdf:
# #             for page in pdf.pages:
# #                 page_text = page.extract_text()
# #                 if page_text:
# #                     text += page_text + "\n"
# #     except Exception as e:
# #         return ""
# #     return text

# # def process_multiple_pdfs(pdf_files):
# #     """Process multiple PDFs, extract text, and store in Pinecone."""
# #     all_text = ""
    
# #     for pdf_file in pdf_files:
# #         text = extract_text_from_pdf(pdf_file)
# #         if not text:
# #             image_paths = convert_pdf_to_images(pdf_file)
# #             for image_path in image_paths:
# #                 text += extract_text_from_image(image_path)
# #         all_text += text + " "
    
# #     if not all_text.strip():
# #         return None
    
# #     # Split text into chunks
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# #     texts = text_splitter.split_text(all_text)
    
# #     documents = [Document(page_content=text) for text in texts]
    
# #     # Generate embeddings for each text chunk
# #     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# #     vectors = []
    
# #     for i, doc in enumerate(documents):
# #         embedding = embeddings.embed_query(doc.page_content)
# #         vectors.append((f"vec_{i}", embedding, {"text": doc.page_content}))
    
# #     # Upsert vectors in smaller batches
# #     batch_size = 100
# #     for i in range(0, len(vectors), batch_size):
# #         batch = vectors[i:i + batch_size]
# #         try:
# #             index.upsert(vectors=batch)
# #         except Exception as e:
# #             return None

# # def get_answer_from_llm(user_input, index, chat_history):
# #     """Retrieve an answer from OpenAI based on Pinecone search results while maintaining chat history."""
# #     try:
# #         query_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query(user_input)
# #         query_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        
# #         context = " ".join([match["metadata"]["text"] for match in query_results["matches"]])
        
# #         chat_history.append({"role": "user", "content": user_input})
        
# #         if len(chat_history) > MAX_HISTORY_LENGTH:
# #             chat_history = chat_history[-MAX_HISTORY_LENGTH:]
        
# #         response = client.chat.completions.create(
# #             model="gpt-4",
# #             messages=[
# #                 {"role": "system", "content": "You are a medical advisor. Provide scientific and credible answers. Mention the original source or expert name if available."},
# #                 *chat_history,
# #                 {"role": "user", "content": f"User asked: {user_input}\n\nScientific context: {context}"}
# #             ],
# #             max_tokens=500,
# #             temperature=0.8
# #         )
        
# #         ai_response = response.choices[0].message.content.strip()
        
# #         chat_history.append({"role": "assistant", "content": ai_response})
        
# #         return ai_response
# #     except Exception as e:
# #         return "Error generating response."

# # def chat_with_bot():
# #     """Run the chatbot with user interaction, maintaining conversation history."""
# #     print("🚀 Starting AI Chatbot...")
# #     chat_history = []
# #     while True:
# #         user_input = input("\n💬 You: ")
# #         if user_input.lower() in ["exit", "quit", "bye"]:
# #             print("👋 Chatbot session ended.")
# #             break
# #         ai_response = get_answer_from_llm(user_input, index, chat_history)
# #         print(f"\n🤖 AI: {ai_response}")

# # # Process PDFs and store vectors in Pinecone
# # process_multiple_pdfs(pdf_files)

# # # Start the chatbot
# # chat_with_bot()








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

# # # Load environment variables
# # load_dotenv()
# # openai_api_key = os.getenv("OPENAI_API_KEY")
# # pinecone_api_key = os.getenv("PINECONE_API_KEY")

# # if not openai_api_key or not pinecone_api_key:
# #     print("\u274C OpenAI API Key or Pinecone API Key is missing. Check your .env file.")
# #     exit()

# # # Initialize Pinecone
# # pc = Pinecone(api_key=pinecone_api_key)
# # PINECONE_INDEX_NAME = "medical-chatbot-index"

# # # Check if Pinecone index already exists
# # if PINECONE_INDEX_NAME not in pc.list_indexes().names():
# #     pc.create_index(
# #         name=PINECONE_INDEX_NAME,
# #         dimension=1536,
# #         metric="cosine",
# #         spec=ServerlessSpec(
# #             cloud="aws",
# #             region="us-east-1"
# #         )
# #     )

# # # Load existing Pinecone index
# # index = pc.Index(PINECONE_INDEX_NAME)

# # # Folder containing PDFs
# # PDF_FOLDER = "pdf_files/"
# # pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))

# # if not pdf_files:
# #     print("\u274C No PDF files found in 'pdf_files/' folder. Please add PDFs and try again.")
# #     exit()

# # # Max chat history length
# # MAX_HISTORY_LENGTH = 10

# # # Initialize OpenAI client
# # client = openai.OpenAI(api_key=openai_api_key)

# # def convert_pdf_to_images(pdf_path):
# #     """Convert PDF pages to images for OCR."""
# #     image_paths = []
# #     try:
# #         images = convert_from_path(pdf_path, poppler_path=r"C:\\Users\\STA\\Downloads\\Release-24.08.0-0\\poppler-24.08.0\\Library\\bin")
# #         for i, image in enumerate(images):
# #             image_path = f"page_{i+1}.png"
# #             image.save(image_path, 'PNG')
# #             image_paths.append(image_path)
# #     except Exception as e:
# #         return []
# #     return image_paths

# # def extract_text_from_image(image_path):
# #     """Extract text from an image using OCR."""
# #     reader = easyocr.Reader(['en'], gpu=False)
# #     result = reader.readtext(image_path)
# #     extracted_text = " ".join([detection[1] for detection in result])
# #     return extracted_text

# # def extract_text_from_pdf(pdf_path):
# #     """Extract text directly from PDF if possible."""
# #     text = ""
# #     try:
# #         with pdfplumber.open(pdf_path) as pdf:
# #             for page in pdf.pages:
# #                 page_text = page.extract_text()
# #                 if page_text:
# #                     text += page_text + "\n"
# #     except Exception as e:
# #         return ""
# #     return text

# # def process_multiple_pdfs(pdf_files):
# #     """Process multiple PDFs, extract text, and store in Pinecone."""
# #     all_text = ""
    
# #     for pdf_file in pdf_files:
# #         text = extract_text_from_pdf(pdf_file)
# #         if not text:
# #             image_paths = convert_pdf_to_images(pdf_file)
# #             for image_path in image_paths:
# #                 text += extract_text_from_image(image_path)
# #         all_text += text + " "
    
# #     if not all_text.strip():
# #         return None
    
# #     # Split text into chunks
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# #     texts = text_splitter.split_text(all_text)
    
# #     documents = [Document(page_content=text) for text in texts]
    
# #     # Generate embeddings for each text chunk
# #     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# #     vectors = []
    
# #     for i, doc in enumerate(documents):
# #         embedding = embeddings.embed_query(doc.page_content)
# #         vectors.append((f"vec_{i}", embedding, {"text": doc.page_content}))
    
# #     # Upsert vectors in smaller batches
# #     batch_size = 100
# #     for i in range(0, len(vectors), batch_size):
# #         batch = vectors[i:i + batch_size]
# #         try:
# #             index.upsert(vectors=batch)
# #         except Exception as e:
# #             return None

# # def get_answer_from_llm(user_input, index, chat_history):
# #     """Retrieve an answer from OpenAI based on Pinecone search results while maintaining chat history."""
# #     try:
# #         query_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query(user_input)
# #         query_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        
# #         context = " ".join([match["metadata"]["text"] for match in query_results["matches"]])
        
# #         chat_history.append({"role": "user", "content": user_input})
        
# #         if len(chat_history) > MAX_HISTORY_LENGTH:
# #             chat_history = chat_history[-MAX_HISTORY_LENGTH:]
        
# #         response = client.chat.completions.create(
# #             model="gpt-4",
# #             messages=[
# #                 {"role": "system", "content": "You are a medical advisor. Provide scientific and credible answers based on the given context. "
# #                 "Ensure that all responses are structured in an easy-to-read format such as bullet points."
# #                 "Always mention the original source (e.g., research papers, expert recommendations, PMID/DOI links) if available. "
# #                 "If no source is provided, clarify that the response is based on general medical knowledge."},
# #                 *chat_history,
# #                 {"role": "user", "content": f"User asked: {user_input}\n\nScientific context: {context}"}
# #             ],
# #             max_tokens=500,
# #             temperature=0.8
# #         )
        
# #         ai_response = response.choices[0].message.content.strip()
        
# #         chat_history.append({"role": "assistant", "content": ai_response})
        
# #         return ai_response
# #     except Exception as e:
# #         return "Error generating response."

# # def chat_with_bot():
# #     """Run the chatbot with user interaction, maintaining conversation history."""
# #     print("🚀 Starting AI Chatbot...")
# #     chat_history = []
# #     while True:
# #         user_input = input("\n💬 You: ")
# #         if user_input.lower() in ["exit", "quit", "bye"]:
# #             print("👋 Chatbot session ended.")
# #             break
# #         ai_response = get_answer_from_llm(user_input, index, chat_history)
# #         print(f"\n🤖 AI: {ai_response}")

# # # Process PDFs and store vectors in Pinecone
# # process_multiple_pdfs(pdf_files)

# # # Start the chatbot
# # chat_with_bot()





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

# # Load environment variables
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")

# if not openai_api_key or not pinecone_api_key:
#     print("❌ Error: Missing API keys. Please set OPENAI_API_KEY and PINECONE_API_KEY in .env file.")
#     exit()

# # Initialize Pinecone
# pc = Pinecone(api_key=pinecone_api_key)
# PINECONE_INDEX_NAME = "medical-chatbot-index"

# # Folder containing PDFs and processed data
# PDF_FOLDER = "pdf_files/"
# PROCESSED_DATA_FILE = "processed_data.pkl"
# pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))

# if not pdf_files:
#     print("❌ No PDF files found in the folder. Please add PDFs to 'pdf_files/' directory.")
#     exit()

# # Max chat history length
# MAX_HISTORY_LENGTH = 10

# # Initialize OpenAI client
# client = openai.OpenAI(api_key=openai_api_key)

# # PDF to PMID mapping with normalized paths (hardcoded PMIDs)
# pdf_pmid_mapping = {
#     os.path.normpath(os.path.join(PDF_FOLDER, "ABPM vs office in HTN_NEJM.pdf")): "PMID-123",
#     os.path.normpath(os.path.join(PDF_FOLDER, "Clinical Cardiology - October 1992 - Pickering - Ambulatory blood pressure monitoring An historical perspective.pdf")): "PMID-456",
#     os.path.normpath(os.path.join(PDF_FOLDER, "Cost savings of ABPM.pdf")): "PMID-789",
#     os.path.normpath(os.path.join(PDF_FOLDER, "jamacardiology_blood_2022_oi_220067_1672335582.056.pdf")): "PMID-1011",
#     os.path.normpath(os.path.join(PDF_FOLDER, "Lee2022_clinical decisions remote BPM.pdf")): "PMID-1213",
#     os.path.normpath(os.path.join(PDF_FOLDER, "NIGHTIME ABPM nightime dippers.risers.pdf")): "PMID-1415",
#     os.path.normpath(os.path.join(PDF_FOLDER, "OptiBP app.pdf")): "PMID-1617",
#     os.path.normpath(os.path.join(PDF_FOLDER, "photophytoelectric signal for BP.pdf")): "PMID-1819",
# }

# def convert_pdf_to_images(pdf_path):
#     """Convert PDF pages to images for OCR."""
#     image_paths = []
#     try:
#         images = convert_from_path(pdf_path, poppler_path=r"C:\\Users\\STA\\Downloads\\Release-24.08.0-0\\poppler-24.08.0\\Library\\bin")
#         for i, image in enumerate(images):
#             image_path = f"page_{i+1}.png"
#             image.save(image_path, 'PNG')
#             image_paths.append(image_path)
#     except Exception as e:
#         print(f"⚠️ Error converting {pdf_path} to images: {e}")
#     return image_paths

# def extract_text_from_image(image_path):
#     """Extract text from an image using OCR."""
#     try:
#         reader = easyocr.Reader(['en'], gpu=False)
#         result = reader.readtext(image_path)
#         extracted_text = " ".join([detection[1] for detection in result])
#         return extracted_text
#     except Exception as e:
#         print(f"⚠️ Error extracting text from image {image_path}: {e}")
#         return ""

# def extract_text_from_pdf(pdf_path):
#     """Extract text from all pages of a PDF."""
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

# def process_and_save_pdfs(pdf_files):
#     """Process PDFs and save combined text with PMID metadata."""
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

# def load_and_upsert_data(index):
#     """Load processed data and upsert to Pinecone with PMID metadata."""
#     if os.path.exists(PROCESSED_DATA_FILE):
#         with open(PROCESSED_DATA_FILE, "rb") as f:
#             processed_data = pickle.load(f)
        
#         stats = index.describe_index_stats()
#         if stats['total_vector_count'] == 0:
#             print("📤 Uploading data to Pinecone...")
#             for pdf_name, data in processed_data.items():
#                 combined_text = data["text"]
#                 metadata = data["metadata"]
#                 if combined_text.strip():
#                     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#                     texts = text_splitter.split_text(combined_text)
#                     documents = [Document(page_content=text) for text in texts]
#                     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#                     vectors = []
#                     for i, doc in enumerate(documents):
#                         embedding = embeddings.embed_query(doc.page_content)
#                         vectors.append((
#                             f"vec_{i}_{metadata['pmid']}",
#                             embedding,
#                             {
#                                 "text": doc.page_content,
#                                 "pmid": metadata["pmid"]
#                             }
#                         ))
#                     batch_size = 50
#                     for i in range(0, len(vectors), batch_size):
#                         batch = vectors[i:i + batch_size]
#                         try:
#                             index.upsert(vectors=batch)
#                         except Exception as e:
#                             print(f"⚠️ Error upserting batch to Pinecone: {e}")
#             print("✅ Data uploaded to Pinecone.")
#     else:
#         process_and_save_pdfs(pdf_files)
#         load_and_upsert_data(index)

# def get_answer_from_llm(user_input, index, chat_history):
#     """Retrieve an answer from OpenAI based on Pinecone search results."""
#     try:
#         query_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query(user_input)
#         query_results = index.query(vector=query_embedding, top_k=10, include_metadata=True)
#         context = ""
#         pmids = set()
#         for match in query_results["matches"]:
#             metadata = match.get("metadata", {})
#             text = metadata.get("text", "")
#             pmid = metadata.get("pmid", "PMID-UNKNOWN")
#             if pmid in pdf_pmid_mapping.values():
#                 context += text + " "
#                 pmids.add(pmid)
#         context = context.strip()
#         pmids = list(pmids)
        
#         chat_history.append({"role": "user", "content": user_input})
#         if len(chat_history) > MAX_HISTORY_LENGTH:
#             chat_history = chat_history[-MAX_HISTORY_LENGTH:]
        
#         if not context:
#             ai_response = "Sorry, I don’t have specific information on this topic in my database, as it focuses primarily on blood pressure monitoring. For accurate advice, please consult a healthcare professional."
#             chat_history.append({"role": "assistant", "content": ai_response})
#             return f"{ai_response}\n\nNo source information found."
        
#         messages = [
#             {
#                 "role": "system",
#                 "content": """
#                 You are a medical advisor. Provide scientific and credible answers based ONLY on the given context below. Structure your response as a list of bullet points summarizing the main findings and including specific details or data from the context if available. Do not use headings like 'Key Points' or 'Details'. After the bullet points, add a 'Sources' section using the format below.

#                 For citations:
#                 - Extract 'title', 'journal', and 'DOI' from the provided context if available.
#                 - Use the PMIDs provided separately (listed as 'PMIDs' below) to associate with the citation.
#                 - Format each source as: 'Source: {title}, {journal}, {doi}, {pmid}' if title, journal, and DOI are found in the context.
#                 - If title, journal, or DOI are not found in the context, use: 'PMID: {pmid}'.
#                 - Only include PMIDs that directly relate to the user’s question and context; ignore irrelevant PMIDs.
#                 - Avoid duplicate citations; list each unique source only once.
#                 - If multiple PMIDs match the same citation, list it once with all relevant PMIDs (e.g., 'Source: {title}, {journal}, {doi}, PMID: {pmid1}, {pmid2}').
#                 - If no relevant information is found, respond exactly with: 'Sorry, I don’t have specific information on this topic in my database, as it focuses primarily on blood pressure monitoring. For accurate advice, please consult a healthcare professional.' followed by 'No source information found.' on a new line.
#                 """
#             },
#             *chat_history,
#             {
#                 "role": "user",
#                 "content": f"User asked: {user_input}\n\nScientific context: {context}\n\nPMIDs: {', '.join(pmids)}"
#             }
#         ]
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=messages,
#             max_tokens=1000,
#             temperature=0.2
#         )
#         ai_response = response.choices[0].message.content.strip()
#         chat_history.append({"role": "assistant", "content": ai_response})
#         return ai_response
#     except Exception as e:
#         return f"Error generating response: {e}"

# def chat_with_bot(index):
#     """Run the chatbot with user interaction."""
#     print("🚀 Welcome to the Medical Chatbot! Ask me anything.")
#     print("ℹ️ Type 'exit', 'quit', or 'bye' to end the session.")
#     chat_history = []
#     while True:
#         user_input = input("\n💬 You: ")
#         if user_input.lower() in ["exit", "quit", "bye"]:
#             print("👋 Thanks for using the Medical Chatbot. Goodbye!")
#             break
#         if not user_input.strip():
#             print("🤔 Please ask something!")
#             continue
#         ai_response = get_answer_from_llm(user_input, index, chat_history)
#         print(f"\n🤖 Answer:\n{ai_response}")

# # Main execution logic
# if PINECONE_INDEX_NAME not in pc.list_indexes().names():
#     print("🛠️ Creating Pinecone index...")
#     pc.create_index(
#         name=PINECONE_INDEX_NAME,
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )
# index = pc.Index(PINECONE_INDEX_NAME)

# load_and_upsert_data(index)
# chat_with_bot(index)
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

# # Load environment variables
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")

# if not openai_api_key or not pinecone_api_key:
#     print("\u274C OpenAI API Key or Pinecone API Key is missing. Check your .env file.")
#     exit()

# # Initialize Pinecone
# pc = Pinecone(api_key=pinecone_api_key)
# PINECONE_INDEX_NAME = "medical-chatbot-index"

# # Check if Pinecone index already exists
# if PINECONE_INDEX_NAME not in pc.list_indexes().names():
#     pc.create_index(
#         name=PINECONE_INDEX_NAME,
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )

# # Load existing Pinecone index
# index = pc.Index(PINECONE_INDEX_NAME)

# # Folder containing PDFs
# PDF_FOLDER = "pdf_files/"
# pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))

# if not pdf_files:
#     print("\u274C No PDF files found in 'pdf_files/' folder. Please add PDFs and try again.")
#     exit()

# # Max chat history length
# MAX_HISTORY_LENGTH = 10

# # Initialize OpenAI client
# client = openai.OpenAI(api_key=openai_api_key)

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
#         return []
#     return image_paths

# def extract_text_from_image(image_path):
#     """Extract text from an image using OCR."""
#     reader = easyocr.Reader(['en'], gpu=False)
#     result = reader.readtext(image_path)
#     extracted_text = " ".join([detection[1] for detection in result])
#     return extracted_text

# def extract_text_from_pdf(pdf_path):
#     """Extract text directly from PDF if possible."""
#     text = ""
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text + "\n"
#     except Exception as e:
#         return ""
#     return text

# def process_multiple_pdfs(pdf_files):
#     """Process multiple PDFs, extract text, and store in Pinecone."""
#     all_text = ""
    
#     for pdf_file in pdf_files:
#         text = extract_text_from_pdf(pdf_file)
#         if not text:
#             image_paths = convert_pdf_to_images(pdf_file)
#             for image_path in image_paths:
#                 text += extract_text_from_image(image_path)
#         all_text += text + " "
    
#     if not all_text.strip():
#         return None
    
#     # Split text into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     texts = text_splitter.split_text(all_text)
    
#     documents = [Document(page_content=text) for text in texts]
    
#     # Generate embeddings for each text chunk
#     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     vectors = []
    
#     for i, doc in enumerate(documents):
#         embedding = embeddings.embed_query(doc.page_content)
#         vectors.append((f"vec_{i}", embedding, {"text": doc.page_content}))
    
#     # Upsert vectors in smaller batches
#     batch_size = 100
#     for i in range(0, len(vectors), batch_size):
#         batch = vectors[i:i + batch_size]
#         try:
#             index.upsert(vectors=batch)
#         except Exception as e:
#             return None

# def get_answer_from_llm(user_input, index, chat_history):
#     """Retrieve an answer from OpenAI based on Pinecone search results while maintaining chat history."""
#     try:
#         query_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query(user_input)
#         query_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        
#         context = " ".join([match["metadata"]["text"] for match in query_results["matches"]])
        
#         chat_history.append({"role": "user", "content": user_input})
        
#         if len(chat_history) > MAX_HISTORY_LENGTH:
#             chat_history = chat_history[-MAX_HISTORY_LENGTH:]
        
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a medical advisor. Provide scientific and credible answers based on the given context. "
#                 "Ensure that all responses are structured in an easy-to-read format such as bullet points."
#                 "Always mention the original source (e.g., research papers, expert recommendations, PMID/DOI links) if available. "
#                 "If no source is provided, clarify that the response is based on general medical knowledge."},
#                 *chat_history,
#                 {"role": "user", "content": f"User asked: {user_input}\n\nScientific context: {context}"}
#             ],
#             max_tokens=500,
#             temperature=0.3
#         )
        
#         ai_response = response.choices[0].message.content.strip()
        
#         chat_history.append({"role": "assistant", "content": ai_response})
        
#         return ai_response
#     except Exception as e:
#         return "Error generating response."

# def chat_with_bot():
#     """Run the chatbot with user interaction, maintaining conversation history."""
#     print("🚀 Starting AI Chatbot...")
#     chat_history = []
#     while True:
#         user_input = input("\n💬 You: ")
#         if user_input.lower() in ["exit", "quit", "bye"]:
#             print("👋 Chatbot session ended.")
#             break
#         ai_response = get_answer_from_llm(user_input, index, chat_history)
#         print(f"\n🤖 AI: {ai_response}")

# # Process PDFs and store vectors in Pinecone
# process_multiple_pdfs(pdf_files)

# # Start the chatbot
# chat_with_bot()










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

# # Load environment variables
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")

# if not openai_api_key or not pinecone_api_key:
#     print("\u274C OpenAI API Key or Pinecone API Key is missing. Check your .env file.")
#     exit()

# # Initialize Pinecone
# pc = Pinecone(api_key=pinecone_api_key)
# PINECONE_INDEX_NAME = "medical-chatbot-index"

# # Check if Pinecone index already exists
# if PINECONE_INDEX_NAME not in pc.list_indexes().names():
#     pc.create_index(
#         name=PINECONE_INDEX_NAME,
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )

# # Load existing Pinecone index
# index = pc.Index(PINECONE_INDEX_NAME)

# # Folder containing PDFs
# PDF_FOLDER = "pdf_files/"
# pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))

# if not pdf_files:
#     print("\u274C No PDF files found in 'pdf_files/' folder. Please add PDFs and try again.")
#     exit()

# # PMID Manual Mapping
# PMID_MAPPING = {
#     "ABPM vs office in HTN_NEJM.pdf": "PMID-567890",
#     "diabetes_research.pdf": "PMID-123456",
#     "hypertension_study.pdf": "PMID-789012",
#     "nutrition_study.pdf": "PMID-345678",
#     "cancer_treatment.pdf": "PMID-901234",
#     "mental_health_analysis.pdf": "PMID-654321",
#     "stroke_prevention.pdf": "PMID-432109",
#     "covid19_impact.pdf": "PMID-876543",
# }

# # Max chat history length
# MAX_HISTORY_LENGTH = 10

# # Initialize OpenAI client
# client = openai.OpenAI(api_key=openai_api_key)

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
#         return []
#     return image_paths

# def extract_text_from_image(image_path):
#     """Extract text from an image using OCR."""
#     reader = easyocr.Reader(['en'], gpu=False)
#     result = reader.readtext(image_path)
#     extracted_text = " ".join([detection[1] for detection in result])
#     return extracted_text

# def extract_text_from_pdf(pdf_path):
#     """Extract text directly from PDF if possible."""
#     text = ""
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text + "\n"
#     except Exception as e:
#         return ""
#     return text

# def process_multiple_pdfs(pdf_files):
#     """Process multiple PDFs, extract text, and store in Pinecone."""
    
#     for pdf_file in pdf_files:
#         pdf_name = os.path.basename(pdf_file)  # Extract file name from path
#         pmid = PMID_MAPPING.get(pdf_name, "PMID-000000")  # Fetch PMID, default if not found
        
#         text = extract_text_from_pdf(pdf_file)
#         if not text:
#             image_paths = convert_pdf_to_images(pdf_file)
#             for image_path in image_paths:
#                 text += extract_text_from_image(image_path)

#         if not text.strip():
#             continue  # Skip empty PDFs
        
#         # Split text into chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         texts = text_splitter.split_text(text)

#         documents = [Document(page_content=t) for t in texts]

#         # Generate embeddings for each text chunk
#         embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#         vectors = []

#         for i, doc in enumerate(documents):
#             embedding = embeddings.embed_query(doc.page_content)
#             vectors.append((f"vec_{i}", embedding, {"text": doc.page_content, "PMID": pmid}))  # Add PMID here

#         # Upsert vectors in smaller batches
#         batch_size = 100
#         for i in range(0, len(vectors), batch_size):
#             batch = vectors[i:i + batch_size]
#             try:
#                 index.upsert(vectors=batch)
#             except Exception as e:
#                 return None

# def get_answer_from_llm(user_input, index, chat_history):
#     """Retrieve an answer from OpenAI based on Pinecone search results while maintaining chat history."""
#     try:
#         query_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query(user_input)
#         query_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

#         sources = set()  # Use a set to avoid duplicate sources
#         context = ""

#         for match in query_results["matches"]:
#             text = match["metadata"]["text"]
#             pmid = match["metadata"].get("PMID", "PMID-Unknown")  # Fetch PMID correctly
#             context += text + "\n\n"
            
#             # Store formatted source properly
#             sources.add(f"{text[:100]}... (PMID-{pmid})")  # Show only first 100 chars for brevity

#         chat_history.append({"role": "user", "content": user_input})

#         if len(chat_history) > MAX_HISTORY_LENGTH:
#             chat_history = chat_history[-MAX_HISTORY_LENGTH:]

#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a medical advisor. Provide scientific and credible answers based on the given context. "
#                 "Ensure that all responses are structured in an easy-to-read format such as bullet points."
#                 "Always mention the original source (e.g., research papers, expert recommendations, PMID/DOI links) if available. "
#                 "If no source is provided, clarify that the response is based on general medical knowledge."},
#                 *chat_history,
#                 {"role": "user", "content": f"User asked: {user_input}\n\nScientific context: {context}"}
#             ],
#             max_tokens=500,
#             temperature=0.3
#         )

#         ai_response = response.choices[0].message.content.strip()

#         # Attach sources in the correct format
#         if sources:
#             ai_response += "\n\nSources:\n" + "\n".join(sources)

#         chat_history.append({"role": "assistant", "content": ai_response})

#         return ai_response
#     except Exception as e:
#         return "Error generating response."

# def chat_with_bot():
#     """Run the chatbot with user interaction, maintaining conversation history."""
#     print("🚀 Starting AI Chatbot...")
#     chat_history = []
#     while True:
#         user_input = input("\n💬 You: ")
#         if user_input.lower() in ["exit", "quit", "bye"]:
#             print("👋 Chatbot session ended.")
#             break
#         ai_response = get_answer_from_llm(user_input, index, chat_history)
#         print(f"\n🤖 AI: {ai_response}")

# # Process PDFs and store vectors in Pinecone
# process_multiple_pdfs(pdf_files)

# # Start the chatbot
# chat_with_bot()






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

# # Load environment variables
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")

# if not openai_api_key or not pinecone_api_key:
#     print("\u274C OpenAI API Key or Pinecone API Key is missing. Check your .env file.")
#     exit()

# # Initialize Pinecone
# pc = Pinecone(api_key=pinecone_api_key)
# PINECONE_INDEX_NAME = "medical-chatbot-index"

# # Check if Pinecone index already exists
# if PINECONE_INDEX_NAME not in pc.list_indexes().names():
#     pc.create_index(
#         name=PINECONE_INDEX_NAME,
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )

# # Load existing Pinecone index
# index = pc.Index(PINECONE_INDEX_NAME)

# # Folder containing PDFs
# PDF_FOLDER = "pdf_files/"
# pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))

# if not pdf_files:
#     print("\u274C No PDF files found in 'pdf_files/' folder. Please add PDFs and try again.")
#     exit()

# # Max chat history length
# MAX_HISTORY_LENGTH = 10

# # Initialize OpenAI client
# client = openai.OpenAI(api_key=openai_api_key)

# # PDF to PMID mapping (আপনি এখানে নিজের মতো করে PMID সেট করুন) 
# pdf_pmid_mapping = {
#     r"C:\Users\STA\Desktop\dr\pdf_files\ABPM vs office in HTN_NEJM.pdf": "PMID-567890",
#     r"C:\Users\STA\Desktop\dr\pdf_files\Clinical Cardiology - October 1992 - Pickering - Ambulatory blood pressure monitoring An historical perspective.pdf": "PMID-123456",
#     r"C:\Users\STA\Desktop\dr\pdf_files\Cost savings of ABPM.pdf": "PMID-789012",
#     r"C:\Users\STA\Desktop\dr\pdf_files\jamacardiology_blood_2022_oi_220067_1672335582.056.pdf": "PMID-345678",
#     r"C:\Users\STA\Desktop\dr\pdf_files\Lee2022_clinical decisions remote BPM.pdf": "PMID-901234",
#     r"C:\Users\STA\Desktop\dr\pdf_files\NIGHTIME ABPM nightime dippers.risers.pdf": "PMID-654321",
#     r"C:\Users\STA\Desktop\dr\pdf_files\OptiBP app.pdf": "PMID-432109",
#     r"C:\Users\STA\Desktop\dr\pdf_files\photophytoelectric signal for BP.pdf": "PMID-876543",
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
#         return []
#     return image_paths

# def extract_text_from_image(image_path):
#     """Extract text from an image using OCR."""
#     reader = easyocr.Reader(['en'], gpu=False)
#     result = reader.readtext(image_path)
#     extracted_text = " ".join([detection[1] for detection in result])
#     return extracted_text

# def extract_text_from_pdf(pdf_path):
#     """Extract text directly from PDF if possible."""
#     text = ""
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text + "\n"
#     except Exception as e:
#         return ""
#     return text

# def process_multiple_pdfs(pdf_files):
#     """Process multiple PDFs, extract text, and store in Pinecone with custom PMIDs."""
#     for pdf_file in pdf_files:
#         pdf_name = os.path.basename(pdf_file)  # PDF ফাইলের নাম বের করুন
#         pmid = pdf_pmid_mapping.get(pdf_name, "PMID-UNKNOWN")  # সংশ্লিষ্ট PMID বের করুন

#         text = extract_text_from_pdf(pdf_file)
#         if not text:
#             image_paths = convert_pdf_to_images(pdf_file)
#             for image_path in image_paths:
#                 text += extract_text_from_image(image_path)

#         if text.strip():
#             # Split text into chunks
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#             texts = text_splitter.split_text(text)
#             documents = [Document(page_content=text) for text in texts]

#             # Generate embeddings for each text chunk
#             embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#             vectors = []

#             for i, doc in enumerate(documents):
#                 embedding = embeddings.embed_query(doc.page_content)
#                 vectors.append((f"vec_{i}", embedding, {"text": doc.page_content, "pmid": pmid}))

#             # Upsert vectors in smaller batches
#             batch_size = 100
#             for i in range(0, len(vectors), batch_size):
#                 batch = vectors[i:i + batch_size]
#                 try:
#                     index.upsert(vectors=batch)
#                 except Exception as e:
#                     print(f"Error upserting vectors for {pdf_name}: {e}")

# def get_answer_from_llm(user_input, index, chat_history):
#     """Retrieve an answer from OpenAI based on Pinecone search results while maintaining chat history."""
#     try:
#         query_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query(user_input)
#         query_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

#         context = " ".join([match["metadata"]["text"] for match in query_results["matches"]])
#         pmids = ", ".join(set([match["metadata"]["pmid"] for match in query_results["matches"]]))  # ইউনিক PMID সংগ্রহ

#         chat_history.append({"role": "user", "content": user_input})

#         if len(chat_history) > MAX_HISTORY_LENGTH:
#             chat_history = chat_history[-MAX_HISTORY_LENGTH:]

#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a medical advisor. Provide scientific and credible answers based on the given context. "
#                 "Ensure that all responses are structured in an easy-to-read format such as bullet points. "
#                 "Always mention the original source (e.g., research papers, expert recommendations, PMID/DOI links) if available. "
#                 "If no source is provided, clarify that the response is based on general medical knowledge."},
#                 *chat_history,
#                 {"role": "user", "content": f"User asked: {user_input}\n\nScientific context: {context}"}
#             ],
#             max_tokens=500,
#             temperature=0.2
#         )

#         ai_response = response.choices[0].message.content.strip()
#         ai_response_with_pmid = f"{ai_response} ({pmids})"  # PMID যোগ করুন

#         chat_history.append({"role": "assistant", "content": ai_response_with_pmid})

#         return ai_response_with_pmid
#     except Exception as e:
#         return "Error generating response."

# def chat_with_bot():
#     """Run the chatbot with user interaction, maintaining conversation history."""
#     print("🚀 Starting AI Chatbot...")
#     chat_history = []
#     while True:
#         user_input = input("\n💬 You: ")
#         if user_input.lower() in ["exit", "quit", "bye"]:
#             print("👋 Chatbot session ended.")
#             break
#         ai_response = get_answer_from_llm(user_input, index, chat_history)
#         print(f"\n🤖 AI: {ai_response}")

# # Process PDFs and store vectors in Pinecone
# process_multiple_pdfs(pdf_files)

# # Start the chatbot
# chat_with_bot()










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

# # Load environment variables
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")

# if not openai_api_key or not pinecone_api_key:
#     print("❌ OpenAI API Key or Pinecone API Key not found. Check your .env file.")
#     exit()

# # Initialize Pinecone
# pc = Pinecone(api_key=pinecone_api_key)
# PINECONE_INDEX_NAME = "medical-chatbot-index"

# if PINECONE_INDEX_NAME not in pc.list_indexes().names():
#     print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
#     pc.create_index(
#         name=PINECONE_INDEX_NAME,
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )

# index = pc.Index(PINECONE_INDEX_NAME)
# print(f"Pinecone index loaded: {PINECONE_INDEX_NAME}")

# # Folder containing PDFs
# PDF_FOLDER = "pdf_files/"
# pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))

# if not pdf_files:
#     print("❌ No PDF files found in 'pdf_files/' folder. Add PDFs and try again.")
#     exit()
# else:
#     print(f"Found PDF files: {pdf_files}")

# # Max chat history length
# MAX_HISTORY_LENGTH = 10

# # Initialize OpenAI client
# client = openai.OpenAI(api_key=openai_api_key)

# # PDF to PMID mapping with normalized paths
# pdf_pmid_mapping = {
#     os.path.normpath(os.path.join(PDF_FOLDER, "ABPM vs office in HTN_NEJM.pdf")): "PMID-567890",
#     os.path.normpath(os.path.join(PDF_FOLDER, "Clinical Cardiology - October 1992 - Pickering - Ambulatory blood pressure monitoring An historical perspective.pdf")): "PMID-123456",
#     os.path.normpath(os.path.join(PDF_FOLDER, "Cost savings of ABPM.pdf")): "PMID-789012",
#     os.path.normpath(os.path.join(PDF_FOLDER, "jamacardiology_blood_2022_oi_220067_1672335582.056.pdf")): "PMID-345678",
#     os.path.normpath(os.path.join(PDF_FOLDER, "Lee2022_clinical decisions remote BPM.pdf")): "PMID-901234",
#     os.path.normpath(os.path.join(PDF_FOLDER, "NIGHTIME ABPM nightime dippers.risers.pdf")): "PMID-654321",
#     os.path.normpath(os.path.join(PDF_FOLDER, "OptiBP app.pdf")): "PMID-432109",
#     os.path.normpath(os.path.join(PDF_FOLDER, "photophytoelectric signal for BP.pdf")): "PMID-876543",
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
#         print(f"Converted {pdf_path} to images: {len(image_paths)} pages")
#     except Exception as e:
#         print(f"Error converting PDF to images: {str(e)}")
#     return image_paths

# def extract_text_from_image(image_path):
#     """Extract text from an image using OCR."""
#     try:
#         reader = easyocr.Reader(['en'], gpu=False)
#         result = reader.readtext(image_path)
#         extracted_text = " ".join([detection[1] for detection in result])
#         print(f"OCR text from {image_path}: {extracted_text[:200]}...")
#         return extracted_text
#     except Exception as e:
#         print(f"OCR error: {str(e)}")
#         return ""

# def extract_text_from_pdf(pdf_path):
#     """Extract text directly from PDF if possible."""
#     text = ""
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text + "\n"
#         print(f"Text extracted from {pdf_path}: {text[:200]}...")
#     except Exception as e:
#         print(f"Error extracting text from PDF: {str(e)}")
#     return text

# def process_multiple_pdfs(pdf_files):
#     """Process multiple PDFs and store in Pinecone."""
#     for pdf_file in pdf_files:
#         pdf_name = os.path.normpath(pdf_file)  # Normalize path for consistency
#         pmid = pdf_pmid_mapping.get(pdf_name, "PMID-UNKNOWN")
#         print(f"Processing: {pdf_name} (PMID: {pmid})")

#         text = extract_text_from_pdf(pdf_file)
#         if not text.strip():
#             print(f"No text extracted from {pdf_name}, running OCR...")
#             image_paths = convert_pdf_to_images(pdf_file)
#             for image_path in image_paths:
#                 text += extract_text_from_image(image_path)

#         if text.strip():
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#             texts = text_splitter.split_text(text)
#             documents = [Document(page_content=text) for text in texts]
#             print(f"Created {len(documents)} chunks from {pdf_name}")

#             embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#             vectors = []
#             for i, doc in enumerate(documents):
#                 embedding = embeddings.embed_query(doc.page_content)
#                 vectors.append((f"vec_{i}", embedding, {"text": doc.page_content, "pmid": pmid}))

#             batch_size = 50
#             for i in range(0, len(vectors), batch_size):
#                 batch = vectors[i:i + batch_size]
#                 try:
#                     index.upsert(vectors=batch)
#                     print(f"Stored {len(batch)} vectors in Pinecone for {pdf_name}")
#                 except Exception as e:
#                     print(f"Error storing vectors in Pinecone: {str(e)}")
#         else:
#             print(f"No text extracted from {pdf_name}")

#     stats = index.describe_index_stats()
#     print(f"Pinecone status: {stats}")

# def get_answer_from_llm(user_input, index, chat_history):
#     """Retrieve an answer from OpenAI based on Pinecone search results."""
#     try:
#         print(f"Question: {user_input}")
#         query_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query(user_input)
#         query_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
#         print(f"Matches retrieved from Pinecone: {len(query_results['matches'])}")

#         # Safely handle metadata
#         context = ""
#         pmids = set()
#         for match in query_results["matches"]:
#             metadata = match.get("metadata", {})
#             text = metadata.get("text", "")
#             pmid = metadata.get("pmid", "PMID-UNKNOWN")
#             context += text + " "
#             pmids.add(pmid)

#         context = context.strip()
#         valid_pmids = [pmid for pmid in pmids if pmid != "PMID-UNKNOWN"]  # Filter out PMID-UNKNOWN
#         pmids_str = ", ".join(valid_pmids) if valid_pmids else "No specific PMID available"
#         print(f"Retrieved context: {context[:500]}...")
#         print(f"Found PMIDs: {pmids_str}")

#         if not context:
#             return "No relevant information found."

#         chat_history.append({"role": "user", "content": user_input})
#         if len(chat_history) > MAX_HISTORY_LENGTH:
#             chat_history = chat_history[-MAX_HISTORY_LENGTH:]

#         messages = [
#             {"role": "system", "content": "You are a medical advisor. Provide scientific and credible answers based on the given context. "
#              "Ensure that all responses are structured in an easy-to-read format such as bullet points. "
#              "Always mention the original source (e.g., research papers, expert recommendations, PMID/DOI links) if available. "
#              "If no source is provided, clarify that the response is based on general medical knowledge."},
#             *chat_history,
#             {"role": "user", "content": f"User asked: {user_input}\n\nScientific context: {context}"}
#         ]
#         print(f"Message sent to OpenAI: {messages[-1]}")

#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=messages,
#             max_tokens=500,
#             temperature=0.2
#         )

#         ai_response = response.choices[0].message.content.strip()
#         ai_response_with_pmid = f"{ai_response} ({pmids_str})"
#         chat_history.append({"role": "assistant", "content": ai_response_with_pmid})
#         return ai_response_with_pmid
#     except Exception as e:
#         print(f"Error generating response: {str(e)}")
#         return f"Error generating response: {str(e)}"

# def chat_with_bot():
#     """Run the chatbot with user interaction."""
#     print("🚀 AI chatbot is starting...")
#     chat_history = []
#     while True:
#         user_input = input("\n💬 You: ")
#         if user_input.lower() in ["exit", "quit", "bye"]:
#             print("👋 Chatbot session ended.")
#             break
#         ai_response = get_answer_from_llm(user_input, index, chat_history)
#         print(f"\n🤖 AI: {ai_response}")

# # Process PDFs and store vectors in Pinecone
# process_multiple_pdfs(pdf_files)

# # Start the chatbot
# chat_with_bot()
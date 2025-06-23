# import os
# import openai
# import pdfplumber
# from pdf2image import convert_from_path
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# import easyocr
# import pinecone  # For Pinecone vector database

# # Load environment variables
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")  # Pinecone API Key

# # Maximum length of chat history to maintain
# MAX_HISTORY_LENGTH = 10

# if not openai_api_key or not pinecone_api_key:
#     print("❌ OpenAI API Key or Pinecone API Key missing. Check your .env file.")
#     exit()

# # Initialize Pinecone
# #us-west1-gcp
# pinecone.init(api_key=pinecone_api_key, environment="us-east-1")  # Set Pinecone environment

# # Pinecone index name
# PINECONE_INDEX_NAME = "medical-chatbot-index"

# # Create or load Pinecone index
# if PINECONE_INDEX_NAME not in pinecone.list_indexes():
#     print("❌ No Pinecone index found, creating a new index...")
#     pinecone.create_index(PINECONE_INDEX_NAME, dimension=1536)  # OpenAI embedding dimension is 1536
# index = pinecone.Index(PINECONE_INDEX_NAME)

# # Extract text directly from PDF using pdfplumber
# def extract_text_from_pdf(pdf_path):
#     extracted_text = ""
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 text = page.extract_text() or ""
#                 extracted_text += text + " "
#         print(f"✅ Extracted {len(extracted_text)} characters directly from PDF: {pdf_path}")
#     except Exception as e:
#         print(f"❌ Error extracting text from PDF {pdf_path}: {e}")
#     return extracted_text

# # Convert PDF file to images
# def convert_pdf_to_images(pdf_path):
#     image_paths = []
#     try:
#         images = convert_from_path(pdf_path, poppler_path=r"C:\Users\STA\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin")
#         for i, image in enumerate(images):
#             image_path = f"page_{i+1}.png"
#             image.save(image_path, 'PNG')
#             image_paths.append(image_path)
#     except Exception as e:
#         print(f"❌ Error converting {pdf_path} to images: {e}")
#     return image_paths

# # Extract text from images
# def extract_text_from_image(image_path):
#     print(f"🔍 Running OCR on: {image_path}...")
#     reader = easyocr.Reader(['en'], gpu=False)  # OCR for English language
#     result = reader.readtext(image_path)
#     extracted_text = " ".join([detection[1] for detection in result])
#     print(f"✅ Extracted {len(extracted_text)} characters from OCR: {image_path}")
#     return extracted_text

# # Predefined health questions
# def get_default_questions_and_answers():
#     return [
#         {"question": "Do you exercise regularly?", "answer": ""},
#         {"question": "Do you sleep 7-8 hours a day?", "answer": ""},
#         {"question": "Do you frequently feel stressed?", "answer": ""},
#         {"question": "Do you have high blood pressure?", "answer": ""},
#         {"question": "Do you consume tobacco or alcohol?", "answer": ""}
#     ]

# # Process PDF files and store vectors in Pinecone
# def process_multiple_pdfs(pdf_files):
#     all_text = ""

#     for pdf_file in pdf_files:
#         print(f"📄 Processing PDF: {pdf_file}")
        
#         # Extract text directly from PDF
#         text = extract_text_from_pdf(pdf_file)
#         print(f"✅ Extracted {len(text)} characters from {pdf_file}.")
#         all_text += text + " "

#         # Convert PDF to images and run OCR
#         print("📸 Extracting images and running OCR...")
#         image_paths = convert_pdf_to_images(pdf_file)
#         for img_path in image_paths:
#             ocr_text = extract_text_from_image(img_path)
#             all_text += ocr_text + " "
#             try:
#                 os.remove(img_path)  # Delete image file
#             except Exception as e:
#                 print(f"⚠️ Error deleting {img_path}: {e}")

#     if not all_text.strip():
#         print("❌ No text found in PDF files.")
#         return None

#     # Chunk the text
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     texts = text_splitter.split_text(all_text)
#     documents = [Document(page_content=text) for text in texts]

#     # Create OpenAI embeddings
#     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     vectors = []

#     # Store vectors in Pinecone
#     for i, doc in enumerate(documents):
#         embedding = embeddings.embed_query(doc.page_content)
#         vectors.append((f"vec_{i}", embedding, {"text": doc.page_content}))  # Vector and metadata

#     # Upload vectors to Pinecone
#     index.upsert(vectors=vectors)
#     print("✅ Vectors stored in Pinecone.")
#     return index

# # Get answer from OpenAI
# def get_answer_from_llm(user_input, index, chat_history):
#     try:
#         # Add user question to chat history
#         chat_history.append({"role": "user", "content": user_input})

#         # Trim chat history
#         if len(chat_history) > MAX_HISTORY_LENGTH:
#             chat_history = chat_history[-MAX_HISTORY_LENGTH:]

#         # Find relevant text from Pinecone
#         query_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query(user_input)
#         query_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)  # Top 3 results
#         context = " ".join([match["metadata"]["text"] for match in query_results["matches"]])

#         # Send question to OpenAI GPT-4
#         client = openai.OpenAI(api_key=openai_api_key)
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a medical advisor. Provide scientific and reliable answers. Mention sources or expert names if available."},
#                 *chat_history,
#                 {"role": "user", "content": f"User asked: {user_input}\n\nScientific context: {context}"}
#             ],
#             max_tokens=500,
#             temperature=0.1
#         )

#         # Get AI response
#         ai_response = response.choices[0].message.content.strip()
#         chat_history.append({"role": "assistant", "content": ai_response})
#         return ai_response
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         return "Error generating response."

# # Generate health summary based on user responses
# def generate_health_summary(user_responses, index):
#     try:
#         combined_answers = " | ".join(user_responses)
#         prompt = f"Based on these health responses, provide a summary and recommendations: {combined_answers}"
        
#         # Find relevant text from Pinecone
#         query_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query(prompt)
#         query_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
#         context = " ".join([match["metadata"]["text"] for match in query_results["matches"]])
        
#         client = openai.OpenAI(api_key=openai_api_key)
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a medical advisor. Create a comprehensive health summary based on the user's responses. Include specific recommendations and potential areas of concern based on their health information."},
#                 {"role": "user", "content": f"User health information: {combined_answers}\n\nRelevant medical context: {context}"}
#             ],
#             max_tokens=800,
#             temperature=0.1
#         )
        
#         summary = response.choices[0].message.content.strip()
#         print("\n✅ Health Summary Generated:")
#         print("----------------------------")
#         print(summary)
#         return summary
        
#     except Exception as e:
#         print(f"❌ Error generating health summary: {e}")
#         return "Error generating health summary."

# # Run chatbot
# def chat_with_bot(pdf_files):
#     print("🚀 Starting AI chatbot...")
    
#     # Load or create Pinecone index
#     if index.describe_index_stats()["total_vector_count"] > 0:
#         print("✅ Pinecone index loaded.")
#     else:
#         print("❌ No Pinecone index found, creating new index...")
#         process_multiple_pdfs(pdf_files)

#     chat_history = []
#     questions = get_default_questions_and_answers()

#     # Collect user responses to predefined questions
#     user_responses = []
#     print("\n📋 Please answer these health questions first:")
#     for qa in questions:
#         print(f"\n❓ {qa['question']}")
#         answer = input("Your answer: ")
#         user_responses.append(f"{qa['question']} {answer}")
#         qa['answer'] = answer

#     # Generate initial health summary
#     print("\n🎯 Generating your health summary...")
#     health_summary = generate_health_summary(user_responses, index)
#     chat_history.append({"role": "assistant", "content": health_summary})

#     # Continue chat until user exits
#     print("\n💬 You can now chat with the medical assistant. Type 'exit', 'quit', or 'bye' to end.")
#     while True:
#         user_input = input("\n💬 You: ")
#         if user_input.lower() in ["exit", "quit", "bye"]:
#             print("👋 Chat session ended.")
#             break

#         ai_response = get_answer_from_llm(user_input, index, chat_history)
#         print(f"\n🤖 AI: {ai_response}")

# # PDF file paths
# pdf_files = [
#     r"C:\Users\STA\Desktop\dr\ABPM vs office in HTN_NEJM.pdf",
#     r"C:\Users\STA\Desktop\dr\Clinical Cardiology - October 1992 - Pickering - Ambulatory blood pressure monitoring An historical perspective.pdf",
#     r"C:\Users\STA\Desktop\dr\Cost savings of ABPM.pdf",
#     r"C:\Users\STA\Desktop\dr\jamacardiology_blood_2022_oi_220067_1672335582.056.pdf",
#     r"C:\Users\STA\Desktop\dr\Lee2022_clinical decisions remote BPM.pdf",
#     r"C:\Users\STA\Desktop\dr\NIGHTIME ABPM nightime dippers.risers.pdf",
#     r"C:\Users\STA\Desktop\dr\OptiBP app.pdf",
#     r"C:\Users\STA\Desktop\dr\photophytoelectric signal for BP.pdf"
# ]

# # Run chatbot
# chat_with_bot(pdf_files)















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
#     print("\u274C No Pinecone index found, creating a new one...")
#     pc.create_index(
#         name=PINECONE_INDEX_NAME,
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )
# else:
#     print("✅ Using existing Pinecone index.")

# # Load existing Pinecone index
# index = pc.Index(PINECONE_INDEX_NAME)

# # Folder containing PDFs
# PDF_FOLDER = "pdf_files/"
# pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))

# if not pdf_files:
#     print("\u274C No PDF files found in 'pdf_files/' folder. Please add PDFs and try again.")
#     exit()

# print(f"📄 Found {len(pdf_files)} PDF files in '{PDF_FOLDER}'.")

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
#         print(f"\u274C Error converting {pdf_path} to images: {e}")
#     return image_paths

# def extract_text_from_image(image_path):
#     """Extract text from an image using OCR."""
#     print(f"🔍 Running OCR on: {image_path}...")
#     reader = easyocr.Reader(['en'], gpu=False)
#     result = reader.readtext(image_path)
#     extracted_text = " ".join([detection[1] for detection in result])
#     print(f"✅ Extracted {len(extracted_text)} characters from OCR: {image_path}")
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
#         print(f"\u274C Error extracting text from {pdf_path}: {e}")
#     return text

# def process_multiple_pdfs(pdf_files):
#     """Process multiple PDFs, extract text, and store in Pinecone."""
#     all_text = ""
    
#     for pdf_file in pdf_files:
#         print(f"📄 Processing PDF: {pdf_file}")
#         text = extract_text_from_pdf(pdf_file)
#         if not text:
#             print(f"⚠️ No text extracted from {pdf_file}, trying OCR...")
#             image_paths = convert_pdf_to_images(pdf_file)
#             for image_path in image_paths:
#                 text += extract_text_from_image(image_path)
#         print(f"✅ Extracted {len(text)} characters from {pdf_file}.")
#         all_text += text + " "
    
#     if not all_text.strip():
#         print("❌ No text found in PDF files.")
#         return None
    
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     texts = text_splitter.split_text(all_text)
    
#     documents = [Document(page_content=text) for text in texts]
    
#     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     vectors = []
    
#     for i, doc in enumerate(documents):
#         embedding = embeddings.embed_query(doc.page_content)
#         vectors.append((f"vec_{i}", embedding, {"text": doc.page_content}))
    
#     # Print sample vector for debugging
#     print("Sample vector format:", vectors[0])
    
#     # Upsert vectors in smaller batches
#     batch_size = 100  # Adjust batch size as needed
#     for i in range(0, len(vectors), batch_size):
#         batch = vectors[i:i + batch_size]
#         try:
#             index.upsert(vectors=batch)
#             print(f"✅ Upserted batch {i // batch_size + 1} with {len(batch)} vectors.")
#         except Exception as e:
#             print(f"\u274C Error upserting batch {i // batch_size + 1}: {e}")
    
#     # Print Pinecone index stats
#     index_stats = index.describe_index_stats()
#     print("Pinecone index stats:", index_stats)

# def get_answer_from_llm(user_input, index, chat_history):
#     """Retrieve an answer from OpenAI based on Pinecone search results while maintaining chat history."""
#     try:
#         query_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query(user_input)
#         query_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        
#         # Print query results for debugging
#         print("Query results from Pinecone:", query_results)
        
#         context = " ".join([match["metadata"]["text"] for match in query_results["matches"]])
        
#         chat_history.append({"role": "user", "content": user_input})
        
#         if len(chat_history) > MAX_HISTORY_LENGTH:
#             chat_history = chat_history[-MAX_HISTORY_LENGTH:]
        
#         # Use OpenAI's new API
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a medical advisor. Provide scientific and credible answers. Mention the original source or expert name if available."},
#                 *chat_history,
#                 {"role": "user", "content": f"User asked: {user_input}\n\nScientific context: {context}"}
#             ],
#             max_tokens=500,
#             temperature=0.1
#         )
        
#         ai_response = response.choices[0].message.content.strip()
        
#         chat_history.append({"role": "assistant", "content": ai_response})
        
#         return ai_response
#     except Exception as e:
#         print(f"\u274C Error: {e}")
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
#     print("\u274C No Pinecone index found, creating a new one...")
#     pc.create_index(
#         name=PINECONE_INDEX_NAME,
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )
# else:
#     print("✅ Using existing Pinecone index.")

# # Load existing Pinecone index
# index = pc.Index(PINECONE_INDEX_NAME)

# # Folder containing PDFs
# PDF_FOLDER = "pdf_files/"
# pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))

# if not pdf_files:
#     print("\u274C No PDF files found in 'pdf_files/' folder. Please add PDFs and try again.")
#     exit()

# print(f"📄 Found {len(pdf_files)} PDF files in '{PDF_FOLDER}'.")

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
#         print(f"\u274C Error converting {pdf_path} to images: {e}")
#     return image_paths

# def extract_text_from_image(image_path):
#     """Extract text from an image using OCR."""
#     print(f"🔍 Running OCR on: {image_path}...")
#     reader = easyocr.Reader(['en'], gpu=False)
#     result = reader.readtext(image_path)
#     extracted_text = " ".join([detection[1] for detection in result])
#     print(f"✅ Extracted {len(extracted_text)} characters from OCR: {image_path}")
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
#         print(f"\u274C Error extracting text from {pdf_path}: {e}")
#     return text

# def process_multiple_pdfs(pdf_files):
#     """Process multiple PDFs, extract text, and store in Pinecone."""
#     all_text = ""
    
#     for pdf_file in pdf_files:
#         print(f"📄 Processing PDF: {pdf_file}")
#         text = extract_text_from_pdf(pdf_file)
#         if not text:
#             print(f"⚠️ No text extracted from {pdf_file}, trying OCR...")
#             image_paths = convert_pdf_to_images(pdf_file)
#             for image_path in image_paths:
#                 text += extract_text_from_image(image_path)
#         print(f"✅ Extracted {len(text)} characters from {pdf_file}.")
#         all_text += text + " "
    
#     if not all_text.strip():
#         print("❌ No text found in PDF files.")
#         return None
    
#     # Split text into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     texts = text_splitter.split_text(all_text)
    
#     # Print text chunks for debugging
#     print(f"Total text chunks: {len(texts)}")
#     print("Sample text chunk:", texts[0])
    
#     documents = [Document(page_content=text) for text in texts]
    
#     # Generate embeddings for each text chunk
#     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     vectors = []
    
#     for i, doc in enumerate(documents):
#         embedding = embeddings.embed_query(doc.page_content)
#         vectors.append((f"vec_{i}", embedding, {"text": doc.page_content}))
    
#     # Print sample vector for debugging
#     print("Sample vector:", vectors[0])
    
#     # Upsert vectors in smaller batches
#     batch_size = 100  # Adjust batch size as needed
#     for i in range(0, len(vectors), batch_size):
#         batch = vectors[i:i + batch_size]
#         try:
#             index.upsert(vectors=batch)
#             print(f"✅ Upserted batch {i // batch_size + 1} with {len(batch)} vectors.")
#         except Exception as e:
#             print(f"\u274C Error upserting batch {i // batch_size + 1}: {e}")
    
#     # Print Pinecone index stats
#     index_stats = index.describe_index_stats()
#     print("Pinecone index stats:", index_stats)

# def get_answer_from_llm(user_input, index, chat_history):
#     """Retrieve an answer from OpenAI based on Pinecone search results while maintaining chat history."""
#     try:
#         query_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query(user_input)
#         query_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        
#         # Print query results for debugging
#         print("Query results from Pinecone:", query_results)
        
#         context = " ".join([match["metadata"]["text"] for match in query_results["matches"]])
        
#         chat_history.append({"role": "user", "content": user_input})
        
#         if len(chat_history) > MAX_HISTORY_LENGTH:
#             chat_history = chat_history[-MAX_HISTORY_LENGTH:]
        
#         # Use OpenAI's new API
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a medical advisor. Provide scientific and credible answers. Mention the original source or expert name if available."},
#                 *chat_history,
#                 {"role": "user", "content": f"User asked: {user_input}\n\nScientific context: {context}"}
#             ],
#             max_tokens=500,
#             temperature=0.1
#         )
        
#         ai_response = response.choices[0].message.content.strip()
        
#         chat_history.append({"role": "assistant", "content": ai_response})
        
#         return ai_response
#     except Exception as e:
#         print(f"\u274C Error: {e}")
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
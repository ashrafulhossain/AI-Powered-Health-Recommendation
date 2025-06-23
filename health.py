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
# import shutil
# import uuid

# def receive_pdfs_and_pmids(uploaded_files, pmids, pinecone_index_name="medical-chatbot-index"):
#     """Receive PDF paths and PMIDs, process them, and upsert to Pinecone."""
#     try:
#         # Environment setup
#         load_dotenv()
#         openai_api_key = os.getenv("OPENAI_API_KEY")
#         pinecone_api_key = os.getenv("PINECONE_API_KEY")
#         if not openai_api_key or not pinecone_api_key:
#             raise Exception("Missing API keys. Set OPENAI_API_KEY and PINECONE_API_KEY in .env file.")

#         # File and folder paths
#         temp_folder = "temp_uploads/"
#         PROCESSED_DATA_FILE = "processed_data.pkl"
#         PDF_PMID_MAPPING_FILE = "pdf_pmid_mapping.json"
#         OCR_CACHE_DIR = "cache/ocr/"
#         EMBEDDING_CACHE_DIR = "cache/embeddings/"

#         os.makedirs(temp_folder, exist_ok=True)
#         os.makedirs(OCR_CACHE_DIR, exist_ok=True)
#         os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

#         # Initialize Pinecone
#         pc = Pinecone(api_key=pinecone_api_key)
#         if pinecone_index_name not in pc.list_indexes().names():
#             print("🛠️ Creating Pinecone index...")
#             pc.create_index(
#                 name=pinecone_index_name,
#                 dimension=1536,
#                 metric="cosine",
#                 spec=ServerlessSpec(cloud="aws", region="us-east-1")
#             )
#         index = pc.Index(pinecone_index_name)

#         # Load PDF-PMID mapping
#         def load_pdf_pmid_mapping():
#             if os.path.exists(PDF_PMID_MAPPING_FILE):
#                 with open(PDF_PMID_MAPPING_FILE, 'r', encoding='utf-8') as f:
#                     return json.load(f)
#             else:
#                 empty_mapping = {}
#                 with open(PDF_PMID_MAPPING_FILE, 'w', encoding='utf-8') as f:
#                     json.dump(empty_mapping, f, indent=2)
#                 return empty_mapping

#         # Extract text from PDF
#         def extract_text_from_pdf(pdf_path):
#             text = ""
#             try:
#                 with pdfplumber.open(pdf_path) as pdf:
#                     for page in pdf.pages:
#                         page_text = page.extract_text()
#                         if page_text:
#                             text += page_text + "\n"
#             except Exception as e:
#                 print(f"⚠️ Error extracting text from {pdf_path}: {e}")
#             return text

#         # Convert PDF to images
#         def convert_pdf_to_images(pdf_path):
#             image_paths = []
#             try:
#                 images = convert_from_path(pdf_path)
#                 for i, image in enumerate(images):
#                     image_path = os.path.join(OCR_CACHE_DIR, f"page_{i+1}_{uuid.uuid4()}.png")
#                     image.save(image_path, 'PNG')
#                     image_paths.append(image_path)
#             except Exception as e:
#                 print(f"⚠️ Error converting {pdf_path} to images: {e}")
#             return image_paths

#         # Extract text from image using OCR
#         def extract_text_from_image(image_path):
#             cache_file = os.path.join(OCR_CACHE_DIR, f"{os.path.basename(image_path)}.txt")
#             if os.path.exists(cache_file):
#                 with open(cache_file, 'r', encoding='utf-8') as f:
#                     print(f"✅ Using cached OCR text for {image_path}")
#                     return f.read()
#             try:
#                 reader = easyocr.Reader(['en'], gpu=True)
#                 result = reader.readtext(image_path)
#                 extracted_text = " ".join([detection[1] for detection in result])
#                 with open(cache_file, 'w', encoding='utf-8') as f:
#                     f.write(extracted_text)
#                 print(f"✅ OCR completed and cached for {image_path}")
#                 return extracted_text
#             except Exception as e:
#                 print(f"⚠️ Error extracting text from image {image_path}: {e}")
#                 return ""

#         # Upsert PDF to Pinecone
#         def upsert_pdf_to_pinecone(pdf_path, pmid, index):
#             pdf_name = os.path.normpath(pdf_path)
#             cache_file = os.path.join(EMBEDDING_CACHE_DIR, f"embeddings_{pmid}.pkl")
#             if os.path.exists(cache_file):
#                 with open(cache_file, "rb") as f:
#                     vectors = pickle.load(f)
#                 print(f"✅ Using cached embeddings for {pdf_name}")
#             else:
#                 text = extract_text_from_pdf(pdf_path)
#                 image_paths = convert_pdf_to_images(pdf_path)
#                 ocr_text = ""
#                 for image_path in image_paths:
#                     ocr_result = extract_text_from_image(image_path)
#                     ocr_text += ocr_result + " "
#                 combined_text = text + " " + ocr_text
#                 if combined_text.strip():
#                     text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
#                     texts = text_splitter.split_text(combined_text)
#                     documents = [Document(page_content=text) for text in texts]
#                     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#                     vectors = []
#                     for i, doc in enumerate(documents):
#                         embedding = embeddings.embed_query(doc.page_content)
#                         vectors.append((
#                             f"vec_{i}_{pmid}",
#                             embedding,
#                             {"text": doc.page_content, "pmid": pmid}
#                         ))
#                     with open(cache_file, "wb") as f:
#                         pickle.dump(vectors, f)
#                     print(f"✅ Embeddings created and cached for {pdf_name}")
            
#             batch_size = 50
#             for i in range(0, len(vectors), batch_size):
#                 batch = vectors[i:i + batch_size]
#                 try:
#                     index.upsert(vectors=batch)
#                 except Exception as e:
#                     print(f"⚠️ Error upserting batch to Pinecone: {e}")
#             print(f"✅ PDF {pdf_name} successfully added to Pinecone index.")

#         # Clean old cache files
#         def clean_cache_files(directory):
#             deleted_count = 0
#             for file in glob.glob(os.path.join(directory, "*.txt")) + glob.glob(os.path.join(directory, "*.pkl")):
#                 if os.path.getmtime(file) < time.time() - (30 * 24 * 3600):
#                     os.remove(file)
#                     print(f"🗑️ Removed old cache file: {file}")
#                     deleted_count += 1
#             print(f"✅ Total {deleted_count} cache files deleted from {directory}")

#         # Process PDFs
#         clean_cache_files(OCR_CACHE_DIR)
#         clean_cache_files(EMBEDDING_CACHE_DIR)
#         pdf_pmid_mapping = load_pdf_pmid_mapping()
#         new_pdfs = []
        
#         for file_path, pmid in zip(uploaded_files, pmids):
#             if not os.path.exists(file_path):
#                 print(f"⚠️ Skipping file: {file_path} (not found)")
#                 continue
#             temp_file_path = os.path.join(temp_folder, os.path.basename(file_path))
#             shutil.copy(file_path, temp_file_path)
#             new_pdfs.append((temp_file_path, pmid))
#             print(f"📄 Processing PDF: {temp_file_path} (PMID: {pmid})")
#             pdf_pmid_mapping[os.path.normpath(temp_file_path)] = pmid
#             with open(PDF_PMID_MAPPING_FILE, 'w', encoding='utf-8') as f:
#                 json.dump(pdf_pmid_mapping, f, indent=2)
#             upsert_pdf_to_pinecone(temp_file_path, pmid, index)

#         # Clean up temporary folder
#         if os.path.exists(temp_folder):
#             shutil.rmtree(temp_folder, ignore_errors=True)
#             print(f"🗑️ Temporary folder {temp_folder} removed.")
        
#         print(f"✅ Processed {len(new_pdfs)} PDFs and PMIDs")
#         return {"status": "success", "processed_pdfs": len(new_pdfs)}

#     except Exception as e:
#         print(f"❌ Error processing PDFs: {e}")
#         return {"error": f"Error processing PDFs: {e}"}

# def process_json_input(json_input, pinecone_index_name="medical-chatbot-index"):
#     """Process JSON input, search Pinecone, and generate health recommendations."""
#     try:
#         # Environment setup
#         load_dotenv()
#         openai_api_key = os.getenv("OPENAI_API_KEY")
#         pinecone_api_key = os.getenv("PINECONE_API_KEY")
#         if not openai_api_key or not pinecone_api_key:
#             raise Exception("Missing API keys. Set OPENAI_API_KEY and PINECONE_API_KEY in .env file.")

#         # Initialize OpenAI and Pinecone clients
#         client = openai.OpenAI(api_key=openai_api_key)
#         pc = Pinecone(api_key=pinecone_api_key)
#         index = pc.Index(pinecone_index_name)

#         # Process JSON input
#         if not json_input:
#             return {"error": "No JSON input provided."}
#         if isinstance(json_input, str) and os.path.exists(json_input):
#             with open(json_input, 'r', encoding='utf-8') as f:
#                 json_data = json.load(f)
#         else:
#             json_data = json.loads(json_input.strip())
        
#         user_name = json_data.get("user_name", "User")
#         form_data = json_data.get("response_form_data", [])

#         # Generate recommendations
#         context = ""
#         pmids = set()
#         for item in form_data:
#             user_input = f"{item.get('question', '')}: {item.get('response_text', '')}"
#             query_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query(user_input)
#             query_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
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
#             print("⚠️ No relevant information found in Pinecone. Using GPT-4 to generate recommendations.")
        
#         messages = [
#             {
#                 "role": "system",
#                 "content": f"""
# You are a professional medical AI assistant. Based on the user's health data and the following medical literature references, provide clear, concise preventive health recommendations. The user data includes responses to the following questions: Name, Age, Gender, Zip Code, Smoking Status, Exercise Frequency, Sleep Duration, Daily Fruit/Vegetable Intake, Weekly Physical Activity, Last Doctor Visit, Blood Pressure, Do you wear seatbelts?, Do you feel safe at home?, Last Dental Checkup, BMI, Cholesterol Level.

# - Format your response as a numbered list.
# - After each recommendation, include a source link in this exact format:  
# Source: https://pubmed.ncbi.nlm.nih.gov/{{pmid}}/
# - Use these PubMed IDs as references for your response: {', '.join(pmids)}.
# - If no PubMed ID is relevant for a recommendation, do NOT include any source line.
# - Use simple, actionable language tailored to the user's specific health data (e.g., address high blood pressure if reported, recommend quitting smoking if the user smokes).
# - Focus on preventive health measures based on the provided questions and user responses.
# """
#             },
#             {
#                 "role": "user",
#                 "content": f"User data: {json.dumps(form_data, ensure_ascii=False)}\nContext: {context}"
#             }
#         ]

#         response = client.chat.completions.create(
#             model="gpt-4-turbo",
#             messages=messages,
#             max_tokens=800,
#             temperature=0.2
#         )

#         ai_response = response.choices[0].message.content.strip()
#         sources_text = f"Sources: {', '.join(pmids)}" if pmids else "Sources: No source information found."

#         return {
#             "user_name": user_name,
#             "message": ai_response,
#             "sources": sources_text
#         }

#     except json.JSONDecodeError as e:
#         return {"error": f"Invalid JSON format: {e}"}
#     except Exception as e:
#         return {"error": f"Error processing JSON data: {e}"}

# # Example usage
# if __name__ == "__main__":
#     # Example PDFs and PMIDs
#     # uploaded_files = [
#     #     r"C:\Users\STA\Desktop\dr\uploads\ABPM vs office in HTN_NEJM.pdf",
#     #     r"C:\Users\STA\Desktop\dr\uploads\OptiBP app.pdf"
#     # ]
#     # pmids = ["PMID-12802026", "PMID-35727426"]

#     # Process PDFs
#     # pdf_result = receive_pdfs_and_pmids(uploaded_files, pmids)
#     # print(pdf_result)

#     # Example JSON input
#     json_input = '''{
#       "user_name": "Rahim",
#       "response_form_data": [
#         {"question": "Name", "response_text": "Rahim Uddin"},
#         {"question": "Age", "response_text": "45"},
#         {"question": "Gender", "response_text": "Male"},
#         {"question": "Zip Code", "response_text": "67890"},
#         {"question": "Smoking Status", "response_text": "Yes, occasionally"},
#         {"question": "Exercise Frequency", "response_text": "1-2 times a week"},
#         {"question": "Sleep Duration", "response_text": "6 hours"},
#         {"question": "Daily Fruit/Vegetable Intake", "response_text": "2 servings"},
#         {"question": "Weekly Physical Activity", "response_text": "2 hours"},
#         {"question": "Last Doctor Visit", "response_text": "6 months ago"},
#         {"question": "Blood Pressure", "response_text": "135/85"},
#         {"question": "Do you wear seatbelts?", "response_text": "Yes"},
#         {"question": "Do you feel safe at home?", "response_text": "Yes"},
#         {"question": "Last Dental Checkup", "response_text": "1 year ago"},
#         {"question": "BMI", "response_text": "26"},
#         {"question": "Cholesterol Level", "response_text": "210 mg/dL"}
#       ]
#     }'''

#     # Process JSON input
#     json_result = process_json_input(json_input)
#     print(json_result.get('message', json_result.get('error')))











import os
import openai
import pdfplumber
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import easyocr
import glob
from pinecone import Pinecone, ServerlessSpec
import pickle
import json
import time
import shutil
import uuid

def receive_pdfs_and_pmids(uploaded_files, pmids, pinecone_index_name=" "):
    """Receive PDF paths and PMIDs, process them, and upsert to Pinecone."""
    try:
        # Environment setup
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not openai_api_key or not pinecone_api_key:
            raise Exception("Missing API keys. Set OPENAI_API_KEY and PINECONE_API_KEY in .env file.")

        # File and folder paths
        temp_folder = "temp_uploads/"
        PROCESSED_DATA_FILE = "processed_data.pkl"
        PDF_PMID_MAPPING_FILE = "pdf_pmid_mapping.json"
        OCR_CACHE_DIR = "cache/ocr/"
        EMBEDDING_CACHE_DIR = "cache/embeddings/"

        os.makedirs(temp_folder, exist_ok=True)
        os.makedirs(OCR_CACHE_DIR, exist_ok=True)
        os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        if pinecone_index_name not in pc.list_indexes().names():
            print("🛠️ Creating Pinecone index...")
            pc.create_index(
                name=pinecone_index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        index = pc.Index(pinecone_index_name)

        # Load PDF-PMID mapping
        def load_pdf_pmid_mapping():
            if os.path.exists(PDF_PMID_MAPPING_FILE):
                with open(PDF_PMID_MAPPING_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                empty_mapping = {}
                with open(PDF_PMID_MAPPING_FILE, 'w', encoding='utf-8') as f:
                    json.dump(empty_mapping, f, indent=2)
                return empty_mapping

        # Extract text from PDF
        def extract_text_from_pdf(pdf_path):
            text = ""
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                print(f"⚠️ Error extracting text from {pdf_path}: {e}")
            return text

        # Convert PDF to images
        def convert_pdf_to_images(pdf_path):
            image_paths = []
            try:
                images = convert_from_path(pdf_path)
                for i, image in enumerate(images):
                    image_path = os.path.join(OCR_CACHE_DIR, f"page_{i+1}_{uuid.uuid4()}.png")
                    image.save(image_path, 'PNG')
                    image_paths.append(image_path)
            except Exception as e:
                print(f"⚠️ Error converting {pdf_path} to images: {e}")
            return image_paths

        # Extract text from image using OCR
        def extract_text_from_image(image_path):
            cache_file = os.path.join(OCR_CACHE_DIR, f"{os.path.basename(image_path)}.txt")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    print(f"✅ Using cached OCR text for {image_path}")
                    return f.read()
            try:
                reader = easyocr.Reader(['en'], gpu=True)
                result = reader.readtext(image_path)
                extracted_text = " ".join([detection[1] for detection in result])
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                print(f"✅ OCR completed and cached for {image_path}")
                return extracted_text
            except Exception as e:
                print(f"⚠️ Error extracting text from image {image_path}: {e}")
                return ""

        # Upsert PDF to Pinecone
        def upsert_pdf_to_pinecone(pdf_path, pmid, index):
            pdf_name = os.path.normpath(pdf_path)
            cache_file = os.path.join(EMBEDDING_CACHE_DIR, f"embeddings_{pmid}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    vectors = pickle.load(f)
                print(f"✅ Using cached embeddings for {pdf_name}")
            else:
                text = extract_text_from_pdf(pdf_path)
                image_paths = convert_pdf_to_images(pdf_path)
                ocr_text = ""
                for image_path in image_paths:
                    ocr_result = extract_text_from_image(image_path)
                    ocr_text += ocr_result + " "
                combined_text = text + " " + ocr_text
                if combined_text.strip():
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
                    texts = text_splitter.split_text(combined_text)
                    documents = [Document(page_content=text) for text in texts]
                    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                    vectors = []
                    for i, doc in enumerate(documents):
                        embedding = embeddings.embed_query(doc.page_content)
                        vectors.append((
                            f"vec_{i}_{pmid}",
                            embedding,
                            {"text": doc.page_content, "pmid": pmid}
                        ))
                    with open(cache_file, "wb") as f:
                        pickle.dump(vectors, f)
                    print(f"✅ Embeddings created and cached for {pdf_name}")
            
            batch_size = 50
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                try:
                    index.upsert(vectors=batch)
                except Exception as e:
                    print(f"⚠️ Error upserting batch to Pinecone: {e}")
            print(f"✅ PDF {pdf_name} successfully added to Pinecone index.")

        # Clean old cache files
        def clean_cache_files(directory):
            deleted_count = 0
            for file in glob.glob(os.path.join(directory, "*.txt")) + glob.glob(os.path.join(directory, "*.pkl")):
                if os.path.getmtime(file) < time.time() - (30 * 24 * 3600):
                    os.remove(file)
                    print(f"🗑️ Removed old cache file: {file}")
                    deleted_count += 1
            print(f"✅ Total {deleted_count} cache files deleted from {directory}")

        # Process PDFs
        clean_cache_files(OCR_CACHE_DIR)
        clean_cache_files(EMBEDDING_CACHE_DIR)
        pdf_pmid_mapping = load_pdf_pmid_mapping()
        new_pdfs = []
        
        for file_path, pmid in zip(uploaded_files, pmids):
            if not os.path.exists(file_path):
                print(f"⚠️ Skipping file: {file_path} (not found)")
                continue
            temp_file_path = os.path.join(temp_folder, os.path.basename(file_path))
            shutil.copy(file_path, temp_file_path)
            new_pdfs.append((temp_file_path, pmid))
            print(f"📄 Processing PDF: {temp_file_path} (PMID: {pmid})")
            pdf_pmid_mapping[os.path.normpath(temp_file_path)] = pmid
            with open(PDF_PMID_MAPPING_FILE, 'w', encoding='utf-8') as f:
                json.dump(pdf_pmid_mapping, f, indent=2)
            upsert_pdf_to_pinecone(temp_file_path, pmid, index)

        # Clean up temporary folder
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder, ignore_errors=True)
            print(f"🗑️ Temporary folder {temp_folder} removed.")
        
        print(f"✅ Processed {len(new_pdfs)} PDFs and PMIDs")
        return {"status": "success", "processed_pdfs": len(new_pdfs)}

    except Exception as e:
        print(f"❌ Error processing PDFs: {e}")
        return {"error": f"Error processing PDFs: {e}"}



def process_json_input(json_input, pinecone_index_name="medical-chatbot-index"):
    """Process JSON input, search Pinecone, and generate health recommendations in plain text with HTML-like format."""
    try:
        # Environment setup
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not openai_api_key or not pinecone_api_key:
            raise Exception("Missing API keys. Set OPENAI_API_KEY and PINECONE_API_KEY in .env file.")

        # Initialize OpenAI and Pinecone clients
        client = openai.OpenAI(api_key=openai_api_key)
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)

        # Process JSON input
        if not json_input:
            return {"error": "No JSON input provided."}
        if isinstance(json_input, str) and os.path.exists(json_input):
            with open(json_input, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        else:
            json_data = json.loads(json_input.strip())
        
        user_name = json_data.get("user_name", "User")
        form_data = json_data.get("response_form_data", [])

        # Generate recommendations
        context = ""
        pmids = set()
        for item in form_data:
            user_input = f"{item.get('question', '')}: {item.get('response_text', '')}"
            query_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query(user_input)
            query_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
            for match in query_results["matches"]:
                metadata = match.get("metadata", {})
                text = metadata.get("text", "")
                pmid = metadata.get("pmid", "PMID-UNKNOWN")
                context += text + " "
                if pmid != "PMID-UNKNOWN":
                    pmids.add(pmid)

        context = context.strip()
        pmids = list(pmids)

        if not context:
            print("⚠️ No relevant information found in Pinecone. Using GPT-4 to generate recommendations.")
        
        messages = [
            {
                "role": "system",
                "content": f"""
You are a professional medical AI assistant. Based on the user's health data and the following medical literature references, provide clear, concise preventive health recommendations. The user data includes responses to the following questions: Name, Age, Gender, Zip Code, Smoking Status, Exercise Frequency, Sleep Duration, Daily Fruit/Vegetable Intake, Weekly Physical Activity, Last Doctor Visit, Blood Pressure, Do you wear seatbelts?, Do you feel safe at home?, Last Dental Checkup, BMI, Cholesterol Level.

- Start the response with a greeting: "Hello [user_name],\nThank you for using our service.\nBased on your responses, here are preventive health recommendations just for you.\n"
- Format your response as a plain numbered list without any markdown symbols (e.g., no ** or bold formatting).
- After each recommendation, include a source link in this exact format:  
Source: https://pubmed.ncbi.nlm.nih.gov/{{pmid}}/
- Use these PubMed IDs as references for your response: {', '.join(pmids)}.
- If no PubMed ID is relevant for a recommendation, do NOT include any source line.
- Use simple, actionable language tailored to the user's specific health data (e.g., address high blood pressure if reported, recommend quitting smoking if the user smokes).
- Focus on preventive health measures based on the provided questions and user responses.
- At the end, add a closing statement: "By implementing these recommendations, you can significantly improve your health and reduce the risk of chronic diseases."
- Do NOT include a separate "Sources" line listing all PMIDs at the end of the response.
"""
            },
            {
                "role": "user",
                "content": f"User data: {json.dumps(form_data, ensure_ascii=False)}\nContext: {context}"
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=800,
            temperature=0.2
        )

        ai_response = response.choices[0].message.content.strip()

        return {
            "user_name": user_name,
            "message": ai_response
        }

    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON format: {e}"}
    except Exception as e:
        return {"error": f"Error processing JSON data: {e}"}

# Example usage
if __name__ == "__main__":
    # Example PDFs and PMIDs
    # uploaded_files = [
    #     r"C:\Users\STA\Desktop\dr\uploads\ABPM vs office in HTN_NEJM.pdf",
    #     r"C:\Users\STA\Desktop\dr\uploads\OptiBP app.pdf"
    # ]
    # pmids = ["PMID-12802026", "PMID-35727426"]

    # Process PDFs
    # pdf_result = receive_pdfs_and_pmids(uploaded_files, pmids)
    # print(pdf_result)

    # Example JSON input
    json_input = '''{
      "user_name": "Rahim",
      "response_form_data": [
        {"question": "Name", "response_text": "Rahim Uddin"},
        {"question": "Age", "response_text": "45"},
        {"question": "Gender", "response_text": "Male"},
        {"question": "Zip Code", "response_text": "67890"},
        {"question": "Smoking Status", "response_text": "Yes, occasionally"},
        {"question": "Exercise Frequency", "response_text": "1-2 times a week"},
        {"question": "Sleep Duration", "response_text": "6 hours"},
        {"question": "Daily Fruit/Vegetable Intake", "response_text": "2 servings"},
        {"question": "Weekly Physical Activity", "response_text": "2 hours"},
        {"question": "Last Doctor Visit", "response_text": "6 months ago"},
        {"question": "Blood Pressure", "response_text": "135/85"},
        {"question": "Do you wear seatbelts?", "response_text": "Yes"},
        {"question": "Do you feel safe at home?", "response_text": "Yes"},
        {"question": "Last Dental Checkup", "response_text": "1 year ago"},
        {"question": "BMI", "response_text": "26"},
        {"question": "Cholesterol Level", "response_text": "210 mg/dL"}
      ]
    }'''

    # Process JSON input
    json_result = process_json_input(json_input)
    print(json_result.get('message', json_result.get('error')))
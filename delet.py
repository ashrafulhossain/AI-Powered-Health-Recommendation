# import os
# import json
# from pinecone import Pinecone
# from datetime import datetime
# from dotenv import load_dotenv

# def delete_pdf_by_pmid(pmid, pinecone_index_name="medical-chatbot-index"):
#     try:
#         print(f"📝 Starting deletion for PMID: {pmid} at {datetime.now()}")

#         # Load environment variables
#         load_dotenv()
#         pinecone_api_key = os.getenv("PINECONE_API_KEY")
#         if not pinecone_api_key:
#             raise Exception("❌ PINECONE_API_KEY missing in .env")

#         # Initialize Pinecone client
#         pc = Pinecone(api_key=pinecone_api_key)

#         # Check if index exists
#         if pinecone_index_name not in pc.list_indexes().names():
#             return {"error": f"Index '{pinecone_index_name}' not found"}

#         # Connect to index
#         index = pc.Index(pinecone_index_name)
#         print(f"✅ Connected to index: {pinecone_index_name}")

#         # Normalize PMID
#         target_pmid = pmid.replace("PMID-", "")

#         # Use metadata filter to find vectors with matching PMID
#         print("🔍 Searching for matching vectors...")
#         matches = index.query(
#             vector=[0.0] * 1536,  # Dummy vector (must match your embedding size)
#             top_k=100,
#             include_metadata=True,
#             filter={"pmid": target_pmid}
#         )

#         # Extract vector IDs
#         vectors_to_delete = [match["id"] for match in matches["matches"]]

#         if not vectors_to_delete:
#             return {"status": "no_match", "message": f"No vectors found for PMID {pmid}"}

#         # Delete vectors in batches (if needed)
#         print(f"🗑️ Deleting {len(vectors_to_delete)} vectors...")
#         for i in range(0, len(vectors_to_delete), 100):
#             batch = vectors_to_delete[i:i+100]
#             index.delete(ids=batch)
#         print(f"✅ Deleted {len(vectors_to_delete)} vectors")

#         # Update mapping file
#         PDF_PMID_MAPPING_FILE = "pdf_pmid_mapping.json"
#         if os.path.exists(PDF_PMID_MAPPING_FILE):
#             with open(PDF_PMID_MAPPING_FILE, 'r') as f:
#                 pdf_pmid_mapping = json.load(f)

#             updated_mapping = {k: v for k, v in pdf_pmid_mapping.items() if v != pmid}
#             with open(PDF_PMID_MAPPING_FILE, 'w') as f:
#                 json.dump(updated_mapping, f, indent=2)
#             print("🗂️ Mapping file updated.")
#         else:
#             print("⚠️ Mapping file not found, skipping update.")

#         return {
#             "status": "success",
#             "deleted": len(vectors_to_delete),
#             "message": f"Data for PMID {pmid} deleted successfully"
#         }

#     except Exception as e:
#         print(f"❌ Error: {e}")
#         return {"error": str(e)}

# if __name__ == "__main__":
#     result = delete_pdf_by_pmid("PMID-31388564")
#     print(json.dumps(result, indent=2))



# import os
# import json
# from pinecone import Pinecone
# from datetime import datetime
# from dotenv import load_dotenv

# # Configurable batch size
# BATCH_SIZE = 500

# def delete_pdf_by_pmid(pmid, pinecone_index_name="medical-chatbot-index"):
#     try:
#         print(f"📝 Starting deletion for PMID: {pmid} at {datetime.now()}")

#         # Load environment variables
#         load_dotenv()
#         pinecone_api_key = os.getenv("PINECONE_API_KEY")
#         if not pinecone_api_key:
#             raise Exception("❌ PINECONE_API_KEY missing in .env")

#         # Initialize Pinecone client
#         pc = Pinecone(api_key=pinecone_api_key)

#         # Check if index exists
#         if pinecone_index_name not in pc.list_indexes().names():
#             return {"error": f"Index '{pinecone_index_name}' not found"}

#         # Connect to index
#         index = pc.Index(pinecone_index_name)
#         print(f"✅ Connected to index: {pinecone_index_name}")

#         # Normalize PMID
#         target_pmid = pmid.replace("PMID-", "")

#         # Use metadata filter to find vectors with matching PMID
#         print("🔍 Searching for matching vectors...")
#         matches = index.query(
#             vector=[0.0] * 1536,  # Dummy vector (must match your embedding size)
#             top_k=100,
#             include_metadata=True,
#             filter={"pmid": target_pmid}
#         )

#         # Extract vector IDs
#         vectors_to_delete = [match["id"] for match in matches["matches"]]

#         if not vectors_to_delete:
#             return {"status": "no_match", "message": f"No vectors found for PMID {pmid}"}

#         # Delete vectors in batches
#         print(f"🗑️ Deleting {len(vectors_to_delete)} vectors...")
#         for i in range(0, len(vectors_to_delete), BATCH_SIZE):
#             batch = vectors_to_delete[i:i + BATCH_SIZE]
#             index.delete(ids=batch)
#         print(f"✅ Deleted {len(vectors_to_delete)} vectors")

#         # Update mapping file
#         PDF_PMID_MAPPING_FILE = "pdf_pmid_mapping.json"
#         if os.path.exists(PDF_PMID_MAPPING_FILE):
#             with open(PDF_PMID_MAPPING_FILE, 'r') as f:
#                 pdf_pmid_mapping = json.load(f)

#             updated_mapping = {k: v for k, v in pdf_pmid_mapping.items() if v != pmid}
#             with open(PDF_PMID_MAPPING_FILE, 'w') as f:
#                 json.dump(updated_mapping, f, indent=2)
#             print("🗂️ Mapping file updated.")
#         else:
#             print("⚠️ Mapping file not found, skipping update.")

#         return {
#             "status": "success",
#             "deleted": len(vectors_to_delete),
#             "message": f"Data for PMID {pmid} deleted successfully"
#         }

#     except Exception as e:
#         print(f"❌ Error: {e}")
#         return {"error": str(e)}

# if __name__ == "__main__":
#     result = delete_pdf_by_pmid("PMID-3424242")
#     print(json.dumps(result, indent=2))







# import os
# import json
# from pinecone import Pinecone
# from datetime import datetime
# from dotenv import load_dotenv

# # Configurable batch size
# BATCH_SIZE = 100  # Adjusted to match the top_k value
# TOP_K = 100  # Number of vectors to fetch per query (pagination)

# def delete_pdf_by_pmid(pmid, pinecone_index_name="medical-chatbot-index"):
#     try:
#         print(f"📝 Starting deletion for PMID: {pmid} at {datetime.now()}")

#         # Load environment variables
#         load_dotenv()
#         pinecone_api_key = os.getenv("PINECONE_API_KEY")
#         if not pinecone_api_key:
#             raise Exception("❌ PINECONE_API_KEY missing in .env")

#         # Initialize Pinecone client
#         pc = Pinecone(api_key=pinecone_api_key)

#         # Check if index exists
#         if pinecone_index_name not in pc.list_indexes().names():
#             return {"error": f"Index '{pinecone_index_name}' not found"}

#         # Connect to index
#         index = pc.Index(pinecone_index_name)
#         print(f"✅ Connected to index: {pinecone_index_name}")

#         # Normalize PMID
#         target_pmid = pmid.replace("PMID-", "")

#         # Initialize variables for pagination
#         total_vectors_to_delete = []
#         offset = 0  # For pagination

#         # Pagination loop to get all matching vectors
#         while True:
#             print(f"🔍 Searching for matching vectors... (Offset: {offset})")
#             matches = index.query(
#                 vector=[0.0] * 1536,  # Dummy vector (must match your embedding size)
#                 top_k=TOP_K,  # Number of vectors to fetch per query
#                 include_metadata=True,
#                 filter={"pmid": target_pmid},
#                 offset=offset  # Set offset to get the next set of vectors
#             )

#             # Extract vector IDs from the results
#             vectors_to_delete = [match["id"] for match in matches["matches"]]

#             # If no vectors are found, break the loop
#             if not vectors_to_delete:
#                 print("❌ No more vectors found for deletion.")
#                 break

#             # Add the current batch of vectors to the total list
#             total_vectors_to_delete.extend(vectors_to_delete)

#             # Update offset for next query
#             offset += TOP_K

#             # If the number of vectors returned is less than TOP_K, it means we've reached the end
#             if len(vectors_to_delete) < TOP_K:
#                 print("✅ Reached the end of available vectors.")
#                 break

#         # Check if any vectors were found for deletion
#         if not total_vectors_to_delete:
#             return {"status": "no_match", "message": f"No vectors found for PMID {pmid}"}

#         # Delete vectors in batches
#         print(f"🗑️ Deleting {len(total_vectors_to_delete)} vectors...")
#         for i in range(0, len(total_vectors_to_delete), BATCH_SIZE):
#             batch = total_vectors_to_delete[i:i + BATCH_SIZE]
#             index.delete(ids=batch)
#         print(f"✅ Deleted {len(total_vectors_to_delete)} vectors")

#         # Update mapping file
#         PDF_PMID_MAPPING_FILE = "pdf_pmid_mapping.json"
#         if os.path.exists(PDF_PMID_MAPPING_FILE):
#             with open(PDF_PMID_MAPPING_FILE, 'r') as f:
#                 pdf_pmid_mapping = json.load(f)

#             updated_mapping = {k: v for k, v in pdf_pmid_mapping.items() if v != pmid}
#             with open(PDF_PMID_MAPPING_FILE, 'w') as f:
#                 json.dump(updated_mapping, f, indent=2)
#             print("🗂️ Mapping file updated.")
#         else:
#             print("⚠️ Mapping file not found, skipping update.")

#         return {
#             "status": "success",
#             "deleted": len(total_vectors_to_delete),
#             "message": f"Data for PMID {pmid} deleted successfully"
#         }

#     except Exception as e:
#         print(f"❌ Error: {e}")
#         return {"error": str(e)}

# if __name__ == "__main__":
#     result = delete_pdf_by_pmid("PMID-16344364")
#     print(json.dumps(result, indent=2))










import os
import json
from pinecone import Pinecone
from datetime import datetime
from dotenv import load_dotenv

# Configurable batch size
BATCH_SIZE = 500

def delete_pdf_by_pmid(pmid, pinecone_index_name="medical-chatbot-index"):
    try:
        print(f"📝 Starting deletion for PMID: {pmid} at {datetime.now()}")

        # Load environment variables
        load_dotenv()
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise Exception("❌ PINECONE_API_KEY missing in .env")

        # Initialize Pinecone client
        pc = Pinecone(api_key=pinecone_api_key)

        # Check if index exists
        if pinecone_index_name not in pc.list_indexes().names():
            return {"error": f"Index '{pinecone_index_name}' not found"}

        # Connect to index
        index = pc.Index(pinecone_index_name)
        print(f"✅ Connected to index: {pinecone_index_name}")

        # Normalize PMID
        target_pmid = pmid.replace("PMID-", "")

        # Use metadata filter to find vectors with matching PMID
        print("🔍 Searching for matching vectors...")
        matches = index.query(
            vector=[0.0] * 1536,  # Dummy vector (must match your embedding size)
            top_k=100,
            include_metadata=True,
            filter={"pmid": target_pmid}
        )

        # Extract vector IDs
        vectors_to_delete = [match["id"] for match in matches["matches"]]

        if not vectors_to_delete:
            return {"status": "no_match", "message": f"No vectors found for PMID {pmid}"}

        # Delete vectors in batches
        print(f"🗑️ Deleting {len(vectors_to_delete)} vectors...")
        for i in range(0, len(vectors_to_delete), BATCH_SIZE):
            batch = vectors_to_delete[i:i + BATCH_SIZE]
            index.delete(ids=batch)
        print(f"✅ Deleted {len(vectors_to_delete)} vectors")

        # Update mapping file
        PDF_PMID_MAPPING_FILE = "pdf_pmid_mapping.json"
        if os.path.exists(PDF_PMID_MAPPING_FILE):
            with open(PDF_PMID_MAPPING_FILE, 'r') as f:
                pdf_pmid_mapping = json.load(f)

            updated_mapping = {k: v for k, v in pdf_pmid_mapping.items() if v != pmid}
            with open(PDF_PMID_MAPPING_FILE, 'w') as f:
                json.dump(updated_mapping, f, indent=2)
            print("🗂️ Mapping file updated.")
        else:
            print("⚠️ Mapping file not found, skipping update.")

        return {
            "status": "success",
            "deleted": len(vectors_to_delete),
            "message": f"Data for PMID {pmid} deleted successfully"
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    result = delete_pdf_by_pmid("PMID-16344364")
    print(json.dumps(result, indent=2))
    
    
    
    
    
    
    
    
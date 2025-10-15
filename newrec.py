import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "    AIzaSyD014epBOONyNMqEaJKo7p9Ytyw-8H6QpY"  # replace with your key

# ----------------------------------------------------------
# STEP 1: Convert product data to text format for embedding
# ----------------------------------------------------------
def process_data(refined_df):
    # Combine product info into a single string
    text_list = []
    for i, row in refined_df.iterrows():
        product_text = f"Product: {row['product_name']}, Category: {row['category']}, Description: {row['description']}"
        text_list.append(product_text)

    # ✅ Use Hugging Face model for embeddings (offline + free)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # ✅ Create FAISS vector store
    vectorstore = FAISS.from_texts(text_list, embeddings)
    return vectorstore

# ----------------------------------------------------------
# STEP 2: Search for similar products using FAISS
# ----------------------------------------------------------
def search_similar_products(vectorstore, query, top_k=5):
    similar_products = vectorstore.similarity_search(query, k=top_k)
    return similar_products

# ----------------------------------------------------------
# STEP 3: Ask Gemini for reasoning/explanation
# ----------------------------------------------------------
def get_gemini_recommendation(query, products):
    chat_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

    product_descriptions = "\n".join([f"- {p.page_content}" for p in products])

    prompt = f"""
    A customer is looking for: "{query}".
    Based on the following available products, recommend the best options and explain why:

    {product_descriptions}

    Give your answer in a friendly e-commerce style, highlighting 2-3 key recommendations.
    """

    response = chat_model.invoke([HumanMessage(content=prompt)])
    return response.content

# ----------------------------------------------------------
# STEP 4: Main function to display recommendations
# ----------------------------------------------------------
def display_product_recommendation(refined_df):
    print("\n🛍️ Welcome to the AI Product Recommender!\n")
    query = input("Enter what you're looking for (e.g., 'running shoes under ₹2000'): ")

    # Create FAISS store
    vectorstore = process_data(refined_df)

    # Search similar products
    similar_products = search_similar_products(vectorstore, query)

    # Ask Gemini for friendly recommendations
    recommendation_text = get_gemini_recommendation(query, similar_products)

    print("\n✨ Recommended Products:\n")
    print(recommendation_text)
    print("\n-----------------------------------\n")

# ----------------------------------------------------------
# Optional: test with dummy data
# ----------------------------------------------------------
if __name__ == "__main__":
    data = {
        "product_name": ["Nike Air Zoom", "Adidas Ultraboost", "Puma Flyer Runner", "ASICS Gel Contend", "Sparx Sneakers"],
        "category": ["Shoes", "Shoes", "Shoes", "Shoes", "Shoes"],
        "description": [
            "Lightweight running shoe with responsive cushioning.",
            "Comfortable and stylish sneakers for long runs.",
            "Affordable running shoe with breathable mesh upper.",
            "Durable training shoe with gel cushioning system.",
            "Casual sneakers perfect for everyday wear."
        ]
    }
    df = pd.DataFrame(data)
    display_product_recommendation(df)

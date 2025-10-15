import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chains import RetrievalQA, LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import DataFrameLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()


def process_data(refined_df):
    """
    Process the refined dataset and create the vector store.
    """
    st.info("🔄 Processing product data and building embeddings... This may take a moment.")

    refined_df['combined_info'] = refined_df.apply(
        lambda row: (
            f"Product ID: {row['pid']}. Product URL: {row['product_url']}. "
            f"Product Name: {row['product_name']}. Primary Category: {row['primary_category']}. "
            f"Retail Price: ${row['retail_price']}. Discounted Price: ${row['discounted_price']}. "
            f"Primary Image Link: {row['primary_image_link']}. Description: {row['description']}. "
            f"Brand: {row['brand']}. Gender: {row['gender']}"
        ),
        axis=1
    )

    loader = DataFrameLoader(refined_df, page_content_column="combined_info")
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    # Keep only valid text chunks
    texts = [t for t in texts if t.page_content and len(t.page_content.strip()) > 10]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    text_list = [t.page_content for t in texts]

    with st.spinner("Creating vector database using Gemini embeddings..."):
        vectorstore = FAISS.from_texts(text_list, embeddings)

    st.success("✅ Vector database successfully created!")
    return vectorstore


def save_vectorstore(vectorstore, directory):
    """Save the FAISS vector store locally."""
    vectorstore.save_local(directory)


def load_vectorstore(directory, embeddings):
    """Load the FAISS vector store from disk."""
    return FAISS.load_local(directory, embeddings, allow_dangerous_deserialization=True)


def display_product_recommendation(refined_df):
    """
    Display the product recommendation section of the Streamlit app.
    """
    st.header("🛍️ E-commerce Product Recommendation")

    vectorstore_dir = 'vectorstore'

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Load or create vectorstore
    if os.path.exists(vectorstore_dir):
        st.info("📦 Loading existing product vector database...")
        vectorstore = load_vectorstore(vectorstore_dir, embeddings)
    else:
        vectorstore = process_data(refined_df)
        save_vectorstore(vectorstore, vectorstore_dir)

    # Manual Recommendation Template
    manual_template = """
    Kindly suggest three similar products based on the description I have provided below:

    Product Department: {department},
    Product Category: {category},
    Product Brand: {brand},
    Maximum Price range: {price}.

    Please provide product name, category, price, and availability if known.
    """
    prompt_manual = PromptTemplate(
        input_variables=["department", "category", "brand", "price"],
        template=manual_template,
    )

    # Gemini Chat model
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    chain = LLMChain(llm=llm, prompt=prompt_manual, verbose=True)

    # Conversational QA setup
    chatbot_template = """
    You are a helpful and friendly e-commerce shopping assistant.
    Use the provided context to help customers find products that match their preferences.

    {context}

    Chat history: {history}

    User Input: {question}
    Response:
    """
    chatbot_prompt = PromptTemplate(
        input_variables=["context", "history", "question"],
        template=chatbot_template,
    )

    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="question",
        return_messages=True
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectorstore.as_retriever(),
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": chatbot_prompt,
            "memory": memory
        }
    )

    # --- Streamlit UI ---
    department = st.text_input("🏬 Product Department")
    category = st.text_input("🧢 Product Category")
    brand = st.text_input("🏷️ Product Brand")
    price = st.text_input("💰 Maximum Price Range")

    if st.button("✨ Get Recommendations"):
        if not department and not category and not brand:
            st.warning("Please enter at least one product detail.")
        else:
            with st.spinner("🔍 Fetching product recommendations..."):
                try:
                    response = chain.run(
                        department=department,
                        category=category,
                        brand=brand,
                        price=price
                    )
                    st.success("✅ Recommendations ready!")
                    st.write(response)
                except Exception as e:
                    st.error(f"❌ Error generating recommendations: {e}")

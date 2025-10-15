E-commerce Product Recommendation System with GenAI

This project is an end-to-end e-commerce product recommendation system that leverages natural language processing (NLP), RAG techniques, and machine learning to provide personalized product recommendations to users.

Built using Python, it integrates libraries such as Streamlit, Pandas, Seaborn, Matplotlib, and LangChain. Unlike traditional recommendation systems, which rely on collaborative filtering or content-based filtering, this system uses RAG techniques to generate recommendations based on user input and preferences, understanding the semantic meaning behind queries to provide highly relevant suggestions.

The frontend is built with Streamlit, allowing users to interact with the system via an intuitive web interface. Users can input preferences such as product department, category, brand, and maximum price range, and receive personalized recommendations that go beyond predefined dataset categories.

Features

Data Analysis: Visualize price distribution across top categories and discount percentage distribution.

Product Recommendation: Generate recommendations based on user-defined preferences, not limited to the dataset’s categories or brands.

NLP-powered Search: Understand user queries using NLP techniques to provide relevant product suggestions.

Efficient Data Processing: Preprocess and tokenize the dataset to create a vector store for fast and efficient recommendation retrieval.

Persistent Storage: Save processed data and the vector store to disk, reducing repeated computation and API calls.

Interactive UI: User-friendly Streamlit interface for seamless interaction.

Installation

Clone the repository:

git clone https://github.com/your-username/ecommerce-product-recommendation.git


Navigate to the project directory:

cd ecommerce-product-recommendation


Install dependencies:

pip install -r requirements.txt


Set up your API key:
Create a .env file in the project root directory and add your Gemini API key:

GOOGLE_API_KEY=your-gemini-api-key


The system uses HuggingFace embeddings for vector storage, which avoids API quota issues.

Usage

Prepare your e-commerce dataset in CSV format and place it in the project directory.

Update the dataset_path variable in app.py with the path to your dataset file.

Run the Streamlit app:

streamlit run app.py


Open the URL provided in your browser.

Explore data analysis visualizations and interact with the recommendation system by providing your preferences (department, category, brand, max price).

Dataset

The project uses a sample Flipkart e-commerce dataset containing product information such as name, description, category, price, and brand. You can replace it with your own dataset in CSV format.

Vector Database

The project generates a FAISS vector store for efficient product recommendation retrieval.

Location: vectorstore/index.faiss

This file is generated automatically the first time you request product recommendations.

It is persistent, so repeated runs do not require recomputation.

Project Structure
ecommerce-product-recommendation/
├── app.py                   # Streamlit application
├── data_processing.py       # Data preprocessing, cleaning, and analysis
├── recommendation_utils.py  # Product recommendation logic using HuggingFace + Gemini
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables for API keys
└── README.md                # Project documentation

Dependencies

Streamlit: Web interface

Pandas: Data manipulation

Seaborn & Matplotlib: Data visualization

LangChain: LLM & RAG integration

HuggingFace (sentence-transformers): Local embeddings

LangChain-Google-GenAI: Gemini for chat-style explanations

For the complete list, see requirements.txt.

Future Enhancements

Add user authentication and personalized profiles

Include user reviews and ratings in recommendations

Integrate with a real-time e-commerce platform

Optimize for large-scale datasets and performance

Explore advanced NLP and deep learning models for improved accuracy

Contributing

Contributions are welcome! To contribute:

Open an issue for suggestions or bug reports

Submit a pull request with improvements

Follow the project’s code of conduct
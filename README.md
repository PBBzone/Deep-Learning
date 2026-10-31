🚀 5-Day Deep Learning & AI Training Bootcamp

This repository contains the models, projects, and applications developed during a 5-day intensive Deep Learning and Artificial Intelligence training module. The focus of this repository is on practical implementation, optimizing models for standard hardware, and building interactive user interfaces.

📅 Course Breakdown & Models Developed

Day 1: Fundamentals of Machine Learning (Update with your specific topics)

Introduction to data preprocessing and exploratory data analysis.

Basic supervised learning models.

Feature engineering and dataset preparation.

Day 2: Introduction to Neural Networks (Update with your specific topics)

Building basic Artificial Neural Networks (ANNs).

Understanding activation functions, forward propagation, and backpropagation.

Model evaluation metrics.

Day 3: Advanced Applications (RAG & CNN)

This day focused on building two major interactive applications:

1. College Knowledge Assistant (RAG System)
An AI assistant designed to answer student queries strictly based on official university documents to prevent hallucinations.

Features: Automatically ingests local PDF and Text files, uses semantic vector search to find relevant policies, and synthesizes answers while explicitly tracking and displaying the source file names for every answer.

Technologies: LangChain, FAISS Vector Database, Hugging Face Embeddings, Ollama (Gemma model), and Streamlit.

2. Optimized CNN Image Classifier
A web application to train and test Convolutional Neural Networks on image data, specifically optimized to run smoothly on low-end PCs without exhausting RAM or CPU.

Features: Includes built-in CIFAR-10 training with data-subsetting capabilities for speed, a custom ZIP dataset uploader that automatically cleans up storage, live accuracy/loss visualizations, and a prediction mode for classifying user-uploaded images.

Technologies: TensorFlow/Keras, Matplotlib, NumPy, and Streamlit.

Day 4: Natural Language Processing / Advanced Vision (Update with your specific topics)

Sequence models and text classification.

Advanced image processing techniques.

Day 5: Deployment & Capstone (Update with your specific topics)

Final project integration.

Model export and web deployment strategies.

🛠️ General Setup Instructions

To use the applications in this repository, ensure you have Python installed on your system along with the necessary virtual environment tools.

You will need to install the dependencies listed in the requirements file provided in the repository. For the RAG application, ensure you have the Ollama application installed and running locally with the required language model downloaded.

To run any of the web applications, navigate to the specific project folder in your terminal and use the Streamlit run command followed by the application's filename. For the RAG project, ensure your official text and PDF documents are placed inside the designated documents folder before launching the interface.

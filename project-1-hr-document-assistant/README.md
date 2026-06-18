# HR Document Intelligence Assistant
A RAG-powered HR document chatbot built on AWS Bedrock.
## What It Does
Answers employee HR questions by intelligently searching a library of
HR policies, offer letter templates, and compliance documents using
Retrieval-Augmented Generation (RAG).
## Architecture
- **AWS S3** — Secure document storage
- **AWS Bedrock Knowledge Bases** — RAG indexing and vector search
- **Amazon Nova** — Answer generation via Bedrock
- **Streamlit** — Chat UI
- **Python / boto3** — AWS integration layer
## Business Context
Built to demonstrate hands-on AI deployment skills, directly extending
enterprise EIM architecture experience (McKinsey) into working
generative AI applications.
## How to Run
```bash
pip install streamlit boto3
streamlit run app.py
```
## Skills Demonstrated
RAG Pipeline Design | AWS Bedrock | Feature Engineering |
AI-Ready Data Architecture | Responsible AI | Python
import streamlit as st
import boto3
import json

# ── Configuration ─────────────────────────────────────────────
KNOWLEDGE_BASE_ID = 'MEIINLQVIY'   # ← replace this
AWS_REGION = 'us-east-1'
MODEL_ID = 'amazon.nova-pro-v1:0'

# ── AWS Clients ───────────────────────────────────────────────
bedrock_agent = boto3.client(
    service_name='bedrock-agent-runtime',
    region_name=AWS_REGION
)
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=AWS_REGION
)

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title='HR Document Assistant',
    page_icon='📋',
    layout='wide'
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title('📋 HR Assistant by Prasad Pate')
    st.markdown('**Powered by AWS Bedrock + Amazon Nova**')
    st.divider()
    st.markdown('### About')
    st.markdown(
        'This assistant uses Retrieval-Augmented Generation (RAG) '
        'to answer questions from your HR document library.'
    )
    st.divider()
    st.markdown('### Sample Questions')
    st.markdown('- What is the annual leave policy?')
    st.markdown('- How does the performance review process work?')
    st.markdown('- What global mobility support is available?')
    st.markdown('- What are my GDPR data rights as an employee?')
    st.markdown('- What is the promotion eligibility criteria?')
    st.divider()
    if st.button('🗑️ Clear Chat History'):
        st.session_state.messages = []
        st.rerun()

# ── Main Header ───────────────────────────────────────────────
st.title('📋 HR Document Intelligence Assistant')
st.caption('Powered by AWS Bedrock · Amazon Nova Pro · Ask anything about HR policies')
st.divider()

# ── Chat History ──────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# ── Query Function ────────────────────────────────────────────
def query_knowledge_base(question):
    try:
        # Step 1 — Retrieve relevant chunks from Knowledge Base
        retrieve_response = bedrock_agent.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={
                'text': question
            },
            retrievalConfiguration={
                'managedSearchConfiguration': {
                    'numberOfResults': 5
                }
            }
        )

        # Step 2 — Extract text chunks and source filenames
        chunks = retrieve_response.get('retrievalResults', [])

        if not chunks:
            return (
                'I could not find relevant information in the HR documents. '
                'Please try rephrasing your question.'
            ), []

        context = '\n\n'.join([
            c['content']['text'] for c in chunks
        ])

        sources = list(set([
            c.get('location', {})
             .get('s3Location', {})
             .get('uri', '')
             .split('/')[-1]
            for c in chunks
            if c.get('location', {})
               .get('s3Location', {})
               .get('uri')
        ]))

        # Step 3 — Build prompt
        prompt = f"""You are a helpful HR assistant. Use the following excerpts
from HR policy documents to answer the employee question accurately.
If the answer is not in the excerpts, say honestly that you do not
have that information and suggest they contact HR directly.

HR DOCUMENT EXCERPTS:
{context}

EMPLOYEE QUESTION: {question}

ANSWER:"""

        # Step 4 — Call Amazon Nova Pro
        body = json.dumps({
            'messages': [
                {
                    'role': 'user',
                    'content': [{'text': prompt}]
                }
            ],
            'inferenceConfig': {
                'maxTokens': 1000,
                'temperature': 0.1
            }
        })

        generate_response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            body=body
        )

        # Step 5 — Parse Nova Pro response format
        answer = json.loads(
            generate_response['body'].read()
        )['output']['message']['content'][0]['text']

        return answer, sources

    except Exception as e:
        return f'Error: {str(e)}', []

# ── Chat Input ────────────────────────────────────────────────
if prompt := st.chat_input('Ask about leave, performance, mobility, GDPR...'):

    # Add user message to history and display it
    st.session_state.messages.append(
        {'role': 'user', 'content': prompt}
    )
    with st.chat_message('user'):
        st.markdown(prompt)

    # Get answer and display it
    with st.chat_message('assistant'):
        with st.spinner('Searching HR documents...'):
            answer, sources = query_knowledge_base(prompt)
        st.markdown(answer)
        if sources:
            st.caption('📎 Sources: ' + ', '.join(sources))

    # Add assistant response to history
    st.session_state.messages.append(
        {'role': 'assistant', 'content': answer}
    )
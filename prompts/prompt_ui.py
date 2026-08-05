from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

# Load environment variables from .env (needs HUGGINGFACEHUB_API_TOKEN)
load_dotenv()

# Set up the base LLM endpoint via Hugging Face's hosted API
# - repo_id / provider: using the combo confirmed working earlier
#   (openai/gpt-oss-120b via the Cerebras provider)
# - task: type of generation task
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    provider="cerebras",
    task="text-generation"
)

# Wrap the raw endpoint in a chat interface
model = ChatHuggingFace(llm=llm)

# Streamlit page header
st.header('Research Tool')

# Dropdown to select which research paper to summarize
paper_input = st.selectbox(
    "Select Research Paper Name",
    ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers",
     "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"]
)

# Dropdown to select explanation style
style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

# Dropdown to select explanation length
length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# Load a pre-built prompt template from file
# (template.json defines the prompt structure with placeholders for the
#  paper, style, and length selections made above)
template = load_prompt('template.json')

# When the user clicks the "Summarize" button:
if st.button('Summarize'):
    # Chain the template into the model: fill the template, then send to the LLM
    chain = template | model

    # Run the chain with the selected dropdown values
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })

    # Display the model's response in the Streamlit app
    st.write(result.content)

# ---------------------------------------------------------------------------
# WHAT IS STREAMLIT?
#
# Streamlit is a Python library used to quickly build interactive web apps
# for data/ML projects — without writing any HTML, CSS, or JavaScript.
#
# You write plain Python (like the code above), and Streamlit turns function
# calls like st.header(), st.selectbox(), and st.button() into actual UI
# elements (headers, dropdowns, buttons) rendered in a browser.
#
# It re-runs the entire script top-to-bottom every time the user interacts
# with a widget (e.g. clicking "Summarize"), which is why the whole flow
# above — from dropdown selection to model call — happens in one linear
# script rather than a traditional event-driven web app structure.
#
# Run this file with:  streamlit run <filename>.py
# ---------------------------------------------------------------------------
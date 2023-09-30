import streamlit as st
from transformers import (
    pipeline,
    AutoModelForQuestionAnswering,
    AutoTokenizer,    
)


st.title("Question Answering App")

model_name = "hzsushiqiren/bert-finetuned-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Use session state to manage input values
if "input_values" not in st.session_state:
    st.session_state.input_values = {"question": "", "context": "", "answer": ""}

# Add text inputs for question and context
question_input = st.text_input("Question:", value=st.session_state.input_values["question"])
context_input = st.text_input("Context:", value=st.session_state.input_values["context"])


# Create a button for get answers
get_button = st.button("Get Answer")

# Create a button for resetting inputs
reset_button = st.button("Reset")

if get_button:
    QA_input = {
        'question': question_input,
        'context': context_input
    }

    res = nlp(QA_input)

    st.text_area("Answer:", res['answer'])
    st.write("Score:", res['score'])
    st.session_state.input_values["answer"] = res['answer']

# Handle reset button click
if reset_button:
    question_input = ""
    context_input = ""    
    st.session_state.input_values = {"question": "", "context": "", "answer": ""}

# Update the input values when the user enters text
if not reset_button:  # Only update when the reset button is not clicked
    st.session_state.input_values["question"] = question_input
    st.session_state.input_values["context"] = context_input

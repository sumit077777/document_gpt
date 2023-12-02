import streamlit as st
import os
from ingest import create_vector_db  # Assuming create_vector_db is a custom function
from test import main

# Create a folder to store uploaded files
upload_folder = 'data'
os.makedirs(upload_folder, exist_ok=True)

# Create a file uploader in the sidebar
uploaded_files = st.sidebar.file_uploader("Upload files", type=['pdf'], accept_multiple_files=True)

# Check if files have been uploaded
if uploaded_files is not None:
    st.sidebar.write("Files uploaded successfully!")

    for file in uploaded_files:
        # Save the file to the "uploads" folder
        file_path = os.path.join(upload_folder, file.name)
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())

    # Additional methods
    if st.sidebar.button("Clear Files"):
        for file in uploaded_files:
            file.empty()  # Clear each file uploader widget

# Process button
process_button = st.sidebar.button("Click to process the files")
if process_button:
    st.spinner("Processing....")
    create_vector_db()  # Call your custom function here
    st.success("Processing complete!")
query=st.text_area(
    label="enter your query:",
    max_chars=100,
    height=20
)
ask=st.button(label="Ask")
if ask:
    st.write(main(query))
    st.spinner("processing...")
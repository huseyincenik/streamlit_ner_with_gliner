import streamlit as st
from gliner import GLiNER
import pandas as pd
import base64
import time
import io

linkedin_profile_link = "https://www.linkedin.com/in/huseyincenik/"
kaggle_profile_link = "https://www.kaggle.com/huseyincenik/"
github_profile_link = "https://github.com/huseyincenik/"

st.sidebar.markdown(
    f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_profile_link}) "
    f"[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)]({kaggle_profile_link}) "
    f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_profile_link})"
)

# GLiNER model seçenekleri
models = {
    "urchade/gliner_multi-v2.1": "urchade/gliner_multi-v2.1",
    "urchade/gliner_medium-v2.1": "urchade/gliner_medium-v2.1",
    "urchade/gliner_small-v2.1": "urchade/gliner_small-v2.1",
    "urchade/gliner_large-v2.1": "urchade/gliner_large-v2.1"
}

# Önbellek için GLiNER modeli
@st.cache_resource
def load_gliner_model(model_name):
    return GLiNER.from_pretrained(model_name, use_fast=False)

st.image("https://miro.medium.com/v2/resize:fit:1400/0*ua9IMC80xZozdy3V.png", use_column_width=True)
st.markdown("**Source:** [Extract Any Entity from Text with GLiNER](https://towardsdatascience.com/extract-any-entity-from-text-with-gliner-32b413cea787)")


# Function to load data
@st.cache_data()
def load_data(file):
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        data = pd.read_excel(file)
    return data

# Function to enter the number of new columns
def enter_number_of_columns():
    num_columns = st.sidebar.number_input("Enter the number of new columns", min_value=1, step=1, help="Number of new columns to be created with ner")
    return num_columns

# Function to enter new column details
def enter_column_details(data, num_columns):
    column_details = []
    for i in range(num_columns):
        st.sidebar.write(f"### New Column {i+1}")
        with st.sidebar.expander(f"New Column {i+1} Details"):
            select_column = st.selectbox(f"Select column {i+1}:", data.columns, key=f"select_column_{i}", help="Select the column from your dataset to use for creating the new column.")
            new_column_name = st.text_input(f"Enter the new column name {i+1}:", key=f"new_column_name_{i}", help="Enter the name of the new column to be created.")
            # Check if the new column name already exists in the data
            if new_column_name in data.columns:
                st.warning("The new column name already exists in the dataset. Please enter a different column name.")
                # continue
            label = st.text_input(f"Enter the label {i+1}:", key=f"label_{i}", help="Enter the label for Named Entity Recognition (NER) model. One expression or multiple expressions separated by commas can be written.")
            threshold = st.slider(f"Select threshold {i+1}:", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=f"threshold_{i}", help="Select the threshold for NER model.")
            column_details.append((select_column, new_column_name, label, threshold))
    return column_details

# Function to process data and show results for each new column
def process_data_for_columns(data, column_details, selected_model):
    model = load_gliner_model(selected_model)
    for i, (selected_column, new_column_name, label, threshold) in enumerate(column_details):
        st.write(f"### Processed Data for **{new_column_name}** (Model: {selected_model})")
        selected_data = {
            'Selected Column': [selected_column],
            'New Column Name': [new_column_name],
            'Label': [label],
            'Threshold': [threshold],
            'Selected Model': [selected_model]
        }
        df_data = pd.DataFrame(selected_data)
        st.write("Column Details:")
        st.write(df_data)

        
        try:
            if selected_column in data.columns:
                total_rows = len(data)
                st.write(f"Number of total data: {total_rows}")
                st.write("Processed data:")
                # Show all outputs in an expander
                with st.expander(f"All Results for {new_column_name}"):
                    for index, row in data.iterrows():
                        text_to_process = row[selected_column]
                        st.write(f"Text {index + 1}: {text_to_process}")
                        entities = model.predict_entities(text_to_process, label.split(','), threshold=threshold)
                        if len(entities) == 0:
                            result = "Unknown"
                        else:
                            if len(label.split(',')) == 1:
                                result = ', '.join(set([entity['text'] for entity in entities]))
                            else:
                                result = ', '.join(set([f"{entity['text']}:{entity['label']}" for entity in entities]))
                        st.write(f"Result: {result}")
                        
                        # Assign to new column
                        data.at[index, new_column_name] = result
                
                st.success(f"Data processing completed for {new_column_name}.")
            else:
                st.error("The selected column was not found in the dataset.")
        except Exception as e:
            st.error("An error occurred. The model will retry after waiting for 1 minute.")
            st.error(f"Error Message: {str(e)}")
            time.sleep(60)  # Wait for 1 minute
            st.info("Please try again.")
            process_data_for_columns(data, column_details, selected_model)  # Call the model again

def main():
    st.title("🔎 NER (Named Entity Recognition) Application with Gliner (v1)🛠️")
    st.write("For detailed information about the models, click [here](https://huggingface.co/models?search=urchade%2Fgliner)")

    # Model selection
    selected_model = st.sidebar.selectbox("Select the GLiNER model:", list(models.keys()), index=1)  # Default: urchade/gliner_medium-v2.1

    # Load the data
    file = st.sidebar.file_uploader("Upload your data file (in CSV or Excel format)", type=["csv", "xlsx"])
    if file:
        data = load_data(file)
        
        # Enter the number of new columns
        num_columns = enter_number_of_columns()
        
        # Enter details for each new column
        column_details = enter_column_details(data, num_columns)
        
        # Process the data and show results for each new column
        if st.sidebar.button("Process Data"):
            process_data_for_columns(data, column_details, models[selected_model])

        # Download the data as Excel
        output_file = io.BytesIO()
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
        data.to_excel(writer, index=False)
        writer.close()
        excel_data = output_file.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="extracted_data.xlsx">Download Data (Excel)</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

import streamlit as st
from gliner import GLiNER
import pandas as pd
import base64
import time
import io
import xlsxwriter

# GLiNER modelini yükle
model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")

linkedin_profile_link = "https://www.linkedin.com/in/huseyincenik/"
kaggle_profile_link = "https://www.kaggle.com/huseyincenik/"
github_profile_link = "https://github.com/huseyincenik/"

st.sidebar.markdown(
    f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_profile_link}) "
    f"[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)]({kaggle_profile_link}) "
    f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_profile_link})"
)



# Function to load data
def load_data():
    uploaded_file = st.sidebar.file_uploader("Upload your data file (in CSV or Excel format)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        return data
    return None

# Function to enter the number of new columns
def enter_number_of_columns():
    num_columns = st.sidebar.number_input("Enter the number of new columns", min_value=1, step=1, help="Number of new columns to be created with ner")
    return num_columns

# Function to enter new column details

# Function to enter new column details
def enter_column_details(data, num_columns):
    column_details = []
    for i in range(num_columns):
        st.sidebar.write(f"### New Column {i+1}")
        with st.sidebar.expander(f"New Column {i+1} Details"):
            select_column = st.selectbox(f"Select column {i+1}:", data.columns, key=f"select_column_{i}", help="Select the column from your dataset to use for creating the new column.")
            new_column_name = st.text_input(f"Enter the new column name {i+1}:", key=f"new_column_name_{i}", help="Enter the name of the new column to be created.")
            label = st.text_input(f"Enter the label {i+1}:", key=f"label_{i}", help="Enter the label for Named Entity Recognition (NER) model.")
            threshold = st.slider(f"Select threshold {i+1}:", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=f"threshold_{i}", help="Select the threshold for NER model.")
            column_details.append((select_column, new_column_name, label, threshold))
    return column_details

# Function to process data and show results for each new column
def process_data_for_columns(data, column_details):
    for i, (selected_column, new_column_name, label, threshold) in enumerate(column_details):
        st.header(f"Processed Data for **{new_column_name}** Column")
        selected_data = {
            'Selected Column': [selected_column],
            'New Column Name': [new_column_name],
            'Label': [label],
            'Threshold': [threshold]
        }
        df_data = pd.DataFrame(selected_data)
        st.dataframe(df_data)
        
        try:
            if selected_column in data.columns:
                total_rows = len(data)
                st.write(f"Number of total data: {total_rows}")
                st.write("Processed data:")
                
                # Show all outputs in an expander
                with st.expander(f"All Results for {new_column_name} column."):
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
                
                st.success(f"Data processing completed for {new_column_name} column.")
            else:
                st.error("The selected column was not found in the dataset.")
        except Exception as e:
            st.error("An error occurred. The model will retry after waiting for 1 minute.")
            st.error(f"Error Message: {str(e)}")
            time.sleep(60)  # Wait for 1 minute
            st.info("Please try again.")
            process_data_for_columns(data, column_details)  # Call the model again
def main():
    st.title("🔎 NER (Named Entity Recognition) Application with Gliner Multiv2.1 🛠️")
    st.write("For detailed information about the model, click [here](https://huggingface.co/spaces/urchade/gliner_multiv2.1)")

    
    # Load the data
    data = load_data()
    if data is not None:
        # Enter the number of new columns
        num_columns = enter_number_of_columns()
        
        # Enter details for each new column
        column_details = enter_column_details(data, num_columns)
        
        # Process the data and show results for each new column
        if st.sidebar.button("Process Data"):
            process_data_for_columns(data, column_details)
            st.snow()

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
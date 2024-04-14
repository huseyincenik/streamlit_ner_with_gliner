# NER (Named Entity Recognition) Application with Gliner

![NER Application](https://miro.medium.com/v2/resize:fit:1400/0*ua9IMC80xZozdy3V.png)

[Extract Any Entity from Text with GLiNER](https://towardsdatascience.com/extract-any-entity-from-text-with-gliner-32b413cea787)

This is a Named Entity Recognition (NER) application powered by Gliner, implemented using Streamlit. NER allows users to identify and classify named entities in unstructured text.

## Features

- Upload your dataset in CSV or Excel format.
- Select a GLiNER model from the available options.
- Define new columns with NER labels (separated by commas for multiple labels in one column).
- Set thresholds for precision.
- Process the data and visualize the results for each new column.
- Download the processed data as an Excel file.

## GLiNER Models

The application provides the following GLiNER models to choose from:
- `urchade/gliner_multi-v2.1`
- `urchade/gliner_medium-v2.1`
- `urchade/gliner_small-v2.1`
- `urchade/gliner_large-v2.1`

## Installation

1. Clone the repository:

```bash
git clone https://github.com/huseyincenik/streamlit_ner_with_gliner.git
cd streamlit_ner_with_gliner
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
streamlit run app.py
```

## Usage

1. Upload your data file (in CSV or Excel format).
2. Select the GLiNER model.
3. Enter the number of new columns and define the details for each new column (column name, label, threshold).
4. Click on the "Process Data" button to start the data processing.
5. Download the processed data as an Excel file.

## Links

- [Streamlit App](https://ner-with-gliner.streamlit.app/)
- [Github Repository](https://github.com/huseyincenik/streamlit_ner_with_gliner)
- [Models Description](https://huggingface.co/models?library=gliner)
- [Linkedin Post](https://www.linkedin.com/feed/update/urn:li:activity:7185252614794600448/)

[![](https://visitcount.itsvg.in/api?id=huseyincenik.streamlit_ner_with_gliner/&label=Visiter%20Count&color=10&icon=9&pretty=false)](https://visitcount.itsvg.in)

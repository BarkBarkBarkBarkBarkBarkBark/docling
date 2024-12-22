import os
import openai
import json
import re
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Retrieve OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
#if not api_key:
#    raise ValueError("OPENAI_API_KEY is not set. Please check your environment variables.")

client = openai.Client(api_key=api_key)

# Configure the pipeline for processing PDF files
pipeline_options = PdfPipelineOptions(do_table_structure=True)
pipeline_options.table_structure_options.do_cell_matching = False

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# Define input and output directories
input_dir = "Amazon_Docs"
markdown_output_dir = "markdown_outputs"
json_output_dir = "json_outputs"

# Create output directories if they do not exist
os.makedirs(markdown_output_dir, exist_ok=True)
os.makedirs(json_output_dir, exist_ok=True)

# Function to clean unwanted Markdown syntax and spaces in JSON
def clean_json_content(json_data):
    if isinstance(json_data, dict):
        return {key: clean_json_content(value) for key, value in json_data.items()}
    elif isinstance(json_data, list):
        return [clean_json_content(item) for item in json_data]
    elif isinstance(json_data, str):
        clean_text = re.sub(r"(\*{1,2}|`)", "", json_data)  # Remove markdown syntax
        clean_text = re.sub(r"\s{2,}", " ", clean_text)  # Remove extra spaces
        return clean_text.strip()
    return json_data

# Function to call OpenAI API and convert Markdown to JSON
def generate_json_from_markdown(markdown_content):
    prompt = f"""
    You are a highly capable assistant tasked with transforming Markdown into JSON. 
    purchases will be matched using the date, so it is critically important that date is formated as YYYY-MM-DD
    Adhere strictly to this structure:
    {{
        "order_number": "string",
        "order_date": "string",
        "shipped_date": "string",
        "shipping_address": {{
            "name": "string",
            "address_line_1": "string",
            "city": "string",
            "state": "string",
            "zip_code": "string",
            "country": "string"
        }},
        "billing_address": {{
            "name": "string",
            "address_line_1": "string",
            "city": "string",
            "state": "string",
            "zip_code": "string",
            "country": "string"
        }},
        "items": [
            {{
                "description": "string",
                "quantity": "integer",
                "price": "float",
                "condition": "string"
            }}
        ],
        "totals": {{
            "subtotal": "float",
            "discounts": "float",
            "shipping_handling": "float",
            "tax": "float",
            "grand_total": "float"
        }},
        "payment_information": {{
            "method": "string",
            "transaction_date": "string",
            "amount": "float"
        }},
        "shipping_speed": "string"
    }}

    Markdown Input:
    {markdown_content}

    Return only JSON without any text or comments.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a JSON conversion assistant. Output JSON only."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content.strip()

        # Extract JSON from Markdown code block
        match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
        if match:
            content = match.group(1).strip()
        
        return content
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return None

# Process PDFs: Convert to Markdown and then to JSON
pdf_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".pdf")]

for pdf_file in pdf_files:
    try:
        # Convert PDF to Markdown
        result = doc_converter.convert(pdf_file)
        markdown_content = result.document.export_to_markdown()

        # Save Markdown file
        markdown_file = os.path.join(markdown_output_dir, os.path.basename(pdf_file).replace(".pdf", ".md"))
        with open(markdown_file, "w") as md_file:
            md_file.write(markdown_content)

        # Convert Markdown to JSON using OpenAI
        print(f"Processing {markdown_file}...")
        json_output = generate_json_from_markdown(markdown_content)

        if json_output:
            try:
                # Attempt to parse JSON
                parsed_json = json.loads(json_output)
                cleaned_json = clean_json_content(parsed_json)

                # Save cleaned JSON to file
                json_file = os.path.join(json_output_dir, os.path.basename(pdf_file).replace(".pdf", ".json"))
                with open(json_file, "w") as j_file:
                    json.dump(cleaned_json, j_file, indent=4)
                print(f"JSON saved: {json_file}")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON format: {e}")
        else:
            print(f"Failed to generate JSON for {markdown_file}")

    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")
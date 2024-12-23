from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
import os

# Configure the pipeline for table structure processing
pipeline_options = PdfPipelineOptions(do_table_structure=True)
pipeline_options.table_structure_options.do_cell_matching = False  # Avoid text cell matching issues

# Initialize the DocumentConverter with pipeline options for PDFs
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# Specify the input directory containing PDFs
input_dir = input("Directory Path: ")
output_dir = "/workspaces/docling/markdown_outputs"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all PDF files in the input directory
pdf_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".pdf")]

# Process each PDF file
for pdf_file in pdf_files:
    try:
        # Convert the PDF to a document object
        result = doc_converter.convert(pdf_file)

        # Export to Markdown
        markdown_content = result.document.export_to_markdown()

        # Save Markdown content to a file
        output_file = os.path.join(output_dir, os.path.basename(pdf_file).replace(".pdf", ".md"))
        with open(output_file, "w") as md_file:
            md_file.write(markdown_content)
        
        print(f"Converted {pdf_file} to {output_file}")
    except Exception as e:
        print(f"Failed to process {pdf_file}: {e}")

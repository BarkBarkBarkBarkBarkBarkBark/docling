from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Configure the pipeline for table structure processing
pipeline_options = PdfPipelineOptions(do_table_structure=True)
pipeline_options.table_structure_options.do_cell_matching = False  # Avoid text cell matching issues

# Initialize the DocumentConverter with pipeline options for PDFs
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# Define the source PDF and perform the conversion
source = input("Full Path of PDF: ")
result = doc_converter.convert(source)

# Define output file path
output_file = source.rsplit('.', 1)[0] + ".md"  # Use the same name as the PDF, but with .md extension

# Export the result to Markdown and save to a file
with open(output_file, "w") as md_file:
    md_file.write(result.document.export_to_markdown())

print(f"Markdown file saved to: {output_file}")

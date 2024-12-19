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
source = "/workspaces/docling/Amazon.com_-_Order_112-4553907-9354650.pdf"
result = doc_converter.convert(source)

# Export the result to Markdown and print
print(result.document.export_to_markdown())

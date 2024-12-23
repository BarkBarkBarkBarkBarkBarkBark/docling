import openai
import sys
import os
import tiktoken
import re
from dotenv import load_dotenv

# Load the OpenAI API key from environment variables
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
print(f"OpenAI API Key: {'Loaded' if openai_api_key else 'Not Loaded'}")

# Function to estimate tokens accurately using tiktoken
def estimate_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Function to read content from a file
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file '{file_path}': {e}")
        sys.exit(1)

# Main execution
if __name__ == "__main__":
    # Check if the input file was provided as a command-line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("Markdown File Path: ")

    # Prompt user for target language
    language = input("Target language for translation: ")

    # Read content from the specified file
    content = read_file(input_file)

    # Set the model name
    model_name = "gpt-4"  # Change to "gpt-3.5-turbo" if GPT-4 is not available

    # Define the maximum context length for the model
    if "gpt-3.5-turbo" in model_name:
        max_context_length = 4096
    else:
        max_context_length = 8192  # For GPT-4

    # Adjusted token calculations
    desired_response_tokens = 800  # Adjustment: Lowered desired output length for more chunks
    buffer_tokens = 100  # Adjustment: Increased buffer for safety
    system_prompt_tokens = estimate_tokens("You machine learning expert with a passion for summarizing complex information.", model=model_name)
    per_message_tokens = 50  # Approximate tokens for prompts

    max_chunk_tokens = min(max_context_length - system_prompt_tokens - desired_response_tokens - buffer_tokens - per_message_tokens, 1500)  # Adjustment: Stricter chunk size

    # Debugging for token calculation
    print(f"Max Chunk Tokens: {max_chunk_tokens}")

    # Function to split text into chunks based on markdown headers (#, ##, ###, ####, #####) and line breaks
    def split_text_into_chunks(text, max_chunk_tokens, model="gpt-4"):
        header_pattern = re.compile(r'(#{1,5} .+\n)')  # Match headers (# to #####)
        parts = re.split(header_pattern, text)

        chunks = []
        encoding = tiktoken.encoding_for_model(model)

        current_chunk = ""
        current_tokens = 0

        for part in parts:
            part_tokens = len(encoding.encode(part))

            # Print token sizes for debugging
            print(f"Part: {part[:30]}... Tokens: {part_tokens}")

            if current_tokens + part_tokens <= max_chunk_tokens:
                current_chunk += part
                current_tokens += part_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    print(f"Chunk created with {current_tokens} tokens.")
                # Reset for the next chunk
                current_chunk = part
                current_tokens = part_tokens

                # If the current part is larger than max_chunk_tokens, append it directly
                if current_tokens > max_chunk_tokens:
                    print(f"Warning: Large part exceeds max_chunk_tokens ({current_tokens} tokens). Splitting as standalone chunk.")
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0

        if current_chunk:
            chunks.append(current_chunk.strip())
            print(f"Final Chunk created with {current_tokens} tokens.")

        return chunks

    # Function to generate the translated text for each chunk
    def translate_text(chunk, model_name, language):
        system_prompt = "You are an expert in machine learning, skilled at summarizing complex topics in an engaging and accessible way. You prioritize clarity and brevity, avoiding unnecessary detail or formalities. Your goal is to explain big-picture ideas, share why they matter, and inspire curiosity, especially for tech-savvy learners with basic machine learning knowledge."        
        user_prompt = f""" You are an expert in machine learning. Please summarize the following content into {language} for a podcast-style audio recording. please reduce each section to a few paragraphes, and elaborate on why the topic matters

        do not translate the material directly, but instead summarize its main points in order. add a little zest and personality in there to make it engaging.

        the text will be converted to speech, so please handle bullet points using natural language

        dont mention details in the prompt,
        
        summarize the material in an awesome and exciting way

Focus on the big-picture ideas, and provide background for difficult concepts to ensure listeners understand the key takeaways. Share your interpretation of why this content matters to machine learning, why it’s exciting, and how it’s valuable for new learners.

Present the information clearly and engagingly, without overexplaining. Assume the listener works in tech and has a basic understanding of machine learning. Avoid directly referencing figures, tables, or formulas. Instead, interpret them and explain their significance. This is an audio format, so content should be digestible and relatable for someone listening while multitasking.

Please summarize the following content and explain its relavence

Content:
{chunk}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Calculate total tokens for messages
        total_message_tokens = sum([estimate_tokens(m['content'], model=model_name) for m in messages])

        # Set desired response tokens and a buffer
        max_response_tokens = max_context_length - total_message_tokens - buffer_tokens

        if max_response_tokens <= 0:
            print("Error: The input chunk is too large to process. Consider reducing the chunk size.")
            sys.exit(1)

        try:
            response = openai.chat.completions.create(  # Adjustment: Fixed the OpenAI API call syntax
                model=model_name,
                messages=messages,
                max_tokens=min(desired_response_tokens, max_response_tokens),
                temperature=0.7,
                n=1
            )
            return response.choices[0].message.content
        except openai.OpenAIError as e:
            print(f"An error occurred during the OpenAI API call: {e}")
            sys.exit(1)

    # Split the content into chunks
    print('Splitting into chunks...')
    chunks = split_text_into_chunks(content, max_chunk_tokens, model=model_name)

    # Translate each chunk and combine the results
    full_translation = ""
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1} of {len(chunks)}...")
        translated_chunk = translate_text(chunk, model_name, language)
        full_translation += translated_chunk + "\n\n"

    # Define the output file name based on the input file name
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{base_name}_translated_{language}.txt"

    # Save the translated content to the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(full_translation)
        print(f"Translated content has been saved to '{output_file}'.")
    except Exception as e:
        print(f"An error occurred while writing to the file '{output_file}': {e}")
        sys.exit(1)

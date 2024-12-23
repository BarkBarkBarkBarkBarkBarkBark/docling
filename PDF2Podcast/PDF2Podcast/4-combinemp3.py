import os
from pydub import AudioSegment

# Directory containing the MP3 files
input_directory = "/workspaces/docling/StudyBuddy/PyMuPDF/output mp3"
output_file = "/workspaces/docling/StudyBuddy/PyMuPDF/combined_output.mp3"

# List all MP3 files in the directory
mp3_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.mp3')]

# Ensure the MP3 files are sorted (optional, for sequential playback)
mp3_files.sort()

# Combine MP3 files
combined_audio = AudioSegment.empty()

print("Combining the following files:")
for file in mp3_files:
    print(file)
    audio = AudioSegment.from_file(file, format="mp3")
    combined_audio += audio

# Export the combined audio to a new MP3 file
combined_audio.export(output_file, format="mp3")
print(f"Combined audio saved to: {output_file}")

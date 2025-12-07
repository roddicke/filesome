import os
from markitdown import MarkItDown

md = MarkItDown()
# Get the directory of the current script
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
script_filename ="【ダウンロード資料】ディープラーニングの概要"
script_fileformat = ".pdf"
# Construct the path to the PDF in the current directory (others)
pdf_path = os.path.join(script_dir, script_filename+script_fileformat)


result = md.convert(pdf_path)

# Define output file path
output_path = os.path.join(script_dir, script_filename+".md")

# Write the content to the file
with open(output_path, "w", encoding="utf-8") as f:
    f.write(result.text_content)

print(f"Conversion complete. Output saved to: {output_path}")
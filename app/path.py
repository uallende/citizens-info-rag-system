import os

print(os.getcwd())

output_dir = './app/pdf_docs'
output_dir_abs = os.path.abspath(output_dir)
print(output_dir_abs)
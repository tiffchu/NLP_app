def clean_docx_file(file):
    file_name = file.name.lower().lstrip("relationship ").rstrip(".docx")
    print(file_name)
    return None
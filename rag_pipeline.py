from pathlib import Path

#define document path
Document_path= Path("documents/banking_faq.txt")

def load_document(file_path):
    with open(file_path,"r", encoding="utf-8") as file:
        text = file.read()

    return text

def split_into_chunks(text):
   """ #split the text into individual chunks"""
   chunks =[
       section.strip()
       for section in text.split("\n\n")
       if section.strip()
   ]

   return chunks
            
            
            


from langchain_community.document_loaders import WebBaseLoader

url = "https://www.samsung.com/in/?srsltid=AfmBOoqdjkBLThagGzizgLerQ7WmTatfqllsdJ3lyNwE9FKQUN2XE8RA"

data = WebBaseLoader(url)

docs = data.load()

print(docs[0].page_content)
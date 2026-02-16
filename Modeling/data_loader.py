from configuration import *

class DataLoaderAgent:
    def __init__(self):
        self.text_splitter = CharacterTextSplitter(chunk_size=CHUNKS_SIZE, chunk_overlap=OVERLAP)

    # --- Method for RAG For CSV and PDF and XLSX (Text Chunks) ---
    def load_documents_for_rag(self, path):
        """Returns chunks for VectorDB"""
        if path.endswith(".pdf"):
            return self._load_pdf_chunks(path)
        elif path.endswith(".csv"):
            loader = CSVLoader(path)
            docs = loader.load()
            return self.text_splitter.split_documents(docs)
        return []

    def _load_pdf_chunks(self, path):
        docs = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    docs.append(Document(page_content=text, metadata={"page": i + 1}))
        return self.text_splitter.split_documents(docs)

    # --- Method for Analysis Just for CSV and Xlsx (DataFrame) ---
    def load_df(self, path):
        """Returns Pandas DataFrame for EDA/ML"""
        if path.endswith(".csv"):
            return pd.read_csv(path)
        elif path.endswith(".xlsx") or path.endswith(".xls"):
            return pd.read_excel(path)
        else:
            raise ValueError("Unsupported file format for DataFrame loading")

    def clean_text_chunks(self, docs):
        cleaned = []
        for d in docs:
            txt = re.sub(r"\s+", " ", d.page_content)
            cleaned.append(Document(page_content=txt))
        return cleaned


def create_vectordb(chunks, embedding_model):
    """Create FAISS vector database from documents"""
    vectordb = FAISS.from_documents(chunks, embedding_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5}) 
    return vectordb , retriever
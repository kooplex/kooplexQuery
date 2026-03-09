import chromadb
import logging
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

logger = logging.getLogger(__name__)


class VectorStore():
    CollectionNames = {
        "EXAMPLES": "examples",
        "DOCS": "docs",
        "ADVICES": "advices",
        "SCHEMA": "schema"
    }

    def __init__(self, persist_directory="./chroma_vector_db"):
    # def __init__(self):
        
        self.embeddings = None

        # self.persist_directory = persist_directory
        # self.vectorstore = None
        
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        # self.chroma_client = chromadb.Client()

        self.example_selector = None 
        self.example_prompt = None
        self.examples = self._init_db(self.CollectionNames["EXAMPLES"])
        self.docs = self._init_db(self.CollectionNames["DOCS"])
        self.advices = self._init_db(self.CollectionNames["ADVICES"])
        self.schema = self._init_db(self.CollectionNames["SCHEMA"])

    def get_collections(self):
        """ Get all collection names in the vectorstore.  """
        # only user defined variables in CollectionNames class
        return [col for col in self.CollectionNames.values()]

    def _init_db(self, collection_name, reset=False):
        # Init vectorstore if not already

            # Reset collection if needed
            if reset:
                try:
                    self.chroma_client.delete_collection(collection_name)
                except:
                    print("'{collection_name}' collection in DB couldn't be found")
                    pass
                    
            if not self.embeddings:
                # with local CPU or GPU Usage
                model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
                gpt4all_kwargs = {'allow_download': 'True'}
                self.embeddings = GPT4AllEmbeddings(
                    model_name=model_name,
                    gpt4all_kwargs=gpt4all_kwargs
                )

            # # Init collection if not already
            try:
                c = self.chroma_client.create_collection(collection_name)
            except:
                logger.info(f"Collection {collection_name} already exists in DB")
                pass

            # if not self.examples:
            #     c = self.chroma_client.get_or_create_collection(collection_name)
            #     if not c.count():
            #         logger.info(f"Collection {collection_name} is empty")
            # try:
            #     collection = self.chroma_client.get_collection(collection_name)
            #     logger.info(f"Collection {collection_name} found in DB")
            # except:
            #     pass
            if collection_name in self.get_collections():
                logger.info(f"Collection {collection_name} initialized in DB")
                return Chroma(client=self.chroma_client, embedding_function=self.embeddings, collection_name=collection_name)
            # return self.chroma_client.get_or_create_collection( embedding_function=self.embeddings, name=collection_name)
            # return self.chroma_client.get_or_create_collection(name=collection_name)
    
    def _select_collection_by_name(self, collection_name):
        """ Select collection by name.  """
        if collection_name == self.CollectionNames["EXAMPLES"]:
            return self.examples
        elif collection_name == self.CollectionNames["DOCS"]:
            return self.docs
        elif collection_name == self.CollectionNames["ADVICES"]:
            return self.advices
        elif collection_name == self.CollectionNames["SCHEMA"]:
            return self.schema  
        else:
            logger.error(f"Collection {collection_name} not found! Available collections: {self.get_collections()}")
            return None

    def _check_similarity(self, text, collection):
        """ Check if item exists in collection.  """
        try:
            distance = collection.similarity_search_with_score(text, k=1)[0][-1]
        except:
            distance = 1
        if distance < 0.0001:
            logger.debug(f"Text {text} already exists in {collection} collection")
            return True
        else:
            return False

    def add_to_examples(self, item):
        """ Check if item exists. Add item to collection.  """

        self.examples = self._init_db(self.CollectionNames["EXAMPLES"])   

        texts = [f"{item['question']}"]
        metadatas = [{"sql": item["sql"],  "question":item["question"], "type":self.CollectionNames["EXAMPLES"]}]


        if not self._check_similarity(texts[0], self.examples):
            self.examples.add_texts(texts = texts, metadatas=metadatas)

            # Make sure that there are no duplicates
            logger.info(f"Added to {self.CollectionNames['EXAMPLES']} collection")

    def add_to_docs(self, item=None, texts="", metadatas=""):
        """ Check if item exists. Add item to docs.  """

        self.docs = self._init_db(self.CollectionNames["DOCS"])   

        if item:
            texts = [f"{item['question']}"]
            metadatas = [{"sql": item["sql"],  "question":item["question"], "type":self.CollectionNames["DOCS"]}]

        # Check similarity
        if not self._check_similarity(texts[0], self.docs):
            self.docs.add_texts(texts = texts, metadatas=metadatas)

            # Make sure that there are no duplicates
            logger.info(f"Added to {self.CollectionNames['DOCS']} collection")

    def add_to_advices(self, item=None, texts="", metadatas=""):
        """ Check if item exists. Add item to advices.  """

        self.advices = self._init_db(self.CollectionNames["ADVICES"])   

        if item:
            texts = [f"{item['advice']}"]
            metadatas = [{"advice": item["advice"], "type":self.CollectionNames["ADVICES"]}]

        # Check similarity
        if not self._check_similarity(texts[0], self.advices):
            self.advices.add_texts(texts = texts, metadatas=metadatas)

            # Make sure that there are no duplicates
            logger.info(f"Added to {self.CollectionNames['ADVICES']} collection")

    # ADDITIONAL DOCUMENTATION
    def load_split_add_csv(self, file_path, collection_name=None, csv_args={'delimiter': '\t'}):
        """
        Load the csv file and split it into chunks to be stored in the collection
        it's reference tag is <collection_name>
        """
        from langchain_community.document_loaders import CSVLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        collection = self._select_collection_by_name(collection_name)

        # Load csv
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n"], chunk_overlap=0)
        loader = CSVLoader(file_path=file_path, csv_args=csv_args)
        data = loader.load()
        all_splits = text_splitter.split_documents(data)
        metadatas=[{"type": collection_name}]
        for idx, spl in enumerate(all_splits):
            texts = spl.model_dump()['page_content']
            if not self._check_similarity(texts, collection):
                collection.add_texts(texts=[texts], metadatas=metadatas)


    def load_split_add_text(self, data, collection_name=None, split_on="\n", chunk_size=300):
        """
        Load the text file and split it into chunks to be stored in the collection
        it's reference tag is "docs"
        """

        if not data:
            logger.error("No data provided for loading into vector store.")
            return
        
        collection = self._select_collection_by_name(collection_name)

        # Load text
        metadatas=[{"type": collection_name}]
        if isinstance(data, str):
            all_splits = data.split(split_on)
        else:
            all_splits = data
        for idx, spl in enumerate(all_splits):
            texts = spl
            if not self._check_similarity(texts, collection):
                collection.add_texts(texts=[texts], metadatas=metadatas)

    def load_split_add_textfile(self, file_path, collection_name=None, split_on=["\n"], chunk_size=300):
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        collection = self._select_collection_by_name(collection_name)

        loader = TextLoader(file_path=file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(separators=split_on, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)
        metadatas=[{"type": collection_name}]
        for idx, spl in enumerate(all_splits):
            texts = spl.model_dump()['page_content']
            if not self._check_similarity(texts, collection):
                collection.add_texts(texts=[texts], metadatas=metadatas)


    def retrieve_documents(self, collection_name=None, text="", threshold=0.05, top_k=30)-> str:
        """
            Retrieve the documents from the collection stored in the Vector DB
        """
        logger.info(f"Retrieving relevant document")
        # Read the documents from the collection stored in the Vector DB
        
        collection = self._select_collection_by_name(collection_name)

        if collection:
            retriever = collection.as_retriever(search_type="similarity_score_threshold", 
                                             search_kwargs={"score_threshold": threshold, "k":top_k}
                                             )
            
            resp = retriever.invoke(text)
            context = "\n".join([doc.page_content for doc in resp])
        else:
            context = ""
        logger.info(f"Retrieved documents")        
        return context
    

    def _init_example_selector(self):
        # self.example_selector = SemanticSimilarityExampleSelector(
        #     vectorstore=self.examples    
        # )

        # Create a prompt template for the examples
        self.example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "Question: {question}"),
                ("ai", "SQL: {sql}"),
            ]
                )

    def examples_to_prompt(self, question, top_k=10) -> str:
        """
        This creates the prompt for the fewshot chat message
        """
        
        try:
            self._init_example_selector()

            # relevant_examples = self.example_selector.similarity_search(question, k=top_k)
            relevant_examples = self.examples.similarity_search(question, k=top_k)
            examples_prompt = "## Examples of questions and their corresponding SQL queries relevant for the User's question:\n"
            for ex in relevant_examples:
                # examples_prompt += f"* Validated question: {ex.metadata['question']}: Generated SQL: {ex.metadata['sql']}\n"
                examples_prompt += self.example_prompt.format(question=ex.metadata['question'], sql=ex.metadata['sql']) + "\n"

            return examples_prompt
        except:
            logger.error("Example Selector could not be initialized!")


if __name__ == '__main__':
    v = VectorStore()
    # vv = v._init_db("examples", reset=True)
    #print(vv.get())
    # item = {'question':"what?", 'sql':"Select *"}
    # v.add_to_examples(item)
    from motor import Motor
    m = Motor()
    examples = m.db_chat.fetch_examples(limit=100)
    # print(examples)
    for e in examples:
        item = {'question':e[0], 'sql':e[1]}
        v.add_to_examples(item)
    # v.load_split_add_csv(pp)
    # print(vv.get_by_ids())
    # print(dir(vv))
    # v.load_split_add_text(pp)
    # print(v.examples.get())
    # print(v.docs.get())
    # print(v.docs.similarity_search_with_score("An extensive metagenomic dataset of sewage samples across five European cities: Abundance tables hold the results of the three different analysis pipelines. Abundance tables for the ARG classification and genomic reference-based classification approaches (‘resfinder_gene_abundance’, ‘resfinder_class_abundance’) contain the number of reads aligned to each resistance gene or genome. Abundance table for the high- and medium-quality dereplicated MAG collection called ‘mag_abundance’. This table contains the number of bases aligned to each MAG per sample.", k=1))
    # v.docs = v._init_db("docs")
    # print(v._retrieve_documents(v.docs, "antimicrobial resistant gene", 0.0, top_k=3))
    

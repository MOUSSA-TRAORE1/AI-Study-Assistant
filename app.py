from dotenv import load_dotenv #THIS AND THE LINE BELOW ARE FOR LOADING MY API KEY INTO THE CODE
load_dotenv()

from pypdf import PdfReader   #For reading  text
from langchain_text_splitters import CharacterTextSplitter  #for cutting text( like a cutter:) )

#from langchain_community.vectorstores import Chroma   #DEPRECATED
from langchain_chroma import Chroma
#from langchain_openai import OpenAIEmbeddings   #I'll switch back to this latter, for now i'LL USE HUNGINGFACE EMBEDDINGS
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import os 
from langchain_community.callbacks.manager import get_openai_callback

'''API key for request and receiving (not using this anymore, api key should be protected not used in code like that especially if its gonna be published)
os.environ["OPENAI_API_KEY"]="sk..." '''
# 1.Load pdf
reader=PdfReader("sample.pdf") # here we open the pdf and create a reader object, FILE LOADED INTO READER (no reading is done for the moment)
                            #reader is now an object that represent our pdf and contains pages,metadata, text structures,etc

text=""

for page in reader.pages:               # reader.pages gives all the pages of the document
    text+=page.extract_text()           # extract the text of each page and add it to the text variable
'''
print("----- RAW TEXT -----")
print(text[:500]) #print the first 500 chars
'''

# 2.Split the text into chunks
text_splitter= CharacterTextSplitter(separator="\n",     
                                     chunk_size=500,
                                     chunk_overlap=100)

chunks=text_splitter.split_text(text)  #NOTE TO MYSELF(chumks is an array/list)
'''
print("\n----- CHUNKS -----")
print(f"Total chunks: {len(chunks)}")
print(chunks[0]) 
'''
# 3.Create Embeddings
#embeddings=OpenAIEmbeddings()   #here embeddings is an embedding model object
                                #like a tool that converts text into vectors
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
# 4. Store in ChromaDB
db= Chroma.from_texts(chunks, embeddings) #create a db, takes each chunk, 
                                        #converts it to embeddings and 
                                        # #stores them (both text and embeddings) in the created db
                                        #  db is an object(database instance) and contains all chunks,
                                        # their embeddings and ability to search
#print("Database created succesfully")

''' Not gonna use this, was there just for testing purpose
# 5. Test search
docs = db.similarity_search("What is this document about?") #docs is a list of results, returned by similarity check 

print(docs[0].page_content) '''

# 5. Ask a question
query="What is this document about ?"
docs=db.similarity_search(query)
#Combine retrieved chunks
context= "\n\n".join([doc.page_content for doc in docs])

# 6. Create LLM
llm=ChatOpenAI(model="gpt-4o-mini",
               temperature=0.3)

#7. Ask LLM   (my prompt, which will be given to ai/llm)
with get_openai_callback() as cb:   #this is for tracking tokens used,etc
    response=llm.invoke(f""" Answer the question below:
    Context:{context}
    Question:{query}
    """)

    print(response.content)

    print("\n📊 Usage:")
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")


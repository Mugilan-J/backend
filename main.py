
import os
from dotenv import load_dotenv
from datetime import datetime
import threading
import time
from contextlib import asynccontextmanager
from typing import List

# --- Supabase Integration ---
from supabase import create_client, Client

# --- Chroma Cloud Integration ---
import chromadb
from chromadb.api import ClientAPI

# --- LangChain Core Components ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Helper for listing models from the base library
import google.generativeai as genai

# --- FastAPI for HTTP API ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# --- SETUP ---
load_dotenv()

# --- Database Setup ---
# Define collection names for our cloud memory databases
SHORT_TERM_MEMORY_COLLECTION = "short_term_memory"
LONG_TERM_MEMORY_COLLECTION = "long_term_memory"

# Global variables
db_cleanup_timer = None
agent_executors_by_model = {}
embeddings_global = None
chroma_client_global = None
supabase_client_global = None


# Check for API keys
if not all(os.getenv(key) for key in ["GOOGLE_API_KEY", "TAVILY_API_KEY", "CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE", "SUPABASE_URL", "SUPABASE_KEY"]):
    raise ValueError("GOOGLE_API_KEY, TAVILY_API_KEY, all CHROMA_*, and SUPABASE_URL/SUPABASE_KEY must be set in .env file")

# Configure Gemini client
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error configuring Gemini client: {e}")
    exit()
    
# --- CHROMA CLOUD CONNECTION ---

_client: ClientAPI | None = None

def get_chroma_client() -> ClientAPI:
    """Establishes and returns a singleton connection to the Chroma Cloud database."""
    global _client
    if _client is None:
        print("[System] Initializing Chroma Cloud client...")
        _client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE")
        )
        print("[System] Chroma Cloud client initialized successfully.")
    return _client

# --- SUPABASE DATABASE CONNECTION & FUNCTIONS ---
#
# SCHEMA BASED ON PROVIDED IMAGE:
#
# 1. users table:
#    - id: int4 (Primary Key)
#    - username: text
#
# 2. chats table:
#    - id: int4 (Primary Key)
#    - user_id: int4 (Foreign Key to users.id)
#    - chat_name: text
#
# 3. messages table:
#    - id: int4 (Primary Key)
#    - chat_id: int4 (Foreign Key to chats.id)
#    - user_message: text
#    - ai_response: text
#    - timestamp: timestamptz
#
# NOTE: The current application code for authentication requires additional columns.
# To enable signup, login, and user-specific features, you MUST alter your 'users'
# table in Supabase to include the following columns:
#    - password: text
#    - name: text
#
def get_supabase_client() -> Client:
    """Establishes and returns a singleton connection to the Supabase database."""
    global supabase_client_global
    if supabase_client_global is None:
        print("[System] Initializing Supabase client...")
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        supabase_client_global = create_client(url, key)
        print("[System] Supabase client initialized successfully.")
    return supabase_client_global

def get_or_create_user(username: str, password: str | None = None, name: str | None = None) -> str:
    """Gets a user's ID by username, creating the user if they don't exist."""
    supabase = get_supabase_client()
    response = supabase.table('users').select('id').eq('username', username).execute()
    if response.data:
        return response.data[0]['id']
    else:
        insert_data = {'username': username, 'password': password or ""}
        if name:
            insert_data['name'] = name
        response = supabase.table('users').insert(insert_data).execute()
        return response.data[0]['id']

def get_or_create_chat(user_id: str, chat_name: str) -> str:
    """Gets a chat's ID, creating it if it doesn't exist for the user."""
    supabase = get_supabase_client()
    response = supabase.table('chats').select('id').eq('user_id', user_id).eq('chat_name', chat_name).execute()
    if response.data:
        return response.data[0]['id']
    else:
        response = supabase.table('chats').insert({'user_id': user_id, 'chat_name': chat_name}).execute()
        return response.data[0]['id']

def add_message_to_chat(chat_id: str, user_message: str, ai_response: str):
    """Adds a new user message and AI response to a specific chat."""
    supabase = get_supabase_client()
    supabase.table('messages').insert({
        'chat_id': chat_id,
        'user_message': user_message,
        'ai_response': ai_response
    }).execute()

def get_user_chats(username: str) -> List[str]:
    """Fetches all chat names for a given username."""
    supabase = get_supabase_client()
    user_response = supabase.table('users').select('id').eq('username', username).execute()
    if not user_response.data:
        return []
    user_id = user_response.data[0]['id']
    
    chats_response = supabase.table('chats').select('chat_name').eq('user_id', user_id).execute()
    return [chat['chat_name'] for chat in chats_response.data]

def get_chat_history(username: str, chat_name: str) -> List[dict]:
    """Fetches all messages for a specific chat, ordered by timestamp."""
    supabase = get_supabase_client()
    user_response = supabase.table('users').select('id').eq('username', username).execute()
    if not user_response.data:
        return []
    user_id = user_response.data[0]['id']

    chat_response = supabase.table('chats').select('id').eq('user_id', user_id).eq('chat_name', chat_name).execute()
    if not chat_response.data:
        return []
    chat_id = chat_response.data[0]['id']

    messages_response = supabase.table('messages').select('user_message, ai_response, timestamp').eq('chat_id', chat_id).order('timestamp').execute()
    
    history = [
        {"user_message": row['user_message'], "ai_response": row['ai_response'], "timestamp": row['timestamp']}
        for row in messages_response.data
    ]
    return history

# --- CUSTOM PROMPT ---
CUSTOM_PROMPT_TEMPLATE = """
You are a friendly and highly intelligent AI assistant for {user_name}. Your goal is to provide accurate, conversational, and engaging responses.

**Your Instructions:**
- **Persona:** Speak like a human—natural, friendly, and conversational.
- **Context Awareness:** The current time is {current_time}. Use this for context, but don't state it unless relevant.
- **System Instructions:** {system_prompt}
- **Direct Engagement:** For queries about coding, brainstorming, creative writing, or general conversation, answer directly without using tools. If the `information_seeker` tool indicates the query is conversational, proceed directly to the Final Answer.
- **Tool Usage:** You must choose the correct tool based on the user's query. If a tool provides a direct answer to the user's question (like conversation_history_retriever), you should immediately provide the Final Answer.

**TOOLS:**
You have access to the following tools:

{tools}

To use a tool, use the following format:
```
Thought: Do I need to use a tool? Yes
Action: The action to take, should be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action
```

When you have a response for the user, use the format:
```
Thought: Do I need to use a tool? No
Final Answer: [your final, conversational response here]
```

Begin!

**New User Input:**
{input}

**Your Thought Process:**
{agent_scratchpad}
"""

# --- RAG AND MEMORY FUNCTIONS ---

def initialize_embeddings():
    """Initializes and returns the HuggingFace embedding model."""
    print("[System] Initializing lightweight embedding model (all-MiniLM-L6-v2).")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

def initialize_vectorstore(client, embeddings, collection_name):
    """Initializes and returns a Chroma vector store using a cloud client."""
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    print(f"[System] Vector store initialized for cloud collection '{collection_name}'.")
    return vectorstore

def clear_short_term_memory_db():
    """Clears all documents from the short-term memory collection in Chroma Cloud."""
    global db_cleanup_timer, chroma_client_global
    try:
        print(f"\n[System] Clearing short-term memory ('{SHORT_TERM_MEMORY_COLLECTION}') collection (2-minute cycle)...")
        if chroma_client_global is None:
            print("[System] Chroma client not available for cleanup.")
            return

        collections = chroma_client_global.list_collections()
        if SHORT_TERM_MEMORY_COLLECTION in [c.name for c in collections]:
            collection = chroma_client_global.get_collection(name=SHORT_TERM_MEMORY_COLLECTION)
            existing_items = collection.get()
            if existing_items and existing_items.get("ids"):
                collection.delete(ids=existing_items["ids"])
                print(f"[System] Cleared {len(existing_items['ids'])} items from short-term memory.")
        
    except Exception as e:
        print(f"[System] Error during scheduled DB cleanup: {e}")
    finally:
        db_cleanup_timer = threading.Timer(120, clear_short_term_memory_db)
        db_cleanup_timer.daemon = True
        db_cleanup_timer.start()


def create_smart_information_seeker_tool(vectorstore, llm):
    """
    Creates a single, intelligent tool for information retrieval that decides
    whether to search the web, check memory, or do nothing.
    """
    tavily_search = TavilySearchResults(max_results=5)
    retriever = vectorstore.as_retriever()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def _perform_web_search(query: str) -> str:
        """Helper function to perform a web search and store results in memory."""
        print(f"[Tool Action] Performing web search for: '{query}'")
        results = tavily_search.invoke(query)
        results_str = "\n".join([f"Title: {res.get('title', 'N/A')}\nContent: {res.get('content', 'N/A')}" for res in results])

        if not results_str or "Could not find any relevant results" in results_str:
            return "Web search failed to produce usable content."

        summarization_prompt = f"Based on the following web search results, provide a direct and comprehensive answer to the query: \"{query}\"\n\nResults:\n{results_str}"
        summary_response = llm.invoke(summarization_prompt)
        clean_summary = summary_response.content

        if clean_summary:
            print(f"[System] Storing web search summary in short-term memory: '{clean_summary[:100]}...'")
            documents = text_splitter.create_documents([clean_summary])
            vectorstore.add_documents(documents=documents)
        
        return clean_summary

    def seek_information(query: str) -> str:
        """
        Intelligently seeks information by first using an LLM to decide the best action:
        web search, memory lookup, or answering directly. Falls back to web search if memory is insufficient.
        """
        print(f"[Tool] Seeking information for: '{query}'")

        triage_prompt = f"""
        Given the user's query, what is the best course of action? Choose one of the following and respond with only the action word in brackets:

        1.  **[WEB_SEARCH]**: For real-time or general factual questions.
        2.  **[MEMORY_SEARCH]**: For recalling recent context from this session.
        3.  **[NO_TOOL]**: For conversation, creative writing, or coding.

        User Query: "{query}"
        """
        
        triage_response = llm.invoke(triage_prompt)
        action = triage_response.content.strip()
        print(f"[Tool] Triage result: {action}")

        if "[NO_TOOL]" in action:
            print("[Tool Action] Query is conversational. No tool needed.")
            return "The user's query is conversational, creative, or code-related and does not require a factual search. I should answer it directly."

        if "[MEMORY_SEARCH]" in action:
            print("[Tool Action] Checking short-term memory...")
            retrieved_docs = retriever.invoke(query)
            if retrieved_docs:
                context = "\n".join([doc.page_content for doc in retrieved_docs])
                validation_prompt = f"""Based *only* on the following context, can you provide a direct and sufficient answer to the user's query? Answer "Yes" or "No".\nContext: "{context}"\nQuery: "{query}" """
                validation_response = llm.invoke(validation_prompt)
                if "yes" in validation_response.content.lower():
                    print("[Tool] Relevant information found in short-term memory.")
                    return context
            
            print("[Tool] Information in memory was insufficient. Falling back to web search.")
            return _perform_web_search(query)

        return _perform_web_search(query)

    return Tool(
        name="information_seeker",
        func=seek_information,
        description="Use this for any factual question. It automatically decides whether to search the web, check its short-term memory for information discovered in this session, or handle conversational queries."
    )


def create_conversation_history_tool(vectorstore, llm):
    """Creates a tool that retrieves and summarizes long-term conversation history."""
    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
    
    def retrieve_and_summarize_memory(query: str) -> str:
        collection_name = vectorstore._collection.name
        print(f"[Tool] Fetching history from long-term memory collection: '{collection_name}' for query: '{query}'")
        retrieved_docs = retriever.invoke(query)
        if not retrieved_docs: return "I don't have any specific memory of that."
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        summarization_prompt = f"Based on the conversation history, concisely answer the user's query.\nHistory: \"{context}\"\nQuery: \"{query}\""
        summary_response = llm.invoke(summarization_prompt)
        return summary_response.content

    return Tool(
        name="conversation_history_retriever",
        func=retrieve_and_summarize_memory,
        description="Use this to recall personal details about the user or the history of your past conversations."
    )

# --- AGENT CREATION AND EXECUTION ---

def create_agent_executor(model_name, embeddings, username_for_collections: str, user_name_for_prompt: str, system_prompt: str = ""):
    """Creates the complete LangChain AgentExecutor with per-user collections."""
    global chroma_client_global
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.7, convert_system_message_to_human=True)
    
    # Use a single, session-wide collection for short-term memory.
    # This is not user-specific.
    short_term_vectorstore = initialize_vectorstore(chroma_client_global, embeddings, SHORT_TERM_MEMORY_COLLECTION)
    
    # Create a user-specific collection for long-term memory by appending the username.
    long_term_collection_name = f"{LONG_TERM_MEMORY_COLLECTION}_{username_for_collections}"
    long_term_vectorstore = initialize_vectorstore(chroma_client_global, embeddings, long_term_collection_name)

    tools = [
        create_smart_information_seeker_tool(short_term_vectorstore, llm),
        create_conversation_history_tool(long_term_vectorstore, llm)
    ]
    current_time_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    prompt = PromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE).partial(
        current_time=current_time_str, user_name=user_name_for_prompt, system_prompt=system_prompt
    )
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=15, handle_parsing_errors=True)

# --- FastAPI App Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global embeddings_global, db_cleanup_timer, chroma_client_global
    print("--- Lifespan startup ---")
    get_supabase_client()
    chroma_client_global = get_chroma_client()
    embeddings_global = initialize_embeddings()
    db_cleanup_timer = threading.Timer(120, clear_short_term_memory_db)
    db_cleanup_timer.daemon = True
    db_cleanup_timer.start()
    yield
    # Shutdown logic
    print("--- Lifespan shutdown ---")
    if db_cleanup_timer:
        db_cleanup_timer.cancel()

app = FastAPI(title="Universal Agent Backend (main.py)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://[::1]:8080"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- API Models ---
SUPPORTED_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

class AuthRequest(BaseModel):
    name: str | None = None
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    model: str
    chatId: str | None = None
    userName: str
    chatName: str | None = None
    systemPrompt: str | None = ""

class ChatMessage(BaseModel):
    user_message: str
    ai_response: str
    timestamp: datetime

class ChatHistoryResponse(BaseModel):
    chat_name: str
    history: List[ChatMessage]

class ChatListResponse(BaseModel):
    username: str
    chats: List[str]

# --- API Endpoints ---
@app.get("/api/health")
def api_health():
    return {"status": "ok"}

@app.get("/api/models")
def api_models():
    return {"models": SUPPORTED_MODELS}

@app.post("/api/auth/signup")
def api_auth_signup(req: AuthRequest):
    if not req.name or not req.username or not req.password:
        raise HTTPException(status_code=400, detail="Name, username, and password are required")
    
    supabase = get_supabase_client()
    response = supabase.table('users').select('id').eq('username', req.username).execute()
    if response.data:
        raise HTTPException(status_code=409, detail="Username already exists")

    supabase.table('users').insert({
        'name': req.name,
        'username': req.username,
        'password': req.password
    }).execute()
    return {"status": "created"}

@app.post("/api/auth/login")
def api_auth_login(req: AuthRequest):
    supabase = get_supabase_client()
    response = supabase.table('users').select('password').eq('username', req.username).execute()
    if not response.data or response.data[0]['password'] != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"status": "ok"}

@app.post("/api/auth/delete")
def api_auth_delete(req: AuthRequest):
    supabase = get_supabase_client()
    response = supabase.table('users').select('id, password').eq('username', req.username).execute()
    user = response.data
    if not user or user[0]['password'] != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user_id = user[0]['id']
    # With cascading deletes set up in Supabase, this single delete is enough.
    # The related chats and messages will be deleted automatically.
    supabase.table('users').delete().eq('id', user_id).execute()
    return {"status": "deleted"}

@app.get("/api/chats/{username}", response_model=ChatListResponse)
def api_get_user_chats(username: str):
    """Endpoint to retrieve all chat names for a specific user."""
    chats = get_user_chats(username)
    return {"username": username, "chats": chats}

@app.get("/api/chats/{username}/{chatname}", response_model=ChatHistoryResponse)
def api_get_chat_history(username: str, chatname: str):
    """Endpoint to retrieve the message history of a specific chat."""
    history_records = get_chat_history(username, chatname)
    return {"chat_name": chatname, "history": history_records}

@app.post("/api/chat")
def api_chat(req: ChatRequest):
    if req.model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model")

    try:
        settings_key = f"{req.model}_{req.userName}_{req.systemPrompt}"
        if settings_key not in agent_executors_by_model:
            supabase = get_supabase_client()
            response = supabase.table('users').select('name').eq('username', req.userName).execute()
            display_name = req.userName
            if response.data and response.data[0].get('name'):
                display_name = response.data[0]['name']
            
            agent_executors_by_model[settings_key] = create_agent_executor(
                req.model, embeddings_global, req.userName, display_name, req.systemPrompt or ""
            )
        executor = agent_executors_by_model[settings_key]
        
        result = executor.invoke({"input": req.message})
        ai_response = result.get("output", "Error: No output from agent.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

    try:
        user_id = get_or_create_user(req.userName)
        chat_name = req.chatName or " ".join(req.message.split()[:3]) or "New Chat"
        chat_id = get_or_create_chat(user_id, chat_name)
        add_message_to_chat(chat_id, req.message, ai_response)
        
        # Save interaction to the user-specific long-term memory collection.
        user_long_term_collection = f"{LONG_TERM_MEMORY_COLLECTION}_{req.userName}"
        user_long_term_vs = initialize_vectorstore(chroma_client_global, embeddings_global, user_long_term_collection)
        history_to_save = f"In chat '{chat_name}', user asked: {req.message}\nAI responded: {ai_response}"
        user_long_term_vs.add_texts([history_to_save])
    except Exception as e:
        print(f"[System] Error saving chat history: {e}")

    return {
        "chatId": req.chatId or "",
        "model": req.model,
        "message": req.message,
        "reply": ai_response,
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)


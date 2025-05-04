import streamlit as st
import requests
import random
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings

# --- Load API Keys from Streamlit Secrets ---
deepseek_key = st.secrets["API_KEYS"]["deep_seek_key"]
openweather_key = st.secrets["API_KEYS"]["openweather_key"]

# --- Load and Process PDF ---
@st.cache_resource
def load_pdf_and_create_vectorstore():
    loader = PyPDFLoader("datapdf1.pdf")
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=10,
        length_function=len
    )
    docs = text_splitter.split_documents(pages)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )
    return vectordb, docs

try:
    new_db, processed_docs = load_pdf_and_create_vectorstore()
except Exception as e:
    st.error(f"Error loading PDF: {e}")
    st.stop()

# --- Language Model ---
llm_deepseek = ChatOpenAI(
    model='deepseek-chat',
    base_url="https://api.deepseek.com",
    api_key=deepseek_key,
    temperature=0
)

# --- Weather Function ---
def get_current_weather(destination, api_key):
    try:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={destination}&limit=1&appid={api_key}"
        geo_response = requests.get(geo_url).json()
        if not geo_response:
            return None, None, "⚠️ Couldn't find location"
        lat = geo_response[0]['lat']
        lon = geo_response[0]['lon']
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        weather_response = requests.get(weather_url).json()
        if 'main' in weather_response:
            temp = weather_response['main']['temp']
            desc = weather_response['weather'][0]['description'].capitalize()
            return temp, desc, f"🌡️ {destination}: {temp}°C, {desc}"
        return None, None, "⚠️ Weather info unavailable"
    except Exception:
        return None, None, "⚠️ Error fetching weather"

# --- Core Functions ---
def get_packing_suggestions(temp, desc):
    prompt = f"List 4-5 essential items for {temp}°C weather ({desc}) with emojis:"
    return llm_deepseek.invoke(prompt).content

def get_itinerary(selected, days):
    prompt = f"Create a {days}-day itinerary for {selected} with daily highlights:"
    return llm_deepseek.invoke(prompt).content

destination_prompt_template = r"""
You are IntelliTrip travel assistant. Use context below to suggest destinations:

* City, Country format
* 2 reasons to visit
* Best time to visit
* Estimated budget

Format:
**City, Country**
* Reason 1
* Reason 2
* 📅 Best time: <>
* 💰 Budget: $<>

Ask: 🌟 Select one that suits you

Context: ```{context}```
Question: {question}
"""

def get_destination_suggestions(query, excluded=None):
    context = "\n".join([doc.page_content for doc in processed_docs])
    excluded = excluded or []
    exclusion_note = f"\nAvoid: {', '.join(excluded)}\n" if excluded else ""
    return llm_deepseek.invoke(exclusion_note + destination_prompt_template.format(
        context=context, question=query
    )).content

# --- Streamlit App ---
st.set_page_config(page_title="IntelliTrip Travel Bot", page_icon="🌍")
st.title("🌍 IntelliTrip - Personal Travel Consultant")

# Session State Initialization
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        'selected': None,
        'suggestions': [],
        'options_round': 0,
        'user_query': None
    }

user_input = st.text_input("How can I help plan your trip? (Type & Enter)")

if user_input:
    # Initialize new search
    if not st.session_state.app_state['user_query']:
        st.session_state.app_state = {
            'selected': None,
            'suggestions': [],
            'options_round': 0,
            'user_query': user_input
        }
    
    state = st.session_state.app_state
    
    if not state['selected']:
        # Show destination suggestions
        suggestions = get_destination_suggestions(
            state['user_query'], 
            excluded=state['suggestions']
        )
        st.markdown(suggestions)
        
        # Extract city options
        city_options = [line.strip().replace("**", "") 
                       for line in suggestions.splitlines() 
                       if "**" in line]
        state['suggestions'].extend(city_options)
        
        # User input handling
        choice = st.text_input(
            "✍️ Choose destination (or 'More options'/'Exit'):",
            key=f"choice_{state['options_round']}"
        )
        
        if choice:
            if choice.lower() == 'exit':
                st.success("👋 Safe travels!")
                st.stop()
            elif choice.lower() == 'more options':
                state['options_round'] += 1
                st.experimental_rerun()
            elif choice in city_options:
                state['selected'] = choice
                st.experimental_rerun()
            else:
                st.warning("⚠️ Please select a valid option")
    
    # Show trip details after selection
    if state['selected']:
        st.success(f"🎯 Planning trip for {state['selected']}...")
        
        days = st.number_input("📆 Trip duration (days):", 1, 30, 3)
        
        st.subheader("📅 Itinerary")
        st.markdown(get_itinerary(state['selected'], days))
        
        st.subheader("🧭 Local Cuisine")
        st.markdown(llm_deepseek.invoke(
            f"Suggest 2-3 must-try foods in {state['selected']}:"
        ).content)
        
        temp, desc, weather = get_current_weather(state['selected'], openweather_key)
        st.subheader("🌦️ Weather")
        st.write(weather)
        
        if temp and desc:
            st.subheader("🎒 Packing List")
            st.markdown(get_packing_suggestions(temp, desc))
        
        st.balloons()
        st.info(random.choice([
            "✨ Need more help? Ask away!",
            "🧳 Ready for your next adventure!",
            "🌟 Happy travels!"
        ]))
        
        if st.button("Start New Trip"):
            st.session_state.app_state = {
                'selected': None,
                'suggestions': [],
                'options_round': 0,
                'user_query': None
            }
            st.experimental_rerun()
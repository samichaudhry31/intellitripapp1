import streamlit as st
import requests
import random
import time
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize all session state variables
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        'selected': None,
        'suggested_options': [],
        'options_round': 0,
        'chat_history': [],
        'current_query': None,
        'days': 3
    }

# Must be first Streamlit command
st.set_page_config(
    page_title="IntelliTrip Travel Bot", 
    page_icon="🌍",
    layout="wide"
)

# --- Load API Keys from Streamlit Secrets ---
deepseek_key = st.secrets["API_KEYS"]["deep_seek_key"]
openweather_key = st.secrets["API_KEYS"]["openweather_key"]

# --- Load and Process PDF ---
@st.cache_resource
def load_pdf_and_create_vectorstore():
    try:
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
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return None, None

new_db, processed_docs = load_pdf_and_create_vectorstore()
if new_db is None:
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
            return None, None, "⚠️ Couldn't find the location for weather info."
        
        lat = geo_response[0]['lat']
        lon = geo_response[0]['lon']
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        weather_response = requests.get(weather_url).json()
        
        if 'main' in weather_response:
            temp = weather_response['main']['temp']
            desc = weather_response['weather'][0]['description'].capitalize()
            weather_text = f"🌡️ {destination}: {temp}°C, {desc}."
            return temp, desc, weather_text
        return None, None, "⚠️ Weather info not available."
    except Exception:
        return None, None, "⚠️ Error fetching weather data."

# --- Packing Suggestions ---
def get_packing_suggestions(temp, desc):
    prompt = f"""List 4-5 essential things a traveler should pack for {temp}°C weather described as "{desc}". 
    Keep it clean and simple with emojis."""
    try:
        response = llm_deepseek.invoke(prompt)
        return response.content
    except Exception:
        return "Couldn't generate packing suggestions at this time."

# --- Day-wise Itinerary ---
def get_itinerary(selected, days):
    prompt = f"""Suggest a simple {days}-day itinerary for {selected}. 
    For each day, give one main activity or highlight only. 
    Use clean bullet points."""
    try:
        response = llm_deepseek.invoke(prompt)
        return response.content
    except Exception:
        return "Couldn't generate itinerary at this time."

# --- Destination Suggestions from PDF ---
destination_prompt_template = r"""
You are IntelliTrip, a witty and helpful travel assistant. 
ONLY use the context provided inside the triple backticks to suggest destinations.
DO NOT guess or suggest any place outside the context. Each destination must include:

* **City, Country** format
* Two short, compelling reasons why it's worth visiting
* Best time to visit (if known)
* Estimated budget in USD for a moderate traveler (include flights, hotel, and food)

Show EXACTLY two options using this format:
**City, Country**
* Line 1
* Line 2
* Line 3
* 📅 Best time to visit: <value from data>
* 💰 Estimated budget: $<amount> USD

Then ask:
🌟 Please select the one that best suits you.

--- `{context}` ---
Traveler's Question: {question}
"""

def get_destination_suggestions(user_query, excluded_destinations=None, docs=processed_docs):
    try:
        context_text = "\n".join([doc.page_content for doc in docs])
        if excluded_destinations:
            exclusions = "\n".join([f"- {item}" for item in excluded_destinations])
            exclusion_note = f"\nAvoid repeating these destinations:\n{exclusions}\n"
        else:
            exclusion_note = ""
            
        final_prompt = destination_prompt_template.format(
            context=context_text, 
            question=user_query
        )
        full_prompt = exclusion_note + final_prompt
        response = llm_deepseek.invoke(full_prompt)
        return response.content
    except Exception:
        return "Couldn't generate destination suggestions at this time."

# --- Food Recommendations ---
def get_food_recommendations(selected):
    prompt = f"""Give 2-3 simple food recommendations or top dishes to try in {selected}. 
    Keep the response clean and straightforward."""
    try:
        response = llm_deepseek.invoke(prompt)
        return response.content
    except Exception:
        return "Couldn't generate food recommendations at this time."

# --- Ending Messages ---
ending_messages = [
    "✨ Let me know what else I can do for you! ✈️😎",
    "🧳 Ready when you are for your next destination!",
    "🌟 I'm here to make your journey smoother — what's next?"
]

# --- Streamlit App UI ---
st.title("🌍 Welcome to IntelliTrip - Your Personal Travel Consultant!")
st.write("💬 Mention your *budget* and *interest* for better recommendations.")

user_message = st.text_input("How can I help you plan your trip today? (Type and press Enter)")

if user_message:
    # Reset state if new query
    if st.session_state.app_state['current_query'] != user_message:
        st.session_state.app_state = {
            'selected': None,
            'suggested_options': [],
            'options_round': 0,
            'chat_history': [{"role": "user", "message": user_message}],
            'current_query': user_message,
            'days': 3
        }
    else:
        st.session_state.app_state['chat_history'].append({"role": "user", "message": user_message})

    state = st.session_state.app_state

    if state['selected'] is None:
        suggestions = get_destination_suggestions(
            user_message, 
            excluded_destinations=state['suggested_options'],
            docs=processed_docs
        )
        st.markdown(suggestions)

        city_options = [line.strip().replace("**", "") for line in suggestions.splitlines() if "**" in line]
        state['suggested_options'].extend(city_options)

        selected_input = st.text_input(
            "✍️ Type your pick (or type 'More options' or 'Exit'):", 
            key=f"dest_input_{state['options_round']}"
        )

        if selected_input:
            if selected_input.lower() == 'exit':
                st.success("👋 Safe travels! Thanks for using IntelliTrip.")
                st.stop()
            elif selected_input.lower() == 'more options':
                state['options_round'] += 1
                st.experimental_rerun()
            elif selected_input in city_options:
                state['selected'] = selected_input
                st.experimental_rerun()
            else:
                st.warning("⚠️ Please type a valid option from the list above, or type 'More options' or 'Exit'.")
    else:
        st.success(f"🎯 Fabulous choice! Let me tailor plans for **{state['selected']}**...")
        
        state['days'] = st.number_input(
            "📆 How many days will you be enjoying there?", 
            min_value=1, 
            step=1, 
            value=state['days']
        )

        st.subheader("📅 Itinerary")
        itinerary = get_itinerary(state['selected'], state['days'])
        st.markdown(itinerary)

        st.subheader("🧭 Local Flavors")
        food_tips = get_food_recommendations(state['selected'])
        st.markdown(food_tips)

        temp, desc, weather_report = get_current_weather(state['selected'], openweather_key)
        st.subheader("🌦️ Weather Report")
        st.write(weather_report)

        if temp is not None and desc is not None:
            st.subheader("🎒 Packing Suggestions")
            packing_suggestions = get_packing_suggestions(temp, desc)
            st.markdown(packing_suggestions)

        st.balloons()
        st.info(random.choice(ending_messages))
        
        if st.button("Start New Search"):
            st.session_state.app_state = {
                'selected': None,
                'suggested_options': [],
                'options_round': 0,
                'chat_history': [],
                'current_query': None,
                'days': 3
            }
            st.experimental_rerun()
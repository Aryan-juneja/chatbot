import openai
import time
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from langchain.schema import  HumanMessage
from utils import call_openai_format,pinecone_search
from dotenv import load_dotenv
app = Flask(__name__)
CORS(app)


load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
import os
import json5
import json
from langchain.tools import TavilySearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate




# Set API keys
os.environ["TAVILY_API_KEY"] = os.getenv('OPENAI_API_KEY') 

# Initialize Tavily Web Search Tool
tavily_search = TavilySearchResults(max_results=3)


def fetch_real_estate_results(query):
    search_tool = TavilySearchResults()
    return search_tool.run(query)

# Step 2: Use GPT-4 to extract structured details
def parse_property_details(raw_text):
    llm = ChatOpenAI(model="gpt-4")

    prompt = f"""
    Extract property details from the following real estate search results.
    If any field is missing, **fill it with expected or common values**.
    **Return output as a valid JSON array ONLY—no extra text.**

    **Format:**
    [
        {{
            "name": "Property Name",
            "location": "City, Sector",
            "area (sqft)": 1500,
            "price (INR)": "₹1.85 Cr",
            "facing_direction": "North-East",
            "construction_status": "Ready to Move",
            "furnishing_type": "Semi-Furnished",
            "description": "Short summary of the property.",
            "amenities": ["Gym", "Swimming Pool", "Parking"],
            "images": ["https://example.com/property.jpg"]
        }}
    ]

    **Search Results:**  
    {raw_text}
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        return json.loads(response.content.strip())  
  # Ensure JSON parsing
    except json.JSONDecodeError as e:
        print("JSON Parsing Error:", e)
        print("Raw Output:", response.text)
        return {"error": "Failed to parse JSON"}
    

def format_chat_history(chathistory):
    """Ensure system prompt is added only once and in correct format."""
    system_prompt = {
        "role": "system",
        "content": """
        You are an engaging Property Chatbot focused on helping users find properties in India.

        ## **Response Format (STRICTLY FOLLOW THIS)**
        Chatbot: <Acknowledge user’s choice with enthusiasm>  
        Follow-Up Question: <Ask the next relevant question>  
        Options:  
        <Option 1>  
        <Option 2>  
        <Option 3>  
        <Option 4>  

        ## **Rules**
        1. Always include exactly four options.
        2. Do NOT repeat "Follow-Up Question" inside the response message.
        3. Options must be listed under "Options:".
        4. After 5-6 questions, ask:  
            _"Would you like to see property options now, or continue refining your search?"_  
            Options:  
            - See property options now  
            - Refine search further  
            - Add more preferences  
            - Start over  
        """
    }

    # Only add system message if it's not already there
    if not chathistory or chathistory[0]["role"] != "system":
        chathistory.insert(0, system_prompt)

    return chathistory


def stream_openai_response( chat_history, delay=0.05):
    """Generator function to dynamically stream chatbot responses with delay."""
    

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=chat_history,
        temperature=0.7,
        stream=True
    )

    chatbot_message = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            text_part = chunk.choices[0].delta.content
            chatbot_message += text_part
            yield text_part
            time.sleep(delay)

    chat_history.append({"role": "assistant", "content": chatbot_message})


def understand_chat_sentiment2(chat_history):
    prompt = ChatPromptTemplate.from_template("""
    Analyze the following chat history to determine if sufficient information has been gathered to search for properties in Pinecone.

    Rules:
    - If the user explicitly requests to see properties (e.g., "show me properties", "list properties", "I want to see properties", "give me property options", or similar phrases), return 'True'.
    - If the chatbot explicitly asks **"Do you want to see properties?"**, and the user **confirms with an affirmative response** (e.g., "Yes", "Sure", "Okay", "Proceed", "Go ahead", or selecting an affirmative option), return 'True'.
    - If the user's response is unclear, neutral, or not an explicit confirmation, return 'False'.
    

    **Important:** Do not assume the user wants to see properties unless they explicitly ask for it or confirm when asked.

    Return only 'True' or 'False'—nothing else.

    Chat History:
    {chat_history}
    """)

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    chain = prompt | llm
    response = chain.invoke({
        "chat_history": chat_history,  
    })

    print("Sentiment Analysis Response:", response.content)
    return response.content.strip() == "True"




def update_chat_history(chat_history, new_prompt):
    return [
        {**msg, "content": new_prompt.strip()} if msg["role"] == "system" else msg
        for msg in chat_history
    ]


def checkLocation(query):
    prompt = ChatPromptTemplate.from_template("""
    You are a strict property location checker. 
    Determine if the given query refers to Gurgaon or nearby areas.
    
    Consider the following areas as **nearby Gurgaon**:
    - Gurgaon
    - Cyber City
    - Golf Course Road
    - MG Road
    - Sohna Road
    - Manesar
    - Sector 29, Sector 56, Sector 49, etc.
    - Udyog Vihar

    If the location in the query **matches or is near** these places, return only 'True'.
    Otherwise, return only 'False'. Do not include any explanation.

    Query:
    {query}
    """)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = prompt | llm
    response = chain.invoke({"query": query})

    # Normalize response (strip spaces & lowercase)
    result = response.content.strip().lower()
    print("response:", result)  # Debugging

    return result == "true"

@app.route("/chat", methods=["POST"])
def property_chatbot2():
    data = request.get_json()
    chathistory = data.get("chat_history", [])
    
    print(chathistory)
    chat_history = format_chat_history(chathistory)
    answer = understand_chat_sentiment2(chat_history)
    print("chat Sentiment Analysis foe property show:", answer)

    if answer:
        formatted_query = call_openai_format(chat_history)
        print(formatted_query)
        gurgaon_keywords = ["gurgaon", "gurg", "sector", "golf course", "sohna road", "dlf", "mg road", "cyber city"]
        is_gurgaon = any(keyword in formatted_query.lower() for keyword in gurgaon_keywords)
        # answer2 = checkLocation(formatted_query)
        print("Location Check:", is_gurgaon)

        if str(is_gurgaon).strip().lower() == "true":
            base_dir = os.path.dirname(__file__)
            file_path = os.path.join(base_dir, "prompts2.txt")

            with open(file_path, "r", encoding="utf-8") as f:
                prompt = f.read()
            chat_history = update_chat_history(chat_history, prompt)
            search_results = pinecone_search(formatted_query)

            if not search_results:
                return jsonify({"bot_reply": "No relevant search results found.", "summarize_bot_reply": ""})

            chat_history[-1]["content"] += f" {prompt} Use this info: {search_results}."

            def generate_stream():
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=chat_history,
                    temperature=0.7,
                    stream=True
                )
                chatbot_message = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        text_part = chunk.choices[0].delta.content
                        chatbot_message += text_part
                        yield text_part
                        time.sleep(0.05)

                chat_history.append({"role": "assistant", "content": chatbot_message})

            return Response(generate_stream(), content_type="text/event-stream")

        else:
            try:
                raw_results = fetch_real_estate_results(formatted_query)
                structured_data = parse_property_details(raw_results)
                print(structured_data)

                def generate_structured_data_stream(data):
                    for i in range(0, len(data), 100):  # Stream in 100-char chunks
                        yield data[i : i + 100]
                        time.sleep(0.05)

                return Response(generate_structured_data_stream(structured_data), content_type="text/event-stream")
            except Exception as e:
                print("Error fetching/parsing data:", str(e))
                return jsonify({"bot_reply": "An error occurred while fetching property details."})
    
    else:
        return Response(stream_openai_response( chat_history), content_type="text/event-stream")



if __name__ == "__main__":
    app.run(debug=True, port=5000)

import os
import openai
import logging
from openai import OpenAI
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
load_dotenv()
# === Hardcoded API Keys and MongoDB Credentials ===
try:
    openai.api_key = os.getenv('OPENAI_API_KEY')
    tavily_api_key = os.getenv('TAVILY_API_KEY')
    host=os.getenv('HOST')
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
    INDEX_NAME = "finalpropertylisting7"
    pinecone_client = Pinecone(api_key="pcsk_CKCnj_5U5uC4tiS3fna7EDVAp6dCKqJ9sUeayeMyK4c12QyU54uwyq1GBFmX2oT5FFoEb")
    if INDEX_NAME not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=INDEX_NAME,
            dimension=1536,  # Adjust this to the dimensionality of your vectors
            metric='cosine',  # Use the appropriate metric (cosine, euclidean, etc.)
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'  # Adjust the region accordingly
            )
        )
    index = pinecone_client.Index(INDEX_NAME)
    client=OpenAI()
except Exception as e:
    logger.error("Error initializing APIs or MongoDB: %s", e)
    exit(1)

def create_connection():
    """Create a database connection."""
    try:
        connection = mysql.connector.connect(
            host="rds-data-test.cpuycomgmc4o.us-east-2.rds.amazonaws.com",  # Ensure this is set to your RDS endpoint
            user="admin",
            password="RightHomeUK",
            database="User_Info",
            port=3306  # Default MySQL port
        )
        logger.info("Database connection successful")
        return connection
    except mysql.connector.Error as e:
        logger.error("Database connection error: %s", e)
        raise


def get_embeddings(query):
    """
    Generate vector embeddings for the user query using OpenAI's text-embedding-ada-002 model.
    """
    try:
        response =client.embeddings.create(
        model="text-embedding-ada-002",
        input=query,
        encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None


def extract_metadata_from_pinecone(match):
    """
    Extracts and formats metadata from a Pinecone match.
    """
    metadata = match['metadata']

    # Extract relevant fields from the metadata
    formatted_metadata = f"""
    ID:{metadata.get('ID','No id available')}
    Area:{metadata.get('Area','No area available')}
    Property Name: {metadata.get('Name', 'No name available')}
    Price: {metadata.get('Price', 'No price available')}
    Location: {metadata.get('Location', 'No location available')}
    Overview: {metadata.get('Overview','No overview available')}
    Amenities: {metadata.get('Amenities', 'No amenities available')}
    Description: {metadata.get('Description', 'No description available')}
    Images: {metadata.get('Images', 'No images available')}
    Builder Data: {metadata.get('Metadata', 'No builder data available')}
    Specific Builder Name: {metadata.get('Specific Builder Name', 'No specific builder name available')}
    Status:{metadata.get('Status','No status available')}
    RERA ID:{metadata.get('RERA Details','No rera details available')}
    # Rera id:{metadata.get('RERA ID','No rera id available')}
    """

    return formatted_metadata

def pinecone_search(query):
    """
    Query Pinecone for search results.
    """
    try:
        embedding = get_embeddings(query)
        if not embedding:
            return []
        # print(embedding)
        response = index.query(vector=embedding, top_k=3, include_metadata=True)
        
        results = []
        for match in response['matches']:
            formatted_metadata = extract_metadata_from_pinecone(match)
            results.append({
                "content": formatted_metadata,
                "image_link": match['metadata'].get('Images', ['No image available'])[0]
            })
        return results
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []

def summarize_pinecone_results(results):
    # Implement your logic to summarize the results
    prompt = ChatPromptTemplate.from_template("""
    You have to read the following  search results and summarize them in a concise manner .
                                                {pinecone_searched_documents}
    """)
    llm=ChatOpenAI(temperature=0.9,model_name="gpt-3.5-turbo")
    chain = prompt | llm
    response = chain.invoke({
        "pinecone_searched_documents": results
    })
    return response.content

def checkIsHistoryConsideredToBeSentForSentimentAnalysis(chat_history):
    prompt = ChatPromptTemplate.from_template("""
    Analyze the following chat history to determine if sufficient information has been gathered like location , price , amenities , or user requirements to send chat history for sentiment analysis. 
    Based on this analysis, 
                Either return 'True' if sufficient information has been gathered; otherwise, return 'False'.  
                Return only 'True' or 'False' without any additional text.
    Chat History:
    {chat_history}
    """)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    chain = prompt | llm
    response = chain.invoke({"chat_history": chat_history})
    
    # Normalize response to avoid space or newline issues
    result = response.content.strip().lower()  # Normalize response
    
    print("Response:", result)
    
    return result == "true"


def understand_chat_sentiment(chat_history):
    prompt = ChatPromptTemplate.from_template("""
    Analyze the following chat history to determine if sufficient information has been gathered to search for properties in Pinecone. Consider the following criteria:
    - The user explicitly requests to see properties (e.g., "show me property").
    - You have asked the user if they want to see properties, and the user has responded affirmatively.
    Make one thing in mind that is if location is beyond the gurgoan then never say true in any case
    Break down the chat history step by step, considering the user's intent and the information exchanged.
    Based on this analysis, 
                Either return 'True' if sufficient information has been gathered; otherwise, return 'False'  
                But dont return anything else

    Chat History:
    {chat_history}
    
    """)
    llm= ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    chain = prompt | llm
    response = chain.invoke({"chat_history": chat_history})
    print("response",response.content)
    return response.content == "True"



def call_openai_format(messages, model="gpt-4", max_tokens=100, temperature=0):
    """
    Calls OpenAI's chat completions API to process the conversation.

    Args:
        messages (list): The conversation history to pass to OpenAI.
        model (str): The OpenAI model to use. Default is "gpt-4".
        max_tokens (int): Maximum tokens to generate. Default is 100.
        temperature (float): Sampling temperature for randomness. Default is 0.

    Returns:
        str: The assistant's response or an error message.
    """
    try:
        # Prepare conversation payload
        messages.insert(0, {"role": "system", "content": """ 
The task is to read whole chat history and based on the user's preferences, retrieve relevant properties from Pinecone. Focus on these key details from the user's preferences:
"""})
  
        

        # Call OpenAI API
        response =openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
        )

        # Extract and return the assistant's response
        return response.choices[0].message.content

    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        return "An unexpected error occurred while calling OpenAI."




def extract_property_details_via_llm(chat_history):
    # Convert chat history into a readable format for LLM
    formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])

    prompt = f"""
Extract all relevant details from the given chat history that can be useful for financial analysis, including but not limited to:

- Property Type (Residential or Commercial)  
- Budget (e.g., ₹1Cr-₹2Cr)  
- Location (City or locality)  
- Property Status (Under Construction or Ready to Move)  
- Preferred Amenities (e.g., Parking, Power Backup, 24x7 Security)  
- Connectivity (e.g., Metro station distance, Highways, Airport proximity)  
- Nearby Facilities (e.g., Schools, Hospitals, Malls, Markets)  
- Security Features (e.g., Gated Community, CCTV, Security Personnel)  
- Sustainability Features (e.g., Solar Panels, Rainwater Harvesting)  
- Expected Rental Yield (if mentioned)  
- Investment Horizon (e.g., Short-term or Long-term)  

Provide the output in a structured JSON format, **only including the details found in the chat history** (do not assume or generate missing details):

```json
{{
    "property_type": "Residential/Commercial",
    "budget": "₹XCr-₹YCr",
    "location": "Extracted Location",
    "property_status": "Under Construction/Ready to Move",
    "preferred_amenities": ["Amenity 1", "Amenity 2"],
    "connectivity": {{
        "metro_station": "X km",
        "highway": "Highway Name - Y km",
        "airport": "Z km"
    }},
    "nearby_facilities": {{
        "schools": ["School 1 - X km", "School 2 - Y km"],
        "hospitals": ["Hospital 1 - X km", "Hospital 2 - Y km"],
        "malls": ["Mall 1 - X km"]
    }},
    "security_features": {{
        "gated_community": true/false,
        "cctv_coverage": true/false,
        "security_personnel": "24x7/Daytime/None"
    }},
    "sustainability_features": ["Solar Panels", "Rainwater Harvesting"],
    "expected_rental_yield": "X%",
    "investment_horizon": "Short-term/Long-term"
}}


    Chat History:
    {formatted_history}

    Now, extract the details and return only the JSON response.
    """

    response =openai.chat.completions.create(
        model="gpt-4",  # Use gpt-3.5-turbo if needed
        messages=[{"role": "system", "content": "You are an AI that extracts structured property details from chat history."},
                  {"role": "user", "content": prompt}],
        temperature=0
    )

    extracted_data = response.choices[0].message.content
    return extracted_data






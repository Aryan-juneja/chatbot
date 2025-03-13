import re
import os
import openai
import logging
import io
import csv
from openai import OpenAI
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
import mysql.connector
import boto3
from datetime import datetime
from boto3.dynamodb.conditions import Key
# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from twilio.rest import Client
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
load_dotenv()
# === Hardcoded API Keys and MongoDB Credentials ===
try:
    openai.api_key = os.getenv('OPENAI_API_KEY')
    tavily_api_key = os.getenv('TAVILY_API_KEY')
    KENDRA_INDEX_ID = os.getenv('KENDRA_INDEX_ID')
    AWS_REGION = os.getenv('AWS_REGION')
    ROLE_ARN = os.getenv('ROLE_ARN')
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    DYNAMODB_TABLE_NAME = "UserConversations"
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "YourAccessKey")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "YourSecretKey")
    host=os.getenv('HOST')
    database=os.getenv('DATABASE')
    user=os.getenv('USER')
    password=os.getenv('PASSWORD')
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
    INDEX_NAME = "finalpropertylisting5"
    pinecone_client = Pinecone(api_key="pcsk_CKCnj_5U5uC4tiS3fna7EDVAp6dCKqJ9sUeayeMyK4c12QyU54uwyq1GBFmX2oT5FFoEb")
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    VERIFY_SERVICE_SID = os.getenv("VERIFY_SERVICE_SID")
    S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'chatbot-conversations-righthome')
    s3_client = boto3.client('s3', region_name='us-east-2')
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
    dynamodb_client = boto3.resource(
    'dynamodb',
    region_name='ap-south-1',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    try:
        dynamodb_table = dynamodb_client.Table(DYNAMODB_TABLE_NAME)
        print(f"Connected to DynamoDB table: {DYNAMODB_TABLE_NAME}")
    except Exception as e:
        print(f"Error connecting to DynamoDB: {e}")
        exit()
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

# def check_sms_verification(phone_number, code):
#     try:
#         client = Client("AC2d2d983c97d5413b5aade0a685d37192", "59e028bae9032c86f1ad9fddad133245 ")
#         verification_check = client.verify.services("VAd690c00de71c8238570b8841bfef41b8").verification_checks.create(
#             to=phone_number,
#             code=code
#         )
#         print(f"Verification check status: {verification_check.status}")
#         return verification_check.status == "approved"
#     except Exception as e:
#         print(f"Error verifying code: {e}")
#         return False
# def send_sms_verification(phone_number):
#     try:
#         client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
#         verification = client.verify.services(VERIFY_SERVICE_SID).verifications.create(
#             to=phone_number,
#             channel="sms"
#         )
#         print(f"Verification SMS sent to {phone_number}. Status: {verification.status}")
#         return verification.status
#     except Exception as e:
#         print(f"Error sending SMS: {e}")
#         return None

def save_conversation_to_dynamodb(user_id, user_query, assistant_response):
    """
    Save the conversation to DynamoDB with a timestamp.
    """
    try:
        dynamodb_table.put_item(
            Item={
                "UserID": user_id,
                "Timestamp": datetime.utcnow().isoformat(),
                "UserQuery": user_query,
                "AssistantResponse": assistant_response
            }
        )
        print("Conversation saved successfully!")
    except Exception as e:
        print(f"Error saving to DynamoDB: {e}")



def retrieve_conversation_from_dynamodb(user_id):
    """
    Retrieve conversation history for a specific user ID from DynamoDB.
    """
    try:
        # Query the DynamoDB table to get all conversations for the given user ID
        response = dynamodb_table.query(
            KeyConditionExpression=Key('UserID').eq(user_id)
        )

        # Extract items from the response
        items = response.get('Items', [])
        if not items:
            print(f"No conversation history found for User ID: {user_id}")
            return None

        # Print all retrieved conversations
        for item in items:
            print(f"\nTimestamp: {item['Timestamp']}")
            print(f"User Query: {item['UserQuery']}")
            print(f"Assistant Response: {item['AssistantResponse']}")
        return items

    except Exception as e:
        print(f"Error retrieving from DynamoDB: {e}")
        return None

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

def chat_answer(messages, buffer_memory):
    """Generate chatbot response."""
    try:


        query = messages[0]["content"]

        if understand_chat_sentiment(messages):
            base_dir = os.path.dirname(__file__)
            file_path = os.path.join(base_dir, "prompts2.txt")

        # Read the prompt
            with open(file_path, "r", encoding="utf-8") as f:
                prompt = f.read()
            formatted_query = call_openai_format(messages)

            search_results = pinecone_search(formatted_query)
            if not search_results:
                return {"bot_reply": "No relevant search results found.", "summarize_bot_reply": ""}

            # Update last message with prompt and search results
            messages[-1]["content"] += f" {prompt} Use this info: {search_results}."

            # OpenAI Chat Completion
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0
            )

            bot_reply = response.choices[0].message.content.strip()
            summarize_bot_reply = summarize_pinecone_results(bot_reply)
            
            return {"bot_reply": bot_reply, "summarize_bot_reply": summarize_bot_reply}

        else:
            base_dir = os.path.dirname(__file__)
            file_path = os.path.join(base_dir, "prompts.txt")

        # Read the prompt
            with open(file_path, "r", encoding="utf-8") as f:
                prompt = f.read()

            messages[-1]["content"] += f" {prompt}"

            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0
            )

            bot_reply = response.choices[0].message.content.strip()

            return {"bot_reply": bot_reply, "summarize_bot_reply": ""}

    except Exception as e:
        logger.exception("Error in chat_answer")
        return {"bot_reply": "An error occurred while generating the response.", "summarize_bot_reply": ""}

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



def sufficient_information_gathered(chat_history):
    """
    Analyzes the chat history to determine if sufficient information has been gathered
    to search for properties. Returns True if conditions are met; otherwise, returns False.
    """
    print("chat history",chat_history)
    # Convert chat history to lowercase for case-insensitive comparison
    user_chats = [chat["content"].lower() for chat in chat_history if chat["role"] == "user"]
    assistant_chats = [chat["content"].lower() for chat in chat_history if chat["role"] == "assistant"]
    
    # Flatten all chat history content into a single string for quick search
    chat_history_lower = " ".join(user_chats + assistant_chats)

    # Check if the user has explicitly requested to see properties
    user_requested_properties = any(
        phrase in chat for chat in user_chats for phrase in ["show me property", "list properties", "property options","show me properties now"]
    )

    # Check if the assistant has asked about showing properties
    assistant_asked_phrases = [
        "can i show you property now",
        "can i show u property now",
        "can i show you properties now",
        "can i show u properties now"
    ]
    assistant_asked = any(phrase in chat_history_lower for phrase in assistant_asked_phrases)

    # Check if the user has responded affirmatively
    affirmative_responses = ["yes", "yeah", "sure", "please", "okay", "ok"]
    user_responded_affirmatively = any(phrase in chat_history_lower for phrase in affirmative_responses)

    # List of possible locations (extendable)
    locations = ["gurgaon", "delhi", "mumbai", "bangalore", "noida","Dwarka Expressway"]
    location_mentioned = any(location in chat_history_lower for location in locations)

    # Determine if sufficient information has been gathered
    return location_mentioned and (user_requested_properties or (assistant_asked and user_responded_affirmatively))

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




def perform_sentiment_analysis(conversation_text):
    try:
        prompt = f"""Analyze this real estate conversation as a lead generation expert. Output in this format:
        1. Location Sentiment: [High/Medium/Low: 1-10]
        2. Price Sentiment: [High/Medium/Low: 1-10]
        3. Amenity Sentiment: [High/Medium/Low: 1-10]
        4. Builder Sentiment: [High/Medium/Low: 1-10]
        5. Emotional Triggers
        6. Unified Lead Title
        7. Recommended Actions

        Conversation: {conversation_text}"""

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a real estate lead qualification expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        return None



# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_sentiment_to_s3(user_id, sentiment_data):
    try:
        # Generate timestamp and filename
        timestamp = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
        cleaned_string = user_id.replace('|', '')
        # Create user folder structure
        user_folder = f"users/{cleaned_string}/sentiment/"
        s3_key = f"{user_folder}sentiment_{cleaned_string}_{timestamp}.csv"
        
        # Parse sentiment_data if it's text
        if isinstance(sentiment_data, str):
            parsed_sentiment = parse_sentiment(sentiment_data)
            
            # Extract values from parsed_sentiment in the correct order
            sentiment_values = [
                parsed_sentiment['Location Sentiment'],
                parsed_sentiment['Price Sentiment'],
                parsed_sentiment['Amenity Sentiment'],
                parsed_sentiment['Builder Sentiment'],
                parsed_sentiment['Emotional Triggers'],
                parsed_sentiment['Unified Lead Title'],
                parsed_sentiment['Recommended Actions']
            ]
        elif isinstance(sentiment_data, list):
            # If already a list, use as is
            sentiment_values = sentiment_data
        else:
            # If it's another format (like dictionary), convert appropriately
            sentiment_values = list(parsed_sentiment.values())
        
        # Set up CSV headers
        headers = [
            "Location Sentiment",
            "Price Sentiment",
            "Amenity Sentiment",
            "Builder Sentiment",
            "Emotional Triggers",
            "Unified Lead Title",
            "Recommended Actions"
        ]
        
        logger.info(f"Saving sentiment data to S3 for user_id {user_id} in folder {user_folder}")
        
        # Create CSV in memory
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(headers)  # Write header row
        writer.writerow(sentiment_values)  # Write data row
        
        # Upload to S3 with metadata
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv',
            Metadata={
                'user_id': user_id,
                'timestamp': timestamp,
                'type': 'sentiment-analysis'
            }
        )
        
        logger.info(f"Sentiment file saved successfully: {s3_key}")
        return s3_key

    except Exception as e:
        logger.error(f"S3 upload error: {str(e)}")
        return None



def parse_sentiment(sentiment_text):
    sentiment_data = {
        'Location Sentiment': '',
        'Price Sentiment': '',
        'Amenity Sentiment': '',
        'Builder Sentiment': '',
        'Emotional Triggers': '',
        'Unified Lead Title': '',
        'Recommended Actions': ''
    }
    
    current_section = None
    section_content = []
    
    for line in sentiment_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Remove numbering (like "1." or "2.") before checking for key matches
        line = re.sub(r'^\d+\.\s*', '', line)

        # Match line with sentiment keys (case insensitive)
        for key in sentiment_data.keys():
            if re.match(fr"^{re.escape(key)}:", line, re.IGNORECASE):
                if current_section:
                    sentiment_data[current_section] = ' '.join(section_content)
                
                # Extract text after ":" safely
                parts = line.split(":", 1)
                sentiment_text = parts[1].strip() if len(parts) > 1 else ""

                current_section = key
                section_content = [sentiment_text]
                break
        else:
            if current_section:
                section_content.append(line)
    
    if current_section:
        sentiment_data[current_section] = ' '.join(section_content)
    
    return sentiment_data




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






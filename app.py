# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
import json
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Define the FastAPI application
app = FastAPI(
    title="Dog Product Recommender API",
    description="An AI-powered service to provide personalized dog product recommendations using Groq LLMs."
)

# Mount the 'templates' directory to serve static files like index.html
# This makes it so that when you access '/', it serves the index.html from 'templates'
# We're also using HTMLResponse for rendering, which is more explicit for HTML content.
app.mount("/static", StaticFiles(directory="templates"), name="static")

# Get Groq API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set. Please create a .env file.")

# Define the request body model for the recommendation endpoint
class RecommendationRequest(BaseModel):
    dog_breed: str
    diet_preference: str
    product_type: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serves the main HTML page for the dog product recommender.
    """
    # For a simple single HTML file in the templates directory, we serve it directly.
    # In a larger app, you might use FastAPI's jinja2 template engine.
    with open("templates/index.html", "r") as f:
        return f.read()


@app.post("/get_recommendation")
async def get_recommendation(request_data: RecommendationRequest):
    """
    Handles the recommendation request by calling the Groq API.
    Receives dog details from the frontend and returns a personalized
    product recommendation and insight.
    """
    # Initialize llm_content outside try block for error logging
    llm_content = ""
    try:
        dog_breed = request_data.dog_breed.strip()
        diet_preference = request_data.diet_preference
        product_type = request_data.product_type.strip()

        # Input validation (FastAPI's BaseModel handles basic validation, but explicit checks are good)
        if not dog_breed or not product_type:
            raise HTTPException(status_code=400, detail="Dog breed and product type are required.")

        # Construct the prompt for the LLM
        prompt = f"""
        You are a helpful AI assistant for a dog product company. Your goal is to provide a personalized product recommendation and a relevant insight based on the dog's profile.

        Dog Breed: {dog_breed}
        Dietary Preference: {diet_preference}
        Desired Product Type: {product_type}

        Please provide:
        1. A specific product recommendation.
        2. An insight related to cost-benefit or community behavior for this product/dog type.

        Format your response strictly as a JSON object with two keys: "recommendation" and "insight".
        Example:
        {{
          "recommendation": "XYZ Brand Organic Chicken Dog Food",
          "insight": "80% of Golden Retriever owners prefer large bags for cost savings."
        }}
        Ensure the recommendation is plausible for the given inputs and the insight is creative and relevant.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }

        payload = {
            "model": "llama3-8b-8192", # Using an efficient open-source model available on Groq
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "response_format": { "type": "json_object" }, # Request JSON output
            "temperature": 0.7 # Adjust creativity/randomness
        }

        # Make request to Groq API
        groq_response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        groq_response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        groq_data = groq_response.json()
        
        # Parse the content which is a JSON string
        llm_content = groq_data['choices'][0]['message']['content']
        parsed_llm_content = json.loads(llm_content)

        recommendation = parsed_llm_content.get('recommendation')
        insight = parsed_llm_content.get('insight')

        if recommendation and insight:
            return {"recommendation": recommendation, "insight": insight}
        else:
            # Log the full LLM response if it doesn't contain expected keys
            print(f"WARNING: LLM response missing expected keys. Content: {llm_content}") # Use print for simple logging
            raise HTTPException(
                status_code=500, 
                detail="Could not parse recommendation or insight from LLM response. LLM might have deviated from format."
            )

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Error calling Groq API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect to recommendation service: {e}")
    except json.JSONDecodeError as e:
        print(f"ERROR: Error parsing LLM JSON response: {e}, Raw Content: {llm_content}")
        raise HTTPException(status_code=500, detail=f"Failed to process LLM response. Invalid JSON from LLM: {e}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {e}")

if __name__ == '__main__':
    # Get port from environment, default to 8000 for local development (FastAPI standard)
    port = int(os.environ.get("PORT", 8000))
    # Run Uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=port)

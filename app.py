# app.py
from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import requests
import json # <--- ADD THIS LINE

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Get Groq API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set. Please create a .env file.")

@app.route('/')
def index():
    """
    Renders the main HTML page for the dog product recommender.
    """
    return render_template('index.html')

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    """
    Handles the recommendation request by calling the Groq API.
    Receives dog details from the frontend and returns a personalized
    product recommendation and insight.
    """
    # Initialize llm_content outside try block for error logging
    llm_content = ""
    try:
        data = request.get_json()
        dog_breed = data.get('dog_breed')
        diet_preference = data.get('diet_preference')
        product_type = data.get('product_type')

        # Input validation
        if not dog_breed or not product_type:
            return jsonify({"error": "Dog breed and product type are required."}), 400

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
            return jsonify({"recommendation": recommendation, "insight": insight})
        else:
            # Log the full LLM response if it doesn't contain expected keys
            app.logger.warning(f"LLM response missing expected keys. Content: {llm_content}")
            return jsonify({"error": "Could not parse recommendation or insight from LLM response. LLM might have deviated from format."}), 500

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error calling Groq API: {e}")
        return jsonify({"error": f"Failed to connect to recommendation service: {e}"}), 500
    except json.JSONDecodeError as e:
        # Include the raw content that failed to parse for better debugging
        app.logger.error(f"Error parsing LLM JSON response: {e}, Raw Content: {llm_content}")
        return jsonify({"error": f"Failed to process LLM response. Invalid JSON from LLM: {e}"}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"error": f"An unexpected server error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True) # Run in debug mode during development

from flask import Flask, request, jsonify, render_template
import base64
import os
import json
import logging
import google.generativeai as genai
from PIL import Image
import io
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key="AIzaSyCgdgMfad-K30PqaNaz1Mv_dkNVqCrGkOM")
model = genai.GenerativeModel('gemini-1.5-pro')

def extract_json_from_response(response_text):
    """Extract JSON from a response that might be wrapped in markdown code blocks."""
    # Try to find JSON between markdown code blocks
    json_match = re.search(r'```(?:json)?\s*({\s*.*?\s*})\s*```', response_text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # If no markdown blocks, try to find JSON directly
    json_match = re.search(r'(\{[^}]+\})', response_text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # If no JSON found, return the original text
    return response_text

def process_image(image_data):
    try:
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Prepare the prompt
        prompt = """
        Analyze this flowchart image and provide a structured description.
        Return ONLY a JSON string with this exact structure:
        {
            "nodes": [
                {"id": "n1", "text": "node text", "type": "process"}
            ],
            "connections": [
                {"from": "n1", "to": "n2", "label": "connection text"}
            ]
        }
        For node types, use only: "start", "end", "process", or "decision".
        Make sure the JSON is valid and all nodes referenced in connections exist in the nodes array.
        """
        
        # Get response from Gemini
        response = model.generate_content([prompt, image])
        
        # Extract the JSON string from the response
        response_text = response.text
        logger.debug(f"Raw Gemini response: {response_text}")
        
        # Extract JSON from the response
        json_str = extract_json_from_response(response_text)
        logger.debug(f"Extracted JSON: {json_str}")
        
        # Validate JSON
        try:
            # Parse JSON to validate it
            flowchart_data = json.loads(json_str)
            
            # Validate required structure
            if 'nodes' not in flowchart_data or 'connections' not in flowchart_data:
                raise ValueError("Missing required nodes or connections in response")
                
            # Validate that all referenced nodes exist
            node_ids = {node['id'] for node in flowchart_data['nodes']}
            for conn in flowchart_data['connections']:
                if conn['from'] not in node_ids or conn['to'] not in node_ids:
                    raise ValueError(f"Connection references non-existent node: {conn}")
            
            return json.dumps(flowchart_data)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise ValueError(f"Invalid JSON in response: {e}")
            
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_flowchart():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({'error': 'Empty file provided'}), 400
            
        if not image_file.filename.lower().endswith('.png'):
            return jsonify({'error': 'Only PNG files are supported'}), 400
        
        # Read and encode image
        image_data = base64.b64encode(image_file.read()).decode()
        
        # Process the image
        result = process_image(f"data:image/png;base64,{image_data}")
        
        # Log successful processing
        logger.info("Successfully processed flowchart")
        
        return jsonify({'result': result})
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': f"Invalid response format: {str(e)}"}), 422
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred while processing the flowchart'}), 500

if __name__ == '__main__':
    app.run(debug=True)
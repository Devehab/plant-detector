import os
from flask import Flask, request, jsonify
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import base64
import io

# Load environment variables
load_dotenv()

# Configure Google API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

app = Flask(__name__, static_folder='static')

# Initialize Gemini model
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_plant():
    try:
        # Get the image data from the request
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Prepare the prompt
        prompt = """
        أنت خبير في النباتات. قم بتحليل هذه الصورة وقدم المعلومات التالية باللغة العربية:
        1. نوع النبات
        2. الحالة العامة للنبات
        3. المشاكل المحتملة (إن وجدت)
        4. نصائح للعناية والحل
        """

        # Generate response
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": img_str}])
        response.resolve()
        
        return jsonify({
            'success': True,
            'analysis': response.text
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)

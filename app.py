import os
from flask import Flask, request, jsonify
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import base64
import io

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY is not set in the environment variables.")
genai.configure(api_key=api_key)

app = Flask(__name__, static_folder='static')

# Initialize Gemini model
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
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
        # Validate input
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'Missing image data'}), 400
        
        # Decode image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Prepare the prompt
        prompt = """
        أنت خبير في تحليل النباتات. قم بتحليل هذه الصورة وتقديم تقرير مفصل يتبع التنسيق التالي بدقة.
        يجب أن يحتوي كل قسم على عنوان متبوعاً بنقطتين (:) ثم المحتوى في السطر التالي.

        نوع النبات:
        [اسم النبات]

        وصف النبات:
        [وصف مختصر للنبات في سطر أو سطرين]

        الحالة العامة:
        [تقييم الحالة العامة للنبات]

        المشاكل المحتملة:
        [قائمة المشاكل التي تراها في النبات]

        نصائح للعناية:
        [نصائح مفصلة للعناية بالنبات وحل المشاكل]

        تعليمات هامة:
        - اكتب كل عنوان بالضبط كما هو موضح أعلاه متبوعاً بنقطتين
        - اكتب المحتوى في السطر التالي مباشرة بعد كل عنوان
        - لا تضف أي أرقام أو نقاط قبل المحتوى
        - لا تغير العناوين أو ترتيبها
        - اكتب إجابة كاملة ومفيدة لكل قسم
        """

        # Generate response
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": img_str}])
        
        return jsonify({
            'success': True,
            'analysis': response.text
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=3000)
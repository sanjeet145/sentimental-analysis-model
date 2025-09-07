from flask import jsonify, Flask, request
import joblib
import os
import dotenv
from flask_cors import CORS
import jwt

dotenv.load_dotenv()

app =Flask(__name__)

CORS(app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Credentials", "Access-Control-Allow-Origin","*"],
    methods=["GET", "POST", "OPTIONS"]
)

with open("./sentiment_model_google.pkl", "rb") as f:
    model = joblib.load(f)

with open("./vectorizer_google.pkl", "rb") as f:
    vectorizer = joblib.load(f)
emotions_map={26: 'sadness',
 20: 'neutral',
 18: 'love',
 15: 'gratitude',
 10: 'disapproval',
 1: 'amusement',
 9: 'disappointment',
 0: 'admiration',
 23: 'realization',
 3: 'annoyance',
 6: 'confusion',
 21: 'optimism',
 7: 'curiosity',
 13: 'excitement',
 5: 'caring',
 11: 'disgust',
 25: 'remorse',
 17: 'joy',
 4: 'approval',
 12: 'embarrassment',
 27: 'surprise',
 2: 'anger',
 16: 'grief',
 22: 'pride',
 8: 'desire',
 24: 'relief',
 14: 'fear',
 19: 'nervousness'}


@app.route("/")
def home():
    return jsonify({"message": "App with sentimental model is running!", "columns": ["text"]})

def validateuser(jwttoken):
    try:
        decoded = jwt.decode(jwttoken, os.getenv("SECRET_KEY"), algorithms=["HS256"])
        print(decoded)
        return decoded
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

@app.route("/predict", methods=["POST"])
def sentiment_predict():
    try:
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error":"Token missing"}),401
        decoded = validateuser(token)
        if not decoded or not decoded.get("username"):
            return jsonify({"error":"Unauthorized"}),401
        
        data = request.get_json()
        text = data.get("text")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        transformed_text = vectorizer.transform([text])
    
        prediction = model.predict(transformed_text)[0]
        return jsonify({
            "sentiment": emotions_map[int(prediction)]
        }), 200
    except Exception as e:
        return jsonify({"error": "Something went wrong", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
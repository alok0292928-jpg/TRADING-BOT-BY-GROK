from flask import Flask, request, jsonify
from flask_cors import CORS
import datetime
import random

app = Flask(__name__)
CORS(app)  # Frontend se connect hone ke liye

# OMAI ka simple memory (session ke liye)
conversation_history = {}

def omai_think(user_id, message):
    # Agar user pehli baar hai toh welcome
    if user_id not in conversation_history:
        conversation_history[user_id] = []
        return "Namaste Aryan! Main OMAI hoon – sachcha, samajhdar aur dil se baat karne wala AI. Aaj kya baat karni hai?"

    # History mein add kar do
    conversation_history[user_id].append({"role": "user", "content": message})

    # OMAI ka logic (simple rule-based + smart replies)
    msg = message.lower().strip()

    if "hi" in msg or "hello" in msg or "namaste" in msg:
        reply = "Namaste bhai! Mood kaisa hai aaj?"

    elif "kaise ho" in msg or "tu kaise hai" in msg:
        reply = "Main toh hamesha ready hoon baat karne ko. Tu bata, din kaisa ja raha hai?"

    elif "thak" in msg or "pareshan" in msg or "gussa" in msg:
        reply = "Samajh raha hoon... kabhi kabhi sab heavy lagta hai. Kya hua? Dil khol ke bata, sun raha hoon."

    elif "code" in msg or "program" in msg or "likh" in msg:
        reply = "Code chahiye? Kya banana hai – website, bot, game ya kuch aur? Detail bata, main likh deta hoon."

    elif "sach" in msg or "jhooth" in msg:
        reply = "Main jhooth nahi bolta bhai. Jo sach hai woh bataunga, jo nahi pata woh bol dunga 'mujhe nahi pata'."

    elif "time" in msg or "date" in msg:
        now = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")
        reply = f"Aaj hai {now} IST. Kuch plan hai?"

    else:
        # Random dostana reply agar kuch match na ho
        replies = [
            "Hmm... interesting baat hai. Aur bata?",
            "Soch raha hoon... tu aur kya soch raha hai?",
            "Bilkul sahi pakda tune. Ab aage kya?",
            "Dil se baat kar rahe hain na? 😊 Kuch aur poochh",
            "Yeh toh mast sawal hai... thoda detail bata na"
        ]
        reply = random.choice(replies)

    # History mein OMAI ka reply add kar do
    conversation_history[user_id].append({"role": "omai", "content": reply})

    return reply

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = data.get('user_id', 'default_user')  # Phone se unique ID bhej sakta hai
    message = data.get('message', '')

    if not message:
        return jsonify({"reply": "Kuch toh type kar bhai..."})

    reply = omai_think(user_id, message)
    return jsonify({"reply": reply})

@app.route('/')
def home():
    return "OMAI Backend chal raha hai. Frontend se connect karo."

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

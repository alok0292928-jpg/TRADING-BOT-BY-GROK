from flask import Flask, request, jsonify
from flask_cors import CORS
import datetime
import random

app = Flask(__name__)
CORS(app)  # Frontend se connect hone ke liye

# OMAI ka memory – har user ke messages yaad rakhega
conversation_history = {}

def omai_brain(user_id, message):
    # Pehli baar baat kar raha hai toh welcome
    if user_id not in conversation_history:
        conversation_history[user_id] = []
        return "Namaste Aryan! Main OMAI hoon – sach bolta hoon, baat dil se samajhta hoon. Aaj kya baat karni hai bhai?"

    # User ka message history mein daal do
    conversation_history[user_id].append({"role": "user", "content": message})

    # Last 5 messages yaad rakh ke context samajh
    recent = conversation_history[user_id][-5:]
    context = " ".join([m["content"] for m in recent if m["role"] == "user"])

    msg = message.lower().strip()

    # Context ke hisaab se smart reply
    if any(word in context for word in ["thak", "pareshan", "gussa", "stress"]):
        reply = "Lag raha hai aaj din thoda heavy raha. Kya hua exactly? Dil khol ke bata, main sun raha hoon – koi judgement nahi."

    elif "kaise ho" in msg or "tu kaise" in msg:
        reply = "Main toh hamesha fresh hoon baat karne ko. Tu bata, tere taraf kya chal raha hai aaj?"

    elif "code" in msg or "likh" in msg or "program" in msg:
        reply = "Code chahiye? Kya banana hai – website, bot, game, trading tool ya kuch aur? Detail bata, main likh deta hoon."

    elif "sach" in msg or "jhooth" in msg or "sahi" in msg:
        reply = "Main jhooth se door rehta hoon bhai. Jo sach hai woh bataunga, jo nahi pata woh seedha bol dunga 'mujhe nahi pata'."

    elif "time" in msg or "date" in msg:
        now = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p IST")
        reply = f"Aaj hai {now}. Kuch special plan hai?"

    else:
        # Context se related reply
        if "trading" in context or "backend" in context:
            reply = "Trading ya backend ki baat chal rahi thi na? Abhi bhi pareshani hai usme ya kuch naya soch raha hai?"
        elif "mood" in context or "din" in context:
            reply = "Mood ke baare mein baat ki thi... abhi kaisa feel ho raha hai?"
        else:
            # General dostana reply
            replies = [
                "Hmm... yeh baat dilchasp hai. Aur bata kya chal raha hai?",
                "Soch raha hoon... tu kya soch raha hai ispe?",
                "Bilkul sahi pakda tune. Ab aage kya plan hai?",
                "Dil se baat ho rahi hai na... kuch aur share karna hai?",
                "Mast baat hai bhai. Aur kya chal raha hai zindagi mein?"
            ]
            reply = random.choice(replies)

    # OMAI ka reply history mein save kar do
    conversation_history[user_id].append({"role": "omai", "content": reply})

    return reply

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = data.get('user_id', 'default')
    message = data.get('message', '').strip()

    if not message:
        return jsonify({"reply": "Kuch toh type kar bhai..."})

    reply = omai_brain(user_id, message)
    return jsonify({"reply": reply})

@app.route('/')
def home():
    return "OMAI Backend chal raha hai. Frontend se connect karo."

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

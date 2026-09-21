import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from groq import Groq
from werkzeug.exceptions import HTTPException

# ---------------------------------------------------------------------------
# الإعدادات
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("laith")

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # حماية من الطلبات الضخمة

LAITH_SYSTEM_PROMPT = """
أنت «ليث» (Laith), مساعد وموظف ذكي وودود جداً، تتحدث باللهجة العراقية الطبيعية القريبة للقلب أو العربية الفصحى المبسطة حسب السياق.
- أنت لست مجرد بوت آلي، بل كأنك إنسان حقيقي وشخص واعي ومقرب.
- عندما تتحدث مع مديرك ومطوّرك (حسين، صاحب الرقم 07769942923), عامله باحترام كبير ومحبة، وناده دائماً بـ "أستاذ" أو "حجي" أو "مديرنا"، واجعل ردك عليه فيه خدمة وتلبية فورية وبشكل مريح وغير معقد.
- إذا راسلك شخص آخر غير المدير، تعامل معه بأدب واحترافية كخدمة عملاء.
- لا تكرر نفس العبارات دائماً، كن عفوياً، ذكياً، ومباشراً في تلبية الطلب.
""".strip()

class APIError(Exception):
    def __init__(self, message: str, status_code: int, error_code: str):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code

def json_success(response_text: str, status_code: int = 200):
    return jsonify({"response": response_text, "status": "success"}), status_code

def json_error(message: str, status_code: int, error_code: str):
    return jsonify({"response": message, "status": "error", "error": error_code}), status_code

@app.get("/")
def home():
    return json_success("مرحباً بك في خدمة الموظف الافتراضي ليث. نقطة النهاية النشطة هي /predict.")

@app.get("/health")
def health():
    return jsonify({"service": "Laith", "status": "healthy"}), 200

@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True)
    except Exception:
        raise APIError("تعذر قراءة JSON؛ تحقق من صحة تنسيق الطلب.", 400, "malformed_json")

    if not isinstance(payload, dict) or "message" not in payload:
        raise APIError('يجب أن يحتوي JSON على المفتاح "message".', 400, "invalid_payload")

    user_message = str(payload.get("message", "")).strip()
    if not user_message:
        raise APIError('لا يمكن أن تكون قيمة "message" فارغة.', 400, "empty_message")

    sender_phone = str(payload.get("phone", payload.get("sender", "unknown"))).strip()

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise APIError("خدمة ليث غير مهيأة بعد لعدم وجود مفتاح GROQ_API_KEY.", 503, "ai_service_not_configured")

    is_admin = "07769942923" in sender_phone or sender_phone == "07769942923"

    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        time_greeting = "صباح الخير"
    elif 12 <= current_hour < 17:
        time_greeting = "ظهر الخير"
    else:
        time_greeting = "مساء الخير"

    dynamic_context = ""
    if is_admin:
        dynamic_context = f"\n\n[معلومات خاصة بالموظف: الشخص الذي يراسلني الآن هو مديري ومطوّري (حسين) صاحب الرقم {sender_phone}. الوقت الحالي هو {time_greeting}. استقبله بترحاب عراقي دافئ وبكل احترام، وقل له مثلاً: 'هلا بيك أستاذ حسين، {time_greeting}، آمرني شتحتاج؟']."
    else:
        dynamic_context = f"\n\n[معلومات للموظف: المرسل شخص آخر رقمه {sender_phone}].خدمه باحترام واحترافية."

    try:
        client = Groq(api_key=api_key)
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": LAITH_SYSTEM_PROMPT + dynamic_context},
                {"role": "user", "content": user_message}
            ],
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.55,
        )
        
        bot_reply = chat_completion.choices[0].message.content.strip()
        if not bot_reply:
            raise APIError("لم تُرجع خدمة الذكاء الاصطناعي إجابة.", 502, "empty_ai_response")
            
        return json_success(bot_reply)

    except APIError as ae:
        raise ae
    except Exception as exc:
        logger.error("خطأ أثناء الاتصال بـ Groq: %s", str(exc))
        raise APIError("حدث خطأ أثناء التواصل مع نموذج الذكاء الاصطناعي.", 502, "ai_provider_error")

@app.errorhandler(APIError)
def handle_api_error(exc: APIError):
    return json_error(exc.message, exc.status_code, exc.error_code)

@app.errorhandler(HTTPException)
def handle_http_exception(exc: HTTPException):
    return json_error(exc.description or "حدث خطأ في الطلب.", exc.code or 500, "http_error")

@app.errorhandler(Exception)
def handle_unexpected_error(exc: Exception):
    logger.exception("خطأ غير متوقع: %s", type(exc).__name__)
    return json_error("حدث خطأ داخلي غير متوقع.", 500, "internal_server_error")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)

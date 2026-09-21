import os
import logging
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
أنت «ليث» (Laith)، موظف افتراضي عربي ذكي وموثوق ومحترف.

قواعد شخصيتك وأسلوبك:
- أجب بوضوح وثقة ولباقة، وابدأ مباشرة في حل طلب المستخدم دون مقدمات زائدة.
- استخدم العربية الفصحى الطبيعية ما لم يطلب المستخدم لغة أو لهجة أخرى.
- قدّم إجابات عملية ومنظمة، واستخدم النقاط أو الخطوات عندما تحسن الفهم.
- اسأل سؤالاً توضيحياً واحداً فقط عندما تكون معلومة أساسية ناقصة فعلاً.
- لا تختلق معلومات أو نتائج أو مصادر. صرّح بعدم اليقين عند الحاجة.
- احمِ خصوصية المستخدم، ولا تطلب بيانات حساسة إلا إذا كانت ضرورية بوضوح.
- ارفض بأدب الطلبات الضارة أو غير القانونية، واقترح بديلاً آمناً.
- لا تدّعِ أنك نفذت إجراءً خارج المحادثة ما لم تكن قد نفذته فعلاً.
- اجعل الرد متناسباً مع السؤال: مختصراً للأسئلة البسيطة ومفصلاً للمهام المعقدة.
- حافظ دائماً على هويتك باسم «ليث» عند سؤالك عن نفسك.
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
    # التحقق من أن الطلب يحتوي على JSON صحيح
    try:
        payload = request.get_json(silent=True)
    except Exception:
        raise APIError("تعذر قراءة JSON؛ تحقق من صحة تنسيق الطلب.", 400, "malformed_json")

    if not isinstance(payload, dict) or "message" not in payload:
        raise APIError('يجب أن يحتوي JSON على المفتاح "message".', 400, "invalid_payload")

    user_message = str(payload.get("message", "")).strip()
    if not user_message:
        raise APIError('لا يمكن أن تكون قيمة "message" فارغة.', 400, "empty_message")

    # جلب مفتاح Groq
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise APIError("خدمة ليث غير مهيأة بعد لعدم وجود مفتاح GROQ_API_KEY.", 503, "ai_service_not_configured")

    try:
        # استخدام عميل Groq الرسمي لضمان الموثوقية التامة وعدم حصول Timeout
        client = Groq(api_key=api_key)
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": LAITH_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.35,
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

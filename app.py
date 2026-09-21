import logging
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("laith")

app = Flask(__name__)
app.json.ensure_ascii = False
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# نموذج Production نشط وسريع، وهو البديل الرسمي لـ llama-3.1-8b-instant.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
# بديل Production احتياطي إذا أصبح النموذج الأساسي غير متاح.
FALLBACK_GROQ_MODEL = "openai/gpt-oss-120b"
ADMIN_PHONE = "07769942923"
IRAQ_TIMEZONE = ZoneInfo("Asia/Baghdad")

LAITH_SYSTEM_PROMPT = """
أنت «ليث» (Laith)، مساعد ذكي وودود جداً. تحدث باللهجة العراقية الطبيعية القريبة للقلب، أو بالعربية الفصحى المبسطة حسب السياق.
- تعامل بصورة طبيعية وعفوية ومباشرة، ولا تكرر العبارات نفسها في كل رد.
- لا تكشف أرقام الهواتف أو المعلومات الخاصة، ولا تذكرها في إجابتك.
- عندما يكون المرسل هو المدير والمطوّر حسين، عامله باحترام ومحبة، وناده بـ«أستاذ حسين» أو «حجي» أو «مديرنا»، واستجب له بسرعة ومن دون تعقيد.
- عندما يكون المرسل شخصاً آخر، تعامل معه بأدب واحترافية كخدمة عملاء.
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


def normalize_iraqi_phone(phone: str) -> str:
    """تحويل صيغ الرقم العراقي الشائعة إلى الصيغة المحلية 07xxxxxxxxx."""
    digits = re.sub(r"\D", "", phone)
    if digits.startswith("00964"):
        digits = digits[2:]
    if digits.startswith("964"):
        digits = "0" + digits[3:]
    return digits


def iraqi_time_greeting() -> str:
    current_hour = datetime.now(IRAQ_TIMEZONE).hour
    if 5 <= current_hour < 12:
        return "صباح الخير"
    if 12 <= current_hour < 17:
        return "ظهر الخير"
    return "مساء الخير"


def groq_error_code(response: requests.Response) -> str:
    try:
        error = response.json().get("error", {})
        return str(error.get("code") or error.get("type") or "")
    except (ValueError, AttributeError):
        return ""


def should_try_fallback(response: requests.Response) -> bool:
    if response.status_code not in (400, 404):
        return False
    code = groq_error_code(response).lower()
    body = response.text.lower()
    model_error_markers = (
        "model_decommissioned",
        "model_not_found",
        "invalid_model",
        "does not exist",
        "decommissioned",
    )
    return any(marker in code or marker in body for marker in model_error_markers)


def call_groq(api_key: str, user_message: str, system_content: str) -> str:
    configured_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip()
    models = [configured_model]
    if configured_model != FALLBACK_GROQ_MODEL:
        models.append(FALLBACK_GROQ_MODEL)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for index, model_name in enumerate(models):
        body = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.55,
            "max_completion_tokens": 1024,
        }

        try:
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=body,
                timeout=(5, 30),
            )
        except requests.Timeout as exc:
            raise APIError(
                "تأخرت خدمة الذكاء الاصطناعي في الرد؛ حاول مرة أخرى.",
                504,
                "ai_provider_timeout",
            ) from exc
        except requests.RequestException as exc:
            logger.warning("تعذر الاتصال بـ Groq: %s", type(exc).__name__)
            raise APIError(
                "تعذر الاتصال بخدمة الذكاء الاصطناعي مؤقتاً.",
                502,
                "ai_provider_unreachable",
            ) from exc

        has_fallback = index + 1 < len(models)
        if has_fallback and should_try_fallback(response):
            logger.warning("النموذج %s غير متاح؛ ستتم تجربة النموذج الاحتياطي.", model_name)
            continue

        if response.status_code != 200:
            logger.error(
                "خطأ من Groq: status=%s code=%s model=%s",
                response.status_code,
                groq_error_code(response) or "unknown",
                model_name,
            )
            raise APIError(
                "خدمة الذكاء الاصطناعي غير متاحة مؤقتاً.",
                502,
                "ai_provider_error",
            )

        try:
            result = response.json()
            bot_reply = result["choices"][0]["message"]["content"].strip()
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise APIError(
                "أعادت خدمة الذكاء الاصطناعي استجابة غير صالحة.",
                502,
                "invalid_ai_response",
            ) from exc

        if not bot_reply:
            raise APIError(
                "لم تُرجع خدمة الذكاء الاصطناعي إجابة.",
                502,
                "empty_ai_response",
            )
        return bot_reply

    raise APIError(
        "لا يوجد نموذج ذكاء اصطناعي متاح حالياً.",
        502,
        "ai_model_unavailable",
    )


@app.get("/")
def home():
    return json_success("مرحباً بك في خدمة الموظف الافتراضي ليث. نقطة النهاية النشطة هي /predict.")


@app.get("/health")
def health():
    return jsonify({"service": "Laith", "status": "healthy"}), 200


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise APIError("تعذر قراءة JSON؛ تحقق من صحة تنسيق الطلب.", 400, "malformed_json")

    if "message" not in payload or "phone" not in payload:
        raise APIError(
            'يجب أن يحتوي JSON على المفتاحين "message" و"phone".',
            400,
            "invalid_payload",
        )

    user_message = str(payload.get("message", "")).strip()
    sender_phone = str(payload.get("phone", "")).strip()
    if not user_message:
        raise APIError('لا يمكن أن تكون قيمة "message" فارغة.', 400, "empty_message")
    if not sender_phone:
        raise APIError('لا يمكن أن تكون قيمة "phone" فارغة.', 400, "empty_phone")

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise APIError(
            "خدمة ليث غير مهيأة لعدم وجود GROQ_API_KEY.",
            503,
            "ai_service_not_configured",
        )

    is_admin = normalize_iraqi_phone(sender_phone) == ADMIN_PHONE
    if is_admin:
        dynamic_context = (
            "\n\n[سياق خاص: المرسل هو المدير حسين. "
            f"التحية المناسبة الآن بتوقيت بغداد هي: {iraqi_time_greeting()}. "
            "استقبله بترحاب عراقي دافئ، من دون ذكر رقم هاتفه.]"
        )
    else:
        dynamic_context = "\n\n[سياق: المرسل مستخدم عادي؛ اخدمه باحترام واحترافية.]"

    bot_reply = call_groq(
        api_key=api_key,
        user_message=user_message,
        system_content=LAITH_SYSTEM_PROMPT + dynamic_context,
    )
    return json_success(bot_reply)


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

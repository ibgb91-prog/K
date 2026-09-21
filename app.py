"""خدمة الموظف الافتراضي الذكي «ليث» باستخدام Flask.

التشغيل محلياً:
    pip install Flask
    export OPENAI_API_KEY="your-api-key"
    python app.py

متغيرات البيئة الاختيارية:
    PORT=8080
    OPENAI_API_KEY=...
    OPENAI_BASE_URL=https://api.openai.com/v1
    OPENAI_MODEL=gpt-4o-mini
    LLM_TIMEOUT=45
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from flask import Flask, jsonify, request
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


@dataclass
class APIError(Exception):
    """خطأ متوقع يمكن تحويله بأمان إلى استجابة JSON."""

    message: str
    status_code: int
    error_code: str


# ---------------------------------------------------------------------------
# أدوات مساعدة
# ---------------------------------------------------------------------------

def json_success(response_text: str, status_code: int = 200):
    """إنشاء استجابة نجاح موحدة."""
    return (
        jsonify(
            {
                "response": response_text,
                "status": "success",
            }
        ),
        status_code,
    )


def json_error(message: str, status_code: int, error_code: str):
    """إنشاء استجابة خطأ موحدة دون كشف تفاصيل داخلية."""
    return (
        jsonify(
            {
                "response": message,
                "status": "error",
                "error": error_code,
            }
        ),
        status_code,
    )


def validate_predict_payload(payload: Any) -> str:
    """التحقق من أن جسم الطلب هو بالضبط: {"message": "..."}."""
    if not isinstance(payload, dict):
        raise APIError(
            "يجب أن يكون جسم الطلب كائن JSON صالحاً.",
            400,
            "invalid_json_object",
        )

    if set(payload.keys()) != {"message"}:
        raise APIError(
            'يجب أن يحتوي JSON على المفتاح "message" فقط.',
            400,
            "invalid_payload_keys",
        )

    message = payload["message"]
    if not isinstance(message, str):
        raise APIError(
            'يجب أن تكون قيمة "message" نصاً.',
            400,
            "message_must_be_string",
        )

    message = message.strip()
    if not message:
        raise APIError(
            'لا يمكن أن تكون قيمة "message" فارغة.',
            400,
            "empty_message",
        )

    if len(message) > 10_000:
        raise APIError(
            "الرسالة طويلة جداً؛ الحد الأقصى هو 10000 حرف.",
            413,
            "message_too_long",
        )

    return message


def generate_laith_response(message: str) -> str:
    """إرسال رسالة المستخدم إلى واجهة متوافقة مع OpenAI وإرجاع رد ليث."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise APIError(
            "خدمة ليث غير مهيأة بعد. يرجى ضبط متغير البيئة OPENAI_API_KEY.",
            503,
            "ai_service_not_configured",
        )

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

    try:
        timeout = float(os.getenv("LLM_TIMEOUT", "45"))
    except ValueError:
        timeout = 45.0

    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": LAITH_SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            "temperature": 0.35,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    upstream_request = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": "Laith-Flask/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(upstream_request, timeout=timeout) as upstream:
            result = json.loads(upstream.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # نسجل رمز الحالة فقط لتجنب تسريب مفاتيح أو تفاصيل حساسة للمستخدم.
        logger.error("مزود الذكاء الاصطناعي أعاد HTTP %s", exc.code)
        if exc.code == 429:
            raise APIError(
                "ليث مشغول حالياً. يرجى المحاولة بعد قليل.",
                503,
                "ai_rate_limited",
            ) from exc
        raise APIError(
            "تعذر الحصول على رد من خدمة الذكاء الاصطناعي.",
            502,
            "ai_provider_error",
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.error("تعذر الاتصال بمزود الذكاء الاصطناعي: %s", type(exc).__name__)
        raise APIError(
            "تعذر الاتصال بخدمة الذكاء الاصطناعي حالياً.",
            503,
            "ai_service_unavailable",
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.error("استجابة غير صالحة من مزود الذكاء الاصطناعي")
        raise APIError(
            "استلم ليث استجابة غير صالحة من مزود الخدمة.",
            502,
            "invalid_ai_response",
        ) from exc

    try:
        response_text = result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        logger.error("بنية استجابة مزود الذكاء الاصطناعي غير متوقعة")
        raise APIError(
            "استلم ليث استجابة غير مكتملة من مزود الخدمة.",
            502,
            "incomplete_ai_response",
        ) from exc

    if not response_text:
        raise APIError(
            "لم تُرجع خدمة الذكاء الاصطناعي إجابة.",
            502,
            "empty_ai_response",
        )

    return response_text


# ---------------------------------------------------------------------------
# المسارات
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    """الصفحة الرئيسية لخدمة ليث."""
    return json_success(
        "مرحباً بك في خدمة الموظف الافتراضي ليث. أرسل طلب POST إلى /predict مع JSON بالشكل: {\"message\": \"مرحباً يا ليث\"}."
    )


@app.get("/health")
def health():
    """فحص صحة خفيف للسيرفر لا يعتمد على مزود الذكاء الاصطناعي."""
    return (
        jsonify(
            {
                "service": "Laith",
                "status": "healthy",
            }
        ),
        200,
    )


@app.post("/predict")
def predict():
    """استقبال رسالة JSON وإرجاع رد الموظف الافتراضي ليث."""
    if not request.is_json:
        raise APIError(
            "يجب إرسال الطلب بنوع المحتوى application/json.",
            415,
            "unsupported_media_type",
        )

    try:
        payload = request.get_json(silent=False)
    except Exception as exc:
        raise APIError(
            "تعذر قراءة JSON؛ تحقق من صحة تنسيق الطلب.",
            400,
            "malformed_json",
        ) from exc

    message = validate_predict_payload(payload)
    response_text = generate_laith_response(message)
    return json_success(response_text)


# ---------------------------------------------------------------------------
# معالجة الأخطاء
# ---------------------------------------------------------------------------

@app.errorhandler(APIError)
def handle_api_error(exc: APIError):
    return json_error(exc.message, exc.status_code, exc.error_code)


@app.errorhandler(HTTPException)
def handle_http_exception(exc: HTTPException):
    messages = {
        404: "المسار المطلوب غير موجود.",
        405: "طريقة HTTP غير مسموحة لهذا المسار.",
        413: "حجم الطلب أكبر من الحد المسموح.",
    }
    return json_error(
        messages.get(exc.code, "حدث خطأ في الطلب."),
        exc.code or 500,
        (exc.name or "http_error").lower().replace(" ", "_"),
    )


@app.errorhandler(Exception)
def handle_unexpected_error(exc: Exception):
    logger.exception("خطأ غير متوقع: %s", type(exc).__name__)
    return json_error(
        "حدث خطأ داخلي غير متوقع. يرجى المحاولة لاحقاً.",
        500,
        "internal_server_error",
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)

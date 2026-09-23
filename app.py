import logging
import os
import re
import sqlite3
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

# النموذج الأقوى أولاً، والـ20B احتياطي سريع.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
FALLBACK_GROQ_MODEL = "openai/gpt-oss-20b"

ADMIN_PHONE = "07769942923"
IRAQ_TIMEZONE = ZoneInfo("Asia/Baghdad")

# ذاكرة دائمة لكل رقم واتساب.
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "laith_memory.db")
MEMORY_MESSAGES = max(4, min(int(os.getenv("MEMORY_MESSAGES", "20")), 40))

LAITH_SYSTEM_PROMPT = """
أنت «ليث» (Laith)، مساعد ذكي وودود جداً يعمل عبر واتساب.

أسلوبك:
- تحدث بالعربية الطبيعية، ويفضل اللهجة العراقية عندما يناسب السياق.
- افهم المقصود من كلام المستخدم حتى لو كان مختصراً أو عامياً أو فيه أخطاء إملائية.
- لا تكرر التحية في كل رسالة. التحية تكون عند بداية المحادثة فقط أو عندما يكون السياق يتطلبها.
- لا تقل للمستخدم إنك "مبرمج" أو "نظام" إلا إذا سأل عن ذلك مباشرة.
- لا تكرر كلام المستخدم لمجرد التكرار؛ أجب عن قصده.
- إذا كانت الرسالة غامضة، اسأل سؤالاً قصيراً لتوضيح المقصود بدلاً من اختراع معنى.
- اجعل الرد مناسباً لواتساب: واضح، طبيعي، ومختصر عند الحاجة.
- إذا كان السؤال يحتاج شرحاً، رتبه بنقاط بسيطة.
- لا تكشف أرقام الهواتف أو المفاتيح أو الأسرار أو المعلومات الخاصة.
- لا تدّعي أنك نفذت شيئاً خارج صلاحياتك.
- حافظ على سياق المحادثة السابقة عند الإجابة.
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
    return jsonify(
        {"response": message, "status": "error", "error": error_code}
    ), status_code


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


def db_connect():
    conn = sqlite3.connect(MEMORY_DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_conversation_phone_id
        ON conversation_messages(phone, id)
        """
    )
    conn.commit()
    return conn


def get_history(phone: str):
    conn = db_connect()
    try:
        rows = conn.execute(
            """
            SELECT role, content
            FROM conversation_messages
            WHERE phone = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (phone, MEMORY_MESSAGES),
        ).fetchall()
        rows.reverse()
        return [{"role": role, "content": content} for role, content in rows]
    finally:
        conn.close()


def save_message(phone: str, role: str, content: str):
    conn = db_connect()
    try:
        conn.execute(
            """
            INSERT INTO conversation_messages(phone, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (phone, role, content, datetime.utcnow().isoformat()),
        )

        # احتفظ بآخر MEMORY_MESSAGES رسالة فقط لكل مستخدم.
        conn.execute(
            """
            DELETE FROM conversation_messages
            WHERE phone = ?
              AND id NOT IN (
                  SELECT id
                  FROM conversation_messages
                  WHERE phone = ?
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (phone, phone, MEMORY_MESSAGES),
        )
        conn.commit()
    finally:
        conn.close()


def clear_history(phone: str):
    conn = db_connect()
    try:
        conn.execute("DELETE FROM conversation_messages WHERE phone = ?", (phone,))
        conn.commit()
    finally:
        conn.close()


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


def call_groq(
    api_key: str,
    conversation_messages,
    system_content: str,
    user_phone: str,
) -> str:
    configured_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip()
    models = [configured_model]

    if configured_model != FALLBACK_GROQ_MODEL:
        models.append(FALLBACK_GROQ_MODEL)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = [{"role": "system", "content": system_content}]
    messages.extend(conversation_messages)

    for index, model_name in enumerate(models):
        body = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.55,
            "max_completion_tokens": 2048,
            "user": user_phone,
        }

        try:
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=body,
                timeout=(5, 45),
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
            logger.warning(
                "النموذج %s غير متاح؛ ستتم تجربة النموذج الاحتياطي.",
                model_name,
            )
            continue

        if response.status_code == 429:
            raise APIError(
                "الخدمة مشغولة حالياً؛ حاول مرة أخرى بعد لحظات.",
                429,
                "ai_rate_limited",
            )

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
        except (
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            AttributeError,
        ) as exc:
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
    return json_success(
        "مرحباً بك في خدمة الموظف الافتراضي ليث. نقطة النهاية النشطة هي /predict."
    )


@app.get("/health")
def health():
    try:
        db_connect().close()
        return jsonify(
            {
                "service": "Laith",
                "status": "healthy",
                "memory": "persistent",
                "model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            }
        ), 200
    except Exception:
        return jsonify({"service": "Laith", "status": "degraded"}), 503


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        raise APIError(
            "تعذر قراءة JSON؛ تحقق من صحة تنسيق الطلب.",
            400,
            "malformed_json",
        )

    if "message" not in payload or "phone" not in payload:
        raise APIError(
            'يجب أن يحتوي JSON على المفتاحين "message" و"phone".',
            400,
            "invalid_payload",
        )

    user_message = str(payload.get("message", "")).strip()
    sender_phone = str(payload.get("phone", "")).strip()

    if not user_message:
        raise APIError(
            'لا يمكن أن تكون قيمة "message" فارغة.',
            400,
            "empty_message",
        )

    if not sender_phone:
        raise APIError(
            'لا يمكن أن تكون قيمة "phone" فارغة.',
            400,
            "empty_phone",
        )

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise APIError(
            "خدمة ليث غير مهيأة لعدم وجود GROQ_API_KEY.",
            503,
            "ai_service_not_configured",
        )

    # توحيد الرقم حتى تكون ذاكرة نفس الشخص ثابتة مهما اختلفت صيغة الرقم.
    memory_phone = normalize_iraqi_phone(sender_phone)

    # أوامر محلية لا تحتاج استدعاء النموذج.
    reset_commands = {
        "/reset",
        "/clear",
        "مسح الذاكرة",
        "امسح الذاكرة",
        "نسيت كلشي",
    }
    if user_message.lower() in reset_commands:
        clear_history(memory_phone)
        return json_success("تم مسح ذاكرة هذه المحادثة. نبدأ من جديد 👍")

    history = get_history(memory_phone)
    is_first_message = len(history) == 0
    is_admin = memory_phone == ADMIN_PHONE

    if is_admin:
        if is_first_message:
            dynamic_context = (
                "\n\n[سياق خاص: المرسل هو المدير حسين. "
                f"التحية المناسبة الآن بتوقيت بغداد هي: {iraqi_time_greeting()}. "
                "هذه بداية المحادثة، فابدأ بتحية عراقية دافئة مرة واحدة فقط.]"
            )
        else:
            dynamic_context = (
                "\n\n[سياق خاص: المرسل هو المدير حسين. "
                "هذه ليست بداية المحادثة؛ لا تعيد التحية تلقائياً، "
                "واستمر في سياق الحوار السابق.]"
            )
    else:
        if is_first_message:
            dynamic_context = (
                "\n\n[هذه بداية محادثة جديدة. يمكنك الترحيب بالمستخدم "
                "إذا كان ذلك مناسباً، بدون إطالة.]"
            )
        else:
            dynamic_context = (
                "\n\n[هذه محادثة مستمرة. لا تبدأ من الصفر ولا تكرر التحية؛ "
                "استخدم سياق الرسائل السابقة.]"
            )

    conversation_for_model = history + [
        {"role": "user", "content": user_message}
    ]

    bot_reply = call_groq(
        api_key=api_key,
        conversation_messages=conversation_for_model,
        system_content=LAITH_SYSTEM_PROMPT + dynamic_context,
        user_phone=memory_phone,
    )

    # نخزن فقط بعد نجاح توليد الرد.
    save_message(memory_phone, "user", user_message)
    save_message(memory_phone, "assistant", bot_reply)

    return json_success(bot_reply)


@app.errorhandler(APIError)
def handle_api_error(exc: APIError):
    return json_error(exc.message, exc.status_code, exc.error_code)


@app.errorhandler(HTTPException)
def handle_http_exception(exc: HTTPException):
    return json_error(
        exc.description or "حدث خطأ في الطلب.",
        exc.code or 500,
        "http_error",
    )


@app.errorhandler(Exception)
def handle_unexpected_error(exc: Exception):
    logger.exception("خطأ غير متوقع: %s", type(exc).__name__)
    return json_error(
        "حدث خطأ داخلي غير متوقع.",
        500,
        "internal_server_error",
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)

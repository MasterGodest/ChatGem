import os
import datetime
from typing import Dict, Any, List

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from google import genai


# ========= КОНФИГ ИЗ ПЕРЕМЕННЫХ ОКРУЖЕНИЯ =========

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# список id админов: ADMIN_IDS="12345,67890"
_admin_ids_str = os.getenv("ADMIN_IDS", "")
ADMIN_IDS = {
    int(x.strip())
    for x in _admin_ids_str.split(",")
    if x.strip().isdigit()
}

DEFAULT_DAILY_LIMIT = 100
MAX_HISTORY_MESSAGES = 20

# ===== ТОЛЬКО МОДЕЛИ GEMINI =====
MODEL_OPTIONS = {
    "gemini": [
        ("gemini-1.5-flash", "Gemini 1.5 Flash"),
        ("gemini-1.5-pro", "Gemini 1.5 Pro"),
    ],
}

DEFAULT_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"


# ========= ИНИЦИАЛИЗАЦИЯ =========

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Не задан TELEGRAM_TOKEN")

if not GEMINI_API_KEY:
    raise RuntimeError("Не задан GEMINI_API_KEY")

genai_client = genai.Client(api_key=GEMINI_API_KEY)


# ========= ПАМЯТЬ =========

user_state: Dict[int, Dict[str, Any]] = {}
user_limits: Dict[int, Dict[str, Any]] = {}
stats: Dict[str, Any] = {
    "total_users": set(),
    "total_messages": 0,
}


# ========= ВСПОМОГАТЕЛЬНЫЕ =========

def get_today_str() -> str:
    return datetime.date.today().isoformat()

def get_user_limit_info(uid: int):
    today = get_today_str()
    info = user_limits.get(uid)
    if not info or info["date"] != today:
        info = {"date": today, "used": 0, "limit": DEFAULT_DAILY_LIMIT}
        user_limits[uid] = info
    return info

def inc_user_usage(uid: int, amount: int = 1):
    info = get_user_limit_info(uid)
    info["used"] += amount

def get_user_state(uid: int):
    if uid not in user_state:
        user_state[uid] = {
            "provider": "gemini",
            "model": DEFAULT_GEMINI_MODEL,
            "history": [],
            "awaiting_image": False,
        }
    return user_state[uid]

def reset_user_history(uid: int):
    user_state[uid]["history"] = []

def is_admin(uid: int):
    return uid in ADMIN_IDS


# ========= КНОПКИ =========

def build_main_keyboard(is_admin_user):
    kb = [
        ["🧠 Выбрать модель", "🆕 Новая сессия"],
        ["ℹ️ Моя информация", "❓ Помощь"],
    ]
    if is_admin_user:
        kb.append(["👑 Админ"])
    return ReplyKeyboardMarkup(kb, resize_keyboard=True)

def build_models_keyboard():
    rows = []
    rows.append([InlineKeyboardButton("✨ Модели Gemini", callback_data="noop")])

    for name, label in MODEL_OPTIONS["gemini"]:
        rows.append(
            [InlineKeyboardButton(label, callback_data=f"gemini|{name}")]
        )

    return InlineKeyboardMarkup(rows)


# ========= ВЫЗОВ GEMINI =========

async def call_gemini_chat(uid: int, text: str, model: str) -> str:
    state = get_user_state(uid)
    history = state["history"]

    lines = []
    for msg in history:
        if msg["role"] == "user":
            lines.append("Пользователь: " + msg["content"])
        else:
            lines.append("Ассистент: " + msg["content"])
    lines.append("Пользователь: " + text)

    prompt = "\n".join(lines)

    resp = genai_client.models.generate_content(
        model=model,
        contents=prompt
    )

    answer = resp.text

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": answer})

    if len(history) > MAX_HISTORY_MESSAGES:
        history[:] = history[-MAX_HISTORY_MESSAGES:]

    return answer


# ========= КОМАНДЫ =========

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    stats["total_users"].add(uid)

    await update.message.reply_text(
        "Привет! 👋\n\n"
        "Я ИИ-бот только на Gemini! 🔥\n\n"
        "Работаю с моделями:\n"
        "• Gemini 1.5 Flash\n"
        "• Gemini 1.5 Pro\n\n"
        "Используй кнопки ниже 😊",
        reply_markup=build_main_keyboard(is_admin(uid))
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start_cmd(update, context)

async def models_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Выбери модель Gemini:",
        reply_markup=build_models_keyboard()
    )

async def new_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reset_user_history(update.effective_user.id)
    await update.message.reply_text("🧹 История очищена!")

async def me_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    st = get_user_state(uid)
    info = get_user_limit_info(uid)

    await update.message.reply_text(
        f"ID: {uid}\n"
        f"Модель: {st['model']}\n"
        f"Лимит: {info['used']} / {info['limit']}"
    )

async def admin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Нет доступа.")
        return

    await update.message.reply_text(
        f"👑 Админ панель\n\n"
        f"Пользователей: {len(stats['total_users'])}\n"
        f"Сообщений: {stats['total_messages']}"
    )


# ========= ИНЛАЙН-КНОПКИ =========

async def model_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if q.data.startswith("noop"):
        await q.answer()
        return

    provider, model = q.data.split("|")
    st = get_user_state(q.from_user.id)

    st["provider"] = provider
    st["model"] = model

    await q.answer()
    await q.edit_message_text(
        f"Модель переключена!\n"
        f"Провайдер: Gemini\n"
        f"Модель: {model}\n\n"
        f"Пиши сообщение 🙂"
    )


# ========= ТЕКСТ =========

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.text:
        return

    uid = msg.from_user.id
    text = msg.text

    if text == "🧠 Выбрать модель":
        await msg.reply_text("Выбери модель:", reply_markup=build_models_keyboard())
        return

    if text == "🆕 Новая сессия":
        reset_user_history(uid)
        await msg.reply_text("Готово! История очищена.")
        return

    if text == "ℹ️ Моя информация":
        await me_cmd(update, context)
        return

    if text == "❓ Помощь":
        await help_cmd(update, context)
        return

    if text == "👑 Админ" and is_admin(uid):
        await admin_cmd(update, context)
        return

    # ПРОСТО ОТВЕТ GEMINI
    info = get_user_limit_info(uid)
    if info["used"] >= info["limit"]:
        await msg.reply_text("🚫 Лимит исчерпан. Попробуй завтра.")
        return

    st = get_user_state(uid)

    try:
        answer = await call_gemini_chat(uid, text, st["model"])
        inc_user_usage(uid)
        stats["total_messages"] += 1
    except Exception as e:
        answer = f"Ошибка при обращении к Gemini: {e}"

    await msg.reply_text(answer)


# ========= MAIN =========

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("models", models_cmd))
    app.add_handler(CommandHandler("new", new_cmd))
    app.add_handler(CommandHandler("me", me_cmd))
    app.add_handler(CommandHandler("admin", admin_cmd))

    app.add_handler(CallbackQueryHandler(model_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    print("BOT STARTED (GEMINI ONLY)")
    app.run_polling()


if __name__ == "__main__":
    main()

import os
import datetime
from typing import Dict, Any
from fastapi import FastAPI, Request
import uvicorn

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

from google import genai


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # https://твой-app.up.railway.app/webhook

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN not set")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

if not WEBHOOK_URL:
    raise RuntimeError("WEBHOOK_URL not set (your Railway domain)")


genai_client = genai.Client(api_key=GEMINI_API_KEY)

app_fastapi = FastAPI()

user_state: Dict[int, Dict[str, Any]] = {}
user_limits: Dict[int, Dict[str, Any]] = {}
stats = {
    "total_users": set(),
    "total_messages": 0,
}

MODEL_OPTIONS = {
    "gemini": [
        ("gemini-1.5-flash", "Gemini 1.5 Flash"),
        ("gemini-1.5-pro", "Gemini 1.5 Pro"),
    ]
}

DEFAULT_MODEL = "gemini-1.5-flash"
MAX_HISTORY = 20
DAILY_LIMIT = 100


def get_today():
    return datetime.date.today().isoformat()


def get_user_state(uid: int):
    if uid not in user_state:
        user_state[uid] = {
            "model": DEFAULT_MODEL,
            "history": []
        }
    return user_state[uid]


def get_user_limit(uid: int):
    today = get_today()
    if uid not in user_limits or user_limits[uid]["date"] != today:
        user_limits[uid] = {
            "date": today,
            "used": 0,
            "limit": DAILY_LIMIT,
        }
    return user_limits[uid]


async def call_gemini(uid: int, text: str, model: str) -> str:
    st = get_user_state(uid)
    history = st["history"]

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
    if len(history) > MAX_HISTORY:
        history[:] = history[-MAX_HISTORY:]

    return answer


# Telegram Application
application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()


def keyboard_main():
    return ReplyKeyboardMarkup(
        [["🧠 Выбрать модель", "🆕 Новая сессия"],
         ["ℹ️ Моя информация", "❓ Помощь"]],
        resize_keyboard=True
    )


def keyboard_models():
    rows = []
    for name, label in MODEL_OPTIONS["gemini"]:
        rows.append([InlineKeyboardButton(label, callback_data=f"gemini|{name}")])
    return InlineKeyboardMarkup(rows)


# Telegram handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    stats["total_users"].add(uid)

    await update.message.reply_text(
        "Привет! Я бот на Gemini 🚀\n\n"
        "Модели:\n"
        "• Gemini 1.5 Flash\n"
        "• Gemini 1.5 Pro\n\n"
        "Используй кнопки ниже.",
        reply_markup=keyboard_main()
    )


async def cmd_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Выбери модель:", reply_markup=keyboard_models())


async def new_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    st = get_user_state(update.effective_user.id)
    st["history"] = []
    await update.message.reply_text("🧹 История очищена!")


async def me(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    st = get_user_state(uid)
    lim = get_user_limit(uid)

    await update.message.reply_text(
        f"Модель: {st['model']}\n"
        f"Сообщения: {lim['used']} / {lim['limit']}"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)


async def callback_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    _, model = q.data.split("|")

    st = get_user_state(q.from_user.id)
    st["model"] = model

    await q.answer()
    await q.edit_message_text(
        f"Модель переключена!\nИспользую: {model}\n\nПиши сообщение."
    )


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    uid = msg.from_user.id

    txt = msg.text

    if txt == "🧠 Выбрать модель":
        await msg.reply_text("Выбери модель:", reply_markup=keyboard_models())
        return

    if txt == "🆕 Новая сессия":
        await new_session(update, context)
        return

    if txt == "ℹ️ Моя информация":
        await me(update, context)
        return

    if txt == "❓ Помощь":
        await help_cmd(update, context)
        return

    lim = get_user_limit(uid)
    if lim["used"] >= lim["limit"]:
        await msg.reply_text("🚫 Лимит исчерпан. Попробуй завтра.")
        return

    st = get_user_state(uid)
    answer = await call_gemini(uid, txt, st["model"])
    lim["used"] += 1
    stats["total_messages"] += 1

    await msg.reply_text(answer)


# Add handlers
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("help", help_cmd))
application.add_handler(CommandHandler("models", cmd_models))
application.add_handler(CommandHandler("new", new_session))
application.add_handler(CommandHandler("me", me))

application.add_handler(CallbackQueryHandler(callback_models))
application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_handler))


# ===== FASTAPI WEBHOOK =====

@app_fastapi.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return {"ok": True}


@app_fastapi.on_event("startup")
async def startup():
    await application.bot.set_webhook(url=WEBHOOK_URL)
    print("Webhook set:", WEBHOOK_URL)


# RUN
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app_fastapi, host="0.0.0.0", port=port)

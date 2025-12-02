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

from openai import OpenAI
from google import genai


# ========= КОНФИГ ИЗ ПЕРЕМЕННЫХ ОКРУЖЕНИЯ =========

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# список id админов через запятую: "123,456"
_admin_ids_str = os.getenv("ADMIN_IDS", "")
ADMIN_IDS = {
    int(x.strip())
    for x in _admin_ids_str.split(",")
    if x.strip().isdigit()
}

DEFAULT_DAILY_LIMIT = 100
MAX_HISTORY_MESSAGES = 20

MODEL_OPTIONS = {
    "openai": [
        ("gpt-5.1", "GPT-5.1"),
        ("gpt-5.1-mini", "GPT-5.1 Mini"),
        ("gpt-4.1", "GPT-4.1"),
        ("o3-mini", "o3-mini (reasoning)"),
    ],
    "gemini": [
        ("gemini-3.0-flash", "Gemini 3.0 Flash"),
        ("gemini-3.0-pro", "Gemini 3.0 Pro"),
        ("gemini-2.0-flash", "Gemini 2.0 Flash"),
        ("gemini-1.5-flash", "Gemini 1.5 Flash"),
    ],
}

DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-5.1-mini"
DEFAULT_GEMINI_MODEL = "gemini-3.0-flash"

IMAGE_MODEL = "gpt-image-1"


# ========= КЛИЕНТЫ API =========

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Не задан TELEGRAM_TOKEN в переменных окружения")

if not OPENAI_API_KEY:
    raise RuntimeError("Не задан OPENAI_API_KEY в переменных окружения")

if not GEMINI_API_KEY:
    raise RuntimeError("Не задан GEMINI_API_KEY в переменных окружения")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
genai_client = genai.Client(api_key=GEMINI_API_KEY)


# ========= СОСТОЯНИЕ В ПАМЯТИ =========

user_state: Dict[int, Dict[str, Any]] = {}
user_limits: Dict[int, Dict[str, Any]] = {}
stats: Dict[str, Any] = {
    "total_messages": 0,
    "total_users": set(),
}


# ========= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =========

def provider_human(provider: str) -> str:
    return "OpenAI" if provider == "openai" else "Gemini"


def get_today_str() -> str:
    return datetime.date.today().isoformat()


def get_user_limit_info(user_id: int) -> Dict[str, Any]:
    today = get_today_str()
    info = user_limits.get(user_id)
    if info is None or info.get("date") != today:
        info = {
            "date": today,
            "used": 0,
            "limit": DEFAULT_DAILY_LIMIT,
        }
        user_limits[user_id] = info
    return info


def inc_user_usage(user_id: int, amount: int = 1) -> None:
    info = get_user_limit_info(user_id)
    info["used"] += amount


def get_user_state(user_id: int) -> Dict[str, Any]:
    if user_id not in user_state:
        user_state[user_id] = {
            "provider": DEFAULT_PROVIDER,
            "model": DEFAULT_OPENAI_MODEL,
            "history": [],
            "awaiting_image_prompt": False,
        }
    return user_state[user_id]


def reset_user_history(user_id: int) -> None:
    state = get_user_state(user_id)
    state["history"] = []


def build_main_keyboard(is_admin: bool) -> ReplyKeyboardMarkup:
    keyboard = [
        ["🧠 Выбрать модель", "🆕 Новая сессия"],
        ["🖼 Картинка", "ℹ️ Моя информация"],
        ["❓ Помощь"],
    ]
    if is_admin:
        keyboard.append(["👑 Админ"])
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


def build_models_keyboard() -> InlineKeyboardMarkup:
    keyboard: List[List[InlineKeyboardButton]] = []

    keyboard.append([InlineKeyboardButton("🤖 Модели OpenAI", callback_data="noop")])
    for model_name, label in MODEL_OPTIONS["openai"]:
        keyboard.append([
            InlineKeyboardButton(label, callback_data=f"openai|{model_name}")
        ])

    keyboard.append([InlineKeyboardButton(" ", callback_data="noop2")])

    keyboard.append([InlineKeyboardButton("✨ Модели Gemini", callback_data="noop")])
    for model_name, label in MODEL_OPTIONS["gemini"]:
        keyboard.append([
            InlineKeyboardButton(label, callback_data=f"gemini|{model_name}")
        ])

    return InlineKeyboardMarkup(keyboard)


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


def format_user_info(user_id: int) -> str:
    state = get_user_state(user_id)
    limit_info = get_user_limit_info(user_id)
    return (
        f"ID: {user_id}\n"
        f"Текущая модель: {provider_human(state['provider'])} ({state['model']})\n"
        f"Лимит на сегодня: {limit_info['used']} / {limit_info['limit']} сообщений."
    )


# ========= ВЫЗОВЫ ИИ =========

async def call_openai_chat(user_id: int, user_text: str, model_name: str) -> str:
    state = get_user_state(user_id)
    history = state["history"]

    if not history:
        history.append({
            "role": "system",
            "content": "Ты дружелюбный и полезный ассистент, отвечай по-русски.",
        })

    history.append({"role": "user", "content": user_text})

    resp = openai_client.chat.completions.create(
        model=model_name,
        messages=history,
    )
    answer = resp.choices[0].message.content

    history.append({"role": "assistant", "content": answer})
    if len(history) > MAX_HISTORY_MESSAGES:
        state["history"] = history[-MAX_HISTORY_MESSAGES:]

    return answer


async def call_gemini_chat(user_id: int, user_text: str, model_name: str) -> str:
    state = get_user_state(user_id)
    history = state["history"]

    lines = []
    for msg in history:
        if msg.get("role") == "user":
            lines.append("Пользователь: " + msg.get("content", ""))
        elif msg.get("role") == "assistant":
            lines.append("Ассистент: " + msg.get("content", ""))

    lines.append("Пользователь: " + user_text)
    prompt = "\n".join(lines)

    response = genai_client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    answer = response.text

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})
    if len(history) > MAX_HISTORY_MESSAGES:
        state["history"] = history[-MAX_HISTORY_MESSAGES:]

    return answer


async def generate_image(prompt: str) -> str:
    img = openai_client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    return img.data[0].url


# ========= КОМАНДЫ ПОЛЬЗОВАТЕЛЯ =========

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    stats["total_users"].add(user_id)

    kb = build_main_keyboard(is_admin(user_id))

    text = (
        "Привет! 👋\n\n"
        "Я ИИ-бот в Telegram.\n\n"
        "Могу работать с:\n"
        "• ChatGPT 5.1 / 5.1-mini / 4.1 / o3-mini\n"
        "• Gemini 3.0 / 2.0 / 1.5\n\n"
        "Кнопки снизу помогут управлять мной 🙂"
    )

    await update.message.reply_text(text, reply_markup=kb)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start_cmd(update, context)


async def models_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Выбери модель:",
        reply_markup=build_models_keyboard(),
    )


async def new_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    reset_user_history(user_id)
    await update.message.reply_text("🧹 История очищена. Начинаем заново!")


async def me_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    info = format_user_info(user_id)
    await update.message.reply_text("Твоя информация:\n\n" + info)


async def img_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    limit_info = get_user_limit_info(user_id)
    if limit_info["used"] >= limit_info["limit"]:
        await update.message.reply_text("🚫 Лимит на сегодня исчерпан.")
        return

    if context.args:
        prompt = " ".join(context.args)
    else:
        await update.message.reply_text(
            "Использование: /img кот в космосе\n"
            "Или через кнопку «🖼 Картинка»."
        )
        return

    await update.message.reply_text("🎨 Генерирую картинку...")

    try:
        url = await generate_image(prompt)
        inc_user_usage(user_id, amount=3)
        stats["total_messages"] += 1
        await update.message.reply_photo(
            photo=url,
            caption=f"Готово!\n\nЗапрос: {prompt}",
        )
    except Exception as e:
        await update.message.reply_text(f"Ошибка при генерации изображения: {e}")


# ========= АДМИНКА =========

async def admin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ Эта команда только для админа.")
        return

    total_users = len(stats["total_users"])
    total_messages = stats["total_messages"]

    txt = (
        "👑 Админ-панель\n\n"
        f"Всего пользователей: {total_users}\n"
        f"Всего сообщений: {total_messages}\n\n"
        "Пока тут только статистика 🙂"
    )
    await update.message.reply_text(txt)


# ========= ОБРАБОТКА ИНЛАЙН-КНОПОК =========

async def model_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    user_id = query.from_user.id

    if data.startswith("noop"):
        await query.answer()
        return

    try:
        provider, model_name = data.split("|", 1)
    except ValueError:
        await query.answer("Ошибка данных кнопки.", show_alert=True)
        return

    state = get_user_state(user_id)
    state["provider"] = provider
    state["model"] = model_name

    await query.answer()
    await query.edit_message_text(
        f"✅ Модель переключена!\n"
        f"Провайдер: {provider_human(provider)}\n"
        f"Модель: {model_name}\n\n"
        f"Теперь просто напиши сообщение, и я отвечу этой моделью."
    )


# ========= ОСНОВНОЙ ТЕКСТОВЫЙ ХЕНДЛЕР =========

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message or not message.text:
        return

    user = update.effective_user
    user_id = user.id
    stats["total_users"].add(user_id)

    state = get_user_state(user_id)
    text = message.text.strip()

    # Кнопки
    if text == "🧠 Выбрать модель":
        await message.reply_text(
            "Выбери модель:",
            reply_markup=build_models_keyboard(),
        )
        return

    if text == "🆕 Новая сессия":
        reset_user_history(user_id)
        await message.reply_text("🧹 История очищена!")
        return

    if text == "🖼 Картинка":
        state["awaiting_image_prompt"] = True
        await message.reply_text("Напиши описание картинки.")
        return

    if text == "ℹ️ Моя информация":
        info = format_user_info(user_id)
        await message.reply_text("Твоя информация:\n\n" + info)
        return

    if text == "❓ Помощь":
        await help_cmd(update, context)
        return

    if text == "👑 Админ" and is_admin(user_id):
        await admin_cmd(update, context)
        return

    # Команды
    if text.startswith("/"):
        cmd, *args = text.split()
        args_str = " ".join(args)
        context.args = args  # чтобы /img работала

        if cmd == "/start":
            await start_cmd(update, context)
        elif cmd == "/help":
            await help_cmd(update, context)
        elif cmd == "/models":
            await models_cmd(update, context)
        elif cmd == "/new":
            await new_cmd(update, context)
        elif cmd == "/me":
            await me_cmd(update, context)
        elif cmd == "/img":
            await img_cmd(update, context)
        elif cmd == "/admin":
            await admin_cmd(update, context)
        else:
            await message.reply_text("Неизвестная команда. Напиши /start.")
        return

    # ожидаем описание для картинки
    if state.get("awaiting_image_prompt"):
        state["awaiting_image_prompt"] = False
        await img_cmd(update, context)
        return

    # проверка лимита
    limit_info = get_user_limit_info(user_id)
    if limit_info["used"] >= limit_info["limit"]:
        await message.reply_text(
            "🚫 Ты исчерпал дневной лимит сообщений. Попробуй завтра."
        )
        return

    provider = state["provider"]
    model_name = state["model"]

    await message.chat.send_action("typing")

    try:
        if provider == "openai":
            answer = await call_openai_chat(user_id, text, model_name)
        else:
            answer = await call_gemini_chat(user_id, text, model_name)

        inc_user_usage(user_id)
        stats["total_messages"] += 1
    except Exception as e:
        answer = (
            f"⚠️ Ошибка при обращении к {provider_human(provider)} "
            f"({model_name}): {e}"
        )

    await message.reply_text(answer)


# ========= MAIN =========

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("models", models_cmd))
    app.add_handler(CommandHandler("new", new_cmd))
    app.add_handler(CommandHandler("me", me_cmd))
    app.add_handler(CommandHandler("img", img_cmd))
    app.add_handler(CommandHandler("admin", admin_cmd))

    app.add_handler(CallbackQueryHandler(model_button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    print("Бот запущен.")
    app.run_polling()


if __name__ == "__main__":
    main()

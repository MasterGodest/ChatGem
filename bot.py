import datetime
from typing import Dict, List, Any, Set

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


# ========= КОНФИГ ============

# 👉 ВСТАВЬ СВОЙ ТОКЕН БОТА ОТ BOTFATHER
TELEGRAM_BOT_TOKEN = "TELEGRAM_TOKEN"

# 👉 ВСТАВЬ СВОИ КЛЮЧИ ИИ
OPENAI_API_KEY = "OPENAI_API_KEY"
GEMINI_API_KEY = "GEMINI_API_KEY"

# 👉 ТВОЙ Telegram ID (чтобы работала админка)
ADMIN_IDS: Set[int] = {
    1831731188,  # замени на свой id
}

# Лимиты
DEFAULT_DAILY_LIMIT = 1000          # сообщений в день для обычного пользователя
MAX_HISTORY_MESSAGES = 200          # максимальная длина истории диалога

# Модели
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

# Модель для картинок OpenAI
IMAGE_MODEL = "gpt-image-1"


# ========= КЛИЕНТЫ API ============

openai_client = OpenAI(api_key=OPENAI_API_KEY)
genai_client = genai.Client(api_key=GEMINI_API_KEY)


# ========= СОСТОЯНИЕ В ПАМЯТИ ============

# user_id -> {
#   "provider": "openai"/"gemini",
#   "model": str,
#   "history": [ {role, content}, ... ],
#   "awaiting_image_prompt": bool,
# }
user_state: Dict[int, Dict[str, Any]] = {}

# Лимиты: user_id -> {"date": "YYYY-MM-DD", "used": int, "limit": int}
user_limits: Dict[int, Dict[str, Any]] = {}

# Статистика
stats: Dict[str, Any] = {
    "total_messages": 0,
    "total_users": set(),  # type: ignore
}


# ========= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ============

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


def reset_user_history(user_id: int):
    state = get_user_state(user_id)
    state["history"] = []


def add_to_history(user_id: int, role: str, content: str):
    state = get_user_state(user_id)
    history: List[Dict[str, str]] = state["history"]
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY_MESSAGES:
        state["history"] = history[-MAX_HISTORY_MESSAGES:]


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


async def call_openai_chat(user_id: int, user_text: str, model_name: str) -> str:
    state = get_user_state(user_id)
    history = state["history"]

    if not history:
        history.append({
            "role": "system",
            "content": "Ты дружелюбный и полезный ассистент, отвечай по-русски.",
        })

    history.append({"role": "user", "content": user_text})
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]
        state["history"] = history

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
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            lines.append(f"Пользователь: {content}")
        elif role == "assistant":
            lines.append(f"Ассистент: {content}")

    lines.append(f"Пользователь: {user_text}")
    prompt = "\n".join(lines)

    resp = genai_client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    answer = resp.text

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


# ========= КОМАНДЫ ПОЛЬЗОВАТЕЛЯ ============

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    stats["total_users"].add(user_id)

    kb = build_main_keyboard(is_admin(user_id))
    text = (
        "Привет! 👋\n\n"
        "Я твой ИИ-бот в Telegram.\n\n"
        "Могу работать с:\n"
        "• ChatGPT 5.1 / 5.1-mini / 4.1 / o3-mini\n"
        "• Gemini 3.0 / 2.0 / 1.5\n\n"
        "Используй кнопки снизу 👇\n\n"
        "Команды:\n"
        "/models – выбрать модель\n"
        "/new – новая сессия\n"
        "/img <описание> – картинка\n"
        "/me – информация о твоём аккаунте\n"
        "/help – помощь\n"
    )
    if is_admin(user_id):
        text += "\nАдмин-команда: /admin"

    await update.message.reply_text(text, reply_markup=kb)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start_cmd(update, context)


async def models_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Выбери модель ИИ:",
        reply_markup=build_models_keyboard(),
    )


async def new_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    reset_user_history(user_id)
    await update.message.reply_text("🧹 История диалога очищена. Начинаем заново!")


async def me_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    info = format_user_info(user_id)
    await update.message.reply_text("Твоя информация:\n\n" + info)


async def img_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    limit_info = get_user_limit_info(user_id)
    if limit_info["used"] >= limit_info["limit"]:
        await update.message.reply_text(
            "🚫 Ты исчерпал дневной лимит сообщений. Попробуй завтра."
        )
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


# ========= АДМИНКА ============

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
        f"Всего обработано сообщений: {total_messages}\n\n"
        "Команды:\n"
        "/setlimit <user_id> <лимит> – задать лимит\n"
        "/user <user_id> – инфо о пользователе\n"
        "/broadcast <текст> – рассылка всем\n"
    )
    await update.message.reply_text(txt)


async def setlimit_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ Только админ.")
        return

    if len(context.args) != 2:
        await update.message.reply_text("Использование: /setlimit <user_id> <лимит>")
        return

    try:
        target_id = int(context.args[0])
        new_limit = int(context.args[1])
    except ValueError:
        await update.message.reply_text("user_id и лимит должны быть числами.")
        return

    info = get_user_limit_info(target_id)
    info["limit"] = new_limit
    await update.message.reply_text(
        f"✅ Лимит для {target_id} установлен: {new_limit} сообщений/день."
    )


async def userinfo_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ Только админ.")
        return

    if len(context.args) != 1:
        await update.message.reply_text("Использование: /user <user_id>")
        return

    try:
        target_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("user_id должен быть числом.")
        return

    info = format_user_info(target_id)
    await update.message.reply_text("Информация о пользователе:\n\n" + info)


async def broadcast_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ Только админ.")
        return

    if not context.args:
        await update.message.reply_text("Использование: /broadcast <текст>")
        return

    text = " ".join(context.args)
    sent = 0
    for uid in list(stats["total_users"]):
        try:
            await context.bot.send_message(chat_id=uid, text=text)
            sent += 1
        except Exception:
            pass

    await update.message.reply_text(f"📨 Разослано {sent} пользователям.")


# ========= КНОПКИ (ВЫБОР МОДЕЛИ) ============

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


# ========= ОСНОВНОЙ ТЕКСТОВЫЙ ХЕНДЛЕР ============

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message or not message.text:
        return

    user = update.effective_user
    user_id = user.id
    stats["total_users"].add(user_id)

    state = get_user_state(user_id)
    text = message.text.strip()

    # Сначала обработаем кнопки-меню
    if text == "🧠 Выбрать модель":
        await message.reply_text(
            "Выбери модель:",
            reply_markup=build_models_keyboard(),
        )
        return

    if text == "🆕 Новая сессия":
        reset_user_history(user_id)
        await message.reply_text("🧹 История очищена. Начинаем заново!")
        return

    if text == "🖼 Картинка":
        state["awaiting_image_prompt"] = True
        await message.reply_text(
            "Напиши описание картинки.\n\nПример: кот в космосе, пиксель-арт."
        )
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

    # Если бот ждёт промпт для картинки – используем введённый текст
    if state.get("awaiting_image_prompt"):
        state["awaiting_image_prompt"] = False

        limit_info = get_user_limit_info(user_id)
        if limit_info["used"] >= limit_info["limit"]:
            await message.reply_text(
                "🚫 Ты исчерпал дневной лимит сообщений. Попробуй завтра."
            )
            return

        await message.reply_text("🎨 Генерирую картинку...")
        try:
            url = await generate_image(text)
            inc_user_usage(user_id, amount=3)
            stats["total_messages"] += 1
            await message.reply_photo(
                photo=url,
                caption=f"Готово!\n\nЗапрос: {text}",
            )
        except Exception as e:
            await message.reply_text(f"Ошибка при генерации изображения: {e}")
        return

    # Дальше — обычный диалог с ИИ

    limit_info = get_user_limit_info(user_id)
    if limit_info["used"] >= limit_info["limit"]:
        await message.reply_text(
            "🚫 Ты исчерпал дневной лимит сообщений. Попробуй завтра.\n"
            "Если нужно больше — попроси админа увеличить лимит."
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

        inc_user_usage(user_id, amount=1)
        stats["total_messages"] += 1
    except Exception as e:
        answer = f"⚠️ Ошибка при обращении к {provider_human(provider)} ({model_name}): {e}"

    await message.reply_text(answer)


# ========= MAIN ============

def main():
    if TELEGRAM_BOT_TOKEN.startswith("СЮДА_ВСТАВЬ"):
        raise RuntimeError("Не забудь вставить TELEGRAM_BOT_TOKEN и ключи API.")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Пользовательские команды
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("models", models_cmd))
    app.add_handler(CommandHandler("new", new_cmd))
    app.add_handler(CommandHandler("me", me_cmd))
    app.add_handler(CommandHandler("img", img_cmd))

    # Админ-команды
    app.add_handler(CommandHandler("admin", admin_cmd))
    app.add_handler(CommandHandler("setlimit", setlimit_cmd))
    app.add_handler(CommandHandler("user", userinfo_cmd))
    app.add_handler(CommandHandler("broadcast", broadcast_cmd))

    # Кнопки выбора модели
    app.add_handler(CallbackQueryHandler(model_button_handler))

    # Любой текст
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    print("Бот запущен. Нажми Ctrl+C для остановки.")
    app.run_polling()


if __name__ == "__main__":
    main()

import os
import json
import re
from datetime import datetime

import requests
import streamlit as st

from config import (
    PLAYLISTS,
    APP_CONFIG,
    DEEPSEEK_CONFIG,
    UI_CONFIG,
    SUPABASE_URL,
    SUPABASE_ANON_KEY,
)
from utils import (
    compare_answers,
    calculate_score,
    generate_progress_report,
    get_subject_emoji,
    SessionManager,
    create_progress_chart_data,
    log_user_action,
)

# ─────────────────────────────────────────────
# set_page_config — ДОЛЖЕН быть первым вызовом
# ─────────────────────────────────────────────
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["initial_sidebar_state"],
)

# === РЕЗОЛВИМ КЛЮЧИ ===
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
try:
    if (not YOUTUBE_API_KEY) and hasattr(st, "secrets") and "YOUTUBE_API_KEY" in st.secrets:
        YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    if (not DEEPSEEK_API_KEY) and hasattr(st, "secrets") and "DEEPSEEK_API_KEY" in st.secrets:
        DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    pass

if not YOUTUBE_API_KEY:
    st.error("Не задан YOUTUBE_API_KEY. Укажи его в .env или в Secrets.")
    st.stop()

DEEPSEEK_ENABLED = bool(DEEPSEEK_API_KEY)

# MathJax
st.markdown(
    """
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
<script>
    MathJax.Hub.Config({
        tex2jax: { inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']], processEscapes: true }
    });
    MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
</script>
""",
    unsafe_allow_html=True,
)

# CSS
st.markdown(
    """
<style>
.main-header { text-align:center; padding:2rem; background:linear-gradient(90deg,#667eea 0%,#764ba2 100%); border-radius:10px; color:#fff; margin-bottom:2rem; }
.progress-card { background:#fff; padding:1.5rem; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,0.1); margin:1rem 0; }
.task-card { background:#f8f9fa; padding:1.5rem; border-radius:8px; border-left:4px solid #007bff; margin:1rem 0; }
.success-animation { animation:pulse 0.5s ease-in-out; }
@keyframes pulse { 0%{transform:scale(1);} 50%{transform:scale(1.05);} 100%{transform:scale(1);} }
.difficulty-badge { display:inline-block; padding:.3rem .8rem; border-radius:15px; font-size:.75rem; font-weight:600; text-transform:uppercase; margin-bottom:.5rem; }
.easy{ background:#d4edda; color:#155724; } .medium{ background:#fff3cd; color:#856404; } .hard{ background:#f8d7da; color:#721c24; }
.notebook-note{ background:#e9f7ef; padding:1rem; border-radius:8px; margin-bottom:1rem; border-left:4px solid #28a745; }
.badge{ display:inline-block; padding:.25rem .5rem; border-radius:6px; font-size:.75rem; font-weight:600; }
.badge-green{ background:#d1fae5; color:#065f46; } .badge-gray{ background:#e5e7eb; color:#374151; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# Вспомогательные функции для генерации теории
# ─────────────────────────────────────────────
def _fallback_mcq(topic: str, i: int):
    """Простая безопасная заглушка, чтобы добить до нужного количества."""
    return {
        "question": f"Короткий проверочный вопрос #{i + 1} по теме «{topic}». Выберите верный вариант.",
        "options": ["A) Верно", "B) Неверно", "C) Не знаю", "D) Трудно сказать"],
        "correct_answer": "A",
        "explanation": f"Объяснение: базовый факт из темы «{topic}».",
    }


def _normalize_questions(raw: dict, topic: str, n: int):
    """Возвращает РОВНО n вопросов: чистим поля, оставляем только A/B/C/D, добиваем заглушками."""
    qs = (raw or {}).get("questions", [])
    clean = []
    for q in qs:
        question = (q.get("question") or "").strip()
        options = q.get("options") or []
        corr = (q.get("correct_answer") or "").strip()
        expl = (q.get("explanation") or "").strip()
        if not question or len(options) != 4:
            continue
        # корректируем правильный ответ: только буква A/B/C/D
        corr_letter = corr.strip().split(")")[0].strip().upper()
        if corr_letter not in ("A", "B", "C", "D"):
            # попробуем вытащить из текста первого варианта
            corr_letter = "A"
        clean.append(
            {
                "question": question,
                "options": [str(o) for o in options],
                "correct_answer": corr_letter,
                "explanation": expl or "См. разбор по теме.",
            }
        )

    while len(clean) < n:
        clean.append(_fallback_mcq(topic, len(clean)))
    return clean[:n]


def ds_call(payload: dict):
    """Надёжный вызов DeepSeek (общий для всех запросов)."""
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    # коннект 10с, чтение побольше — на генерацию
    timeout = (10, max(25, int(DEEPSEEK_CONFIG.get("timeout", 30))))
    for attempt in range(DEEPSEEK_CONFIG.get("retry_attempts", 3)):
        try:
            r = requests.post(
                "https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload, timeout=timeout
            )
            if r.status_code == 402:
                st.warning("DeepSeek: 402 (недостаточно средств).")
                return {"error": "402"}
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            return {"content": content}
        except requests.exceptions.Timeout:
            if attempt == DEEPSEEK_CONFIG.get("retry_attempts", 3) - 1:
                st.error("Таймаут DeepSeek API.")
                return {"error": "timeout"}
        except requests.exceptions.HTTPError as e:
            if attempt == DEEPSEEK_CONFIG.get("retry_attempts", 3) - 1:
                st.error(f"HTTP ошибка DeepSeek: {e.response.status_code}")
                return {"error": f"http_{e.response.status_code}"}
        except Exception as e:
            if attempt == DEEPSEEK_CONFIG.get("retry_attempts", 3) - 1:
                st.error(f"Ошибка DeepSeek: {str(e)}")
                return {"error": "exception"}


def parse_json_from_text(text: str) -> dict:
    """Аккуратно вытаскиваем JSON даже если модель обернула текстом."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


# ─────────────────────────────────────────────
# Класс тьютора
# ─────────────────────────────────────────────
class EnhancedAITutor:
    def __init__(self):
        self.youtube_api_key = YOUTUBE_API_KEY
        self.playlists = PLAYLISTS
        self.config = APP_CONFIG

    # YouTube
    def get_playlist_videos(self, playlist_id):
        if not (isinstance(playlist_id, str) and playlist_id.startswith("PL")):
            st.error(f"Неверный формат ID плейлиста: {playlist_id}. Ожидается начало 'PL'.")
            log_user_action("invalid_playlist_id", {"playlist_id": playlist_id})
            return []
        url = "https://www.googleapis.com/youtube/v3/playlistItems"
        params = {
            "part": "snippet,contentDetails",
            "playlistId": playlist_id,
            "maxResults": self.config["youtube_max_results"],
            "key": self.youtube_api_key,
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            videos = []
            for item in data.get("items", []):
                sn = item.get("snippet", {}) or {}
                res_id = sn.get("resourceId", {}) or {}
                thumbs = sn.get("thumbnails", {}) or {}
                thumb = thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}
                vid = res_id.get("videoId")
                if not vid:
                    continue
                videos.append(
                    {
                        "title": sn.get("title", "Без названия"),
                        "video_id": vid,
                        "description": (sn.get("description") or ""),
                        "thumbnail": thumb.get("url", ""),
                        "published_at": sn.get("publishedAt", ""),
                    }
                )
            log_user_action("playlist_loaded", {"count": len(videos), "playlist_id": playlist_id})
            return videos
        except requests.exceptions.Timeout:
            st.error("Превышено время ожидания ответа от YouTube API")
            log_user_action("playlist_error", {"error": "timeout", "playlist_id": playlist_id})
            return []
        except requests.exceptions.HTTPError as e:
            st.error(f"Ошибка HTTP при загрузке плейлиста: {e.response.status_code}")
            log_user_action("playlist_error", {"error": str(e), "playlist_id": playlist_id})
            return []
        except Exception as e:
            st.error(f"Ошибка при загрузке видео: {str(e)}")
            log_user_action("playlist_error", {"error": str(e), "playlist_id": playlist_id})
            return []

    # Теория — РОВНО N вопросов, строго по теме/классу
    def generate_theory_questions(self, topic, subject, grade):
        n = int(APP_CONFIG.get("theory_questions_count", 10))
        if not DEEPSEEK_ENABLED:
            return {"questions": _normalize_questions({}, topic, n)}  # только заглушки

        sys_msg = (
            "Ты генератор учебных материалов. Строго придерживайся заданной ТЕМЫ и КЛАССА. "
            "Никаких тем из других классов. Формат только валидный JSON. Без комментариев и без '...'."
        )
        prompt = f"""
Сгенерируй РОВНО {n} тестовых вопросов по теме «{topic}» для {grade}-го класса по предмету «{subject}».

Требования:
- Строго соответствуй теме «{topic}» и {grade}-му классу.
- Каждый вопрос: 4 варианта (строго в виде строк "A) ...", "B) ...", "C) ...", "D) ...").
- Один правильный ответ — только буква A/B/C/D.
- Краткое объяснение причины верного ответа.
- Возвращай строго валидный JSON (без комментариев и многоточий) следующей формы:

{{
  "questions": [
    {{
      "question": "Вопрос по теме «{topic}».",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct_answer": "A",
      "explanation": "Короткое объяснение."
    }}
  ]
}}
"""
        payload = {
            "model": DEEPSEEK_CONFIG["model"],
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ],
            "temperature": DEEPSEEK_CONFIG["temperature"],
            "max_tokens": DEEPSEEK_CONFIG["max_tokens"],
        }
        res = ds_call(payload)
        if "error" in res:
            return {"questions": _normalize_questions({}, topic, n)}

        parsed = parse_json_from_text(res["content"])
        questions = _normalize_questions(parsed, topic, n)
        # если пришло меньше — догенерируем до 2 раз
        attempts = 0
        while len(questions) < n and attempts < 2 and DEEPSEEK_ENABLED:
            missing = n - len(questions)
            add_prompt = f"""
Добавь ещё {missing} вопросов по теме «{topic}» для {grade}-го класса тем же JSON-форматом:
{{ "questions": [ ... ] }}. Никаких повторов, строго по теме/классу.
"""
            payload2 = {
                "model": DEEPSEEK_CONFIG["model"],
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": add_prompt},
                ],
                "temperature": DEEPSEEK_CONFIG["temperature"],
                "max_tokens": DEEPSEEK_CONFIG["max_tokens"],
            }
            res2 = ds_call(payload2)
            parsed2 = parse_json_from_text(res2.get("content", "")) if "error" not in res2 else {}
            extra = _normalize_questions(parsed2, topic, missing)
            # склеиваем и снова нормализуем (на случай странностей)
            merged = {"questions": questions + extra}
            questions = _normalize_questions(merged, topic, n)
            attempts += 1

        return {"questions": questions}

    # Практика
    def generate_practice_tasks(self, topic, subject, grade, user_performance=None):
        if not DEEPSEEK_ENABLED:
            return {"easy": [], "medium": [], "hard": []}

        adjust = ""
        if user_performance is not None:
            if user_performance < 60:
                adjust = "Сделай акцент на более простые задания с подробными объяснениями."
            elif user_performance > 85:
                adjust = "Добавь больше нестандартных и сложных задач."

        tconf = APP_CONFIG["tasks_per_difficulty"]
        prompt = f"""
Составь практические задания по теме «{topic}» для {grade}-го класса по предмету «{subject}».

- {tconf["easy"]} лёгких (базовый уровень),
- {tconf["medium"]} средних,
- {tconf["hard"]} сложных.

{adjust}

Для каждой задачи верни:
- "question": условие (формулы в LaTeX допустимы),
- "answer": правильный ответ (текст/число; только символы, без LaTeX),
- "solution": краткое пошаговое решение (можно с LaTeX),
- "hint": короткая подсказка без LaTeX.

Верни строго валидный JSON (без комментариев/многоточий) вида:
{{
  "easy": [{{"question":"...","answer":"...","solution":"...","hint":"..."}}], 
  "medium": [...],
  "hard": [...]
}}
"""
        payload = {
            "model": DEEPSEEK_CONFIG["model"],
            "messages": [
                {"role": "system", "content": "Генерируй строго по теме и классу. Формат — только валидный JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": DEEPSEEK_CONFIG["temperature"],
            "max_tokens": DEEPSEEK_CONFIG["max_tokens"],
        }
        res = ds_call(payload)
        if "error" in res:
            return {"easy": [], "medium": [], "hard": []}
        parsed = parse_json_from_text(res["content"]) or {}
        # минимальная нормализация
        for key in ("easy", "medium", "hard"):
            parsed[key] = parsed.get(key) or []
        return parsed

    def get_hint(self, question, user_answer, correct_answer):
        if not DEEPSEEK_ENABLED:
            return "Подумайте ещё раз: сравните ваш ответ с условиями задачи."
        prompt = f"""
Задача: "{question}"
Правильный ответ: "{correct_answer}"
Ответ студента: "{user_answer}"

Дай краткую подсказку (1–2 предложения) без LaTeX, чтобы навести на правильное решение, но не раскрывай его.
"""
        payload = {
            "model": DEEPSEEK_CONFIG["model"],
            "messages": [
                {"role": "system", "content": "Отвечай коротко и по делу, без LaTeX."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.5,
            "max_tokens": 300,
        }
        res = ds_call(payload)
        if "error" in res:
            return "Попробуйте проанализировать условие ещё раз и выделить ключевые элементы."
        return res.get("content", "Подумайте про ключевые свойства и определения по теме.")


# ─────────────────────────────────────────────
# Основной UI
# ─────────────────────────────────────────────
def main():
    st.markdown('<div class="main-header"><h1>📚 AI Тьютор — персональное обучение</h1></div>', unsafe_allow_html=True)

    # user / supabase
    st.sidebar.markdown("### 👤 Пользователь")
    user_id = st.sidebar.text_input("Идентификатор (для облака)", placeholder="например, email или ник")
    sb_on = bool(
        (SUPABASE_URL or (hasattr(st, "secrets") and st.secrets.get("SUPABASE_URL")))
        and (SUPABASE_ANON_KEY or (hasattr(st, "secrets") and st.secrets.get("SUPABASE_ANON_KEY")))
    )
    if user_id and sb_on:
        st.sidebar.markdown('<span class="badge badge-green">Supabase: подключено</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="badge badge-gray">Supabase: локальное хранение</span>', unsafe_allow_html=True)

    tutor = EnhancedAITutor()
    session = SessionManager(user_id=user_id if user_id else None)

    # Курс
    st.sidebar.header("📖 Выбор курса")
    subjects = list(tutor.playlists.keys())
    selected_subject = st.sidebar.selectbox("Предмет:", subjects, format_func=lambda x: f"{get_subject_emoji(x)} {x}")
    if selected_subject:
        grades = list(tutor.playlists[selected_subject].keys())
        selected_grade = st.sidebar.selectbox("Класс:", grades)
        if selected_grade:
            session.set_course(selected_subject, selected_grade)
            playlist_id = tutor.playlists[selected_subject][selected_grade]
            if st.sidebar.button("Начать обучение", type="primary"):
                with st.spinner("Загрузка видео из плейлиста..."):
                    videos = tutor.get_playlist_videos(playlist_id)
                    if videos:
                        session.start_course(videos)
                        st.success(f"Загружено {len(videos)} видео")
                        st.rerun()
                    else:
                        st.error("Не удалось загрузить видео из плейлиста")

    # Прогресс
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Ваш прогресс")
    progress_data = session.get_progress()
    st.sidebar.metric("Пройдено тем", len(progress_data["completed_topics"]))
    chart_data = create_progress_chart_data(progress_data)
    if chart_data:
        st.sidebar.plotly_chart(chart_data, use_container_width=True)

    # Роутинг
    stage = session.get_stage()
    if stage == "video":
        display_video_content(tutor, session)
    elif stage == "theory_test":
        show_theory_test(tutor, session)
    elif stage == "practice":
        show_practice_stage(tutor, session)
    else:
        st.info("👆 Выберите предмет и класс в боковой панели и нажмите «Начать обучение»")


def display_video_content(tutor: EnhancedAITutor, session: SessionManager):
    videos = session.get_videos()
    if not videos:
        st.warning("Видео из плейлиста не загружены. Попробуйте перезагрузить страницу.")
        return
    current_video = videos[session.get_current_video_index()]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"📺 {current_video['title']}")
        st.video(f"https://www.youtube.com/watch?v={current_video['video_id']}")
        if current_video["description"]:
            with st.expander("Описание урока"):
                st.write(current_video["description"])

    with col2:
        st.markdown('<div class="progress-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Текущий урок")
        st.info(f"Урок {session.get_current_video_index() + 1} из {len(videos)}")
        st.progress((session.get_current_video_index() + 1) / len(videos))
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Готов к тесту", type="primary"):
                session.set_stage("theory_test")
                log_user_action("start_theory_test", {"video": current_video["title"]})
                st.rerun()
        with col_btn2:
            if st.button("Пересмотреть"):
                log_user_action("rewatch_video", {"video": current_video["title"]})
                st.rerun()
        if session.get_current_video_index() > 0 and st.button("← Предыдущий урок"):
            session.prev_video()
            log_user_action("previous_video", {"video_index": session.get_current_video_index()})
            st.rerun()
        if session.get_current_video_index() < len(videos) - 1 and st.button("Следующий урок →"):
            session.next_video()
            log_user_action("next_video", {"video_index": session.get_current_video_index()})
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def show_theory_test(tutor: EnhancedAITutor, session: SessionManager):
    current_video = session.get_videos()[session.get_current_video_index()]
    st.subheader("📝 Тест по теории")
    st.info(f"Тема: {current_video['title']}")
    topic = current_video["title"]

    if "theory_questions" not in st.session_state:
        with st.spinner("Генерация вопросов..."):
            data = tutor.generate_theory_questions(topic, session.get_subject(), session.get_grade())
            st.session_state.theory_questions = data.get("questions", [])
            st.session_state.theory_answers = {}

    if st.session_state.theory_questions:
        for i, q in enumerate(st.session_state.theory_questions):
            st.markdown('<div class="task-card">', unsafe_allow_html=True)
            st.markdown(f"**Вопрос {i+1}:** {q.get('question','')}", unsafe_allow_html=True)
            options = q.get("options", [])
            answer_key = f"theory_q_{i}"
            selected = st.radio("Выберите ответ:", options, key=answer_key, index=None)
            if selected:
                st.session_state.theory_answers[i] = selected[0].upper()
            st.markdown("</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Вернуться к видео"):
                session.clear_theory_data()
                session.set_stage("video")
                st.rerun()
        with col2:
            if st.button("Проверить ответы", type="primary"):
                if len(st.session_state.theory_answers) == len(st.session_state.theory_questions):
                    show_theory_results(tutor, session)
                else:
                    st.error("Пожалуйста, ответьте на все вопросы")
    else:
        st.error("Не удалось сгенерировать вопросы. Попробуйте снова.")


def show_theory_results(tutor: EnhancedAITutor, session: SessionManager):
    current_video = session.get_videos()[session.get_current_video_index()]
    topic_key = f"{session.get_subject()}_{session.get_grade()}_{current_video['title']}"

    st.markdown('<div class="progress-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Результаты тестирования")

    correct_count = 0
    total = len(st.session_state.theory_questions)
    for i, q in enumerate(st.session_state.theory_questions):
        ua = st.session_state.theory_answers.get(i)
        ca = (q.get("correct_answer") or "").upper()
        if compare_answers(ua, ca):
            correct_count += 1
            st.markdown('<div class="success-animation">', unsafe_allow_html=True)
            st.success(f"Вопрос {i+1}: Правильно!")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error(f"Вопрос {i+1}: Неправильно")
            st.info(f"**Объяснение:** {q.get('explanation','')}", unsafe_allow_html=True)

    score = calculate_score(correct_count, total)
    st.metric("Ваш результат", f"{correct_count}/{total} ({score:.0f}%)")
    session.save_theory_score(topic_key, score)

    if score < APP_CONFIG["theory_pass_threshold"]:
        st.warning("Рекомендуем пересмотреть видео для лучшего понимания темы")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Пересмотреть урок"):
            session.clear_theory_data()
            session.set_stage("video")
            st.rerun()
    with col2:
        if st.button("Начать практику", type="primary"):
            session.clear_theory_data()
            session.set_stage("practice")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def show_practice_stage(tutor: EnhancedAITutor, session: SessionManager):
    current_video = session.get_videos()[session.get_current_video_index()]
    topic = current_video["title"]

    st.subheader("💪 Практические задания")
    st.info(f"Тема: {topic}")
    st.markdown(
        """
    <div class="notebook-note">
        📝 <b>Совет:</b> Для сложных задач используйте тетрадь. Ответ вводите в точном формате.
        Для неравенств — <code>x >= 2</code> или <code>[2, inf)</code>. Для нескольких условий — <code>and</code> или <code>,</code>.
    </div>
    """,
        unsafe_allow_html=True,
    )

    if "practice_tasks" not in st.session_state:
        with st.spinner("Генерация заданий..."):
            theory_score = session.get_theory_score(topic)
            data = tutor.generate_practice_tasks(topic, session.get_subject(), session.get_grade(), theory_score)
            st.session_state.practice_tasks = data
            st.session_state.task_attempts = {}
            st.session_state.completed_tasks = []
            st.session_state.current_task_type = "easy"
            st.session_state.current_task_index = 0

    if any(len(st.session_state.practice_tasks.get(t, [])) for t in ["easy", "medium", "hard"]):
        show_current_task(tutor, session)
    else:
        st.error("Нет заданий. Попробуйте позже или пополните баланс DeepSeek.")


def show_current_task(tutor: EnhancedAITutor, session: SessionManager):
    task_types = ["easy", "medium", "hard"]
    ttype = st.session_state.current_task_type
    idx = st.session_state.current_task_index
    tasks = st.session_state.practice_tasks.get(ttype, [])

    if idx >= len(tasks):
        pos = task_types.index(ttype)
        if pos < len(task_types) - 1:
            st.session_state.current_task_type = task_types[pos + 1]
            st.session_state.current_task_index = 0
            st.rerun()
        else:
            show_practice_completion(tutor, session)
            return

    task = tasks[idx]
    task_key = f"{ttype}_{idx}"
    total = sum(len(st.session_state.practice_tasks.get(t, [])) for t in task_types)
    done = len(st.session_state.completed_tasks)

    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown('<div class="progress-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Прогресс")
        st.progress(done / total if total else 0)
        st.metric("Выполнено", f"{done}/{total}")
        names = UI_CONFIG["task_type_names"]
        st.markdown(f'<span class="difficulty-badge {ttype}">{names[ttype]}</span>', unsafe_allow_html=True)
        st.markdown(f"**Задание:** {idx + 1} из {len(tasks)}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col1:
        st.markdown(
            f'<div class="task-card"><span class="difficulty-badge {ttype}">{UI_CONFIG["task_type_names"][ttype]}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"### Задание {idx + 1}")
        st.markdown(task.get("question", ""), unsafe_allow_html=True)

        user_answer = st.text_input("Ваш ответ:", key=f"answer_{task_key}")
        attempts = st.session_state.task_attempts.get(task_key, 0)
        max_attempts = APP_CONFIG["max_attempts_per_task"]

        if attempts < max_attempts:
            col_check, col_skip = st.columns(2)
            with col_check:
                if st.button("Проверить ответ", type="primary"):
                    if user_answer.strip():
                        check_answer(tutor, session, task, user_answer, task_key)
                    else:
                        st.error("Введите ответ!")
            with col_skip:
                if st.button("Пропустить"):
                    log_user_action("skip_task", {"task_key": task_key})
                    move_to_next_task()
        else:
            st.error(f"Исчерпаны все попытки ({max_attempts})")
            st.info(f"**Правильный ответ:** {task.get('answer','')}", unsafe_allow_html=True)
            st.info(f"**Решение:** {task.get('solution','')}", unsafe_allow_html=True)
            if st.button("Следующее задание"):
                move_to_next_task()

        if task_key in st.session_state and "hints" in st.session_state[task_key]:
            st.markdown("### 💡 Подсказки:")
            for hint in st.session_state[task_key]["hints"]:
                st.info(hint)
        st.markdown("</div>", unsafe_allow_html=True)


def check_answer(tutor: EnhancedAITutor, session: SessionManager, task: dict, user_answer: str, task_key: str):
    st.session_state.task_attempts[task_key] = st.session_state.task_attempts.get(task_key, 0) + 1
    attempts = st.session_state.task_attempts[task_key]
    max_attempts = APP_CONFIG["max_attempts_per_task"]

    is_correct = compare_answers((user_answer or "").strip().lower(), (task.get("answer") or "").strip().lower())
    if is_correct:
        st.markdown('<div class="success-animation">', unsafe_allow_html=True)
        st.success("Правильно! Отличная работа.")
        st.markdown("</div>", unsafe_allow_html=True)
        if task_key not in st.session_state.completed_tasks:
            st.session_state.completed_tasks.append(task_key)
        log_user_action("correct_answer", {"task_key": task_key, "attempts": attempts})
        if st.button("Следующее задание"):
            move_to_next_task()
    else:
        if attempts < max_attempts:
            st.error(f"Неправильно. Попытка {attempts} из {max_attempts}")
            with st.spinner("Получаю подсказку..."):
                hint = tutor.get_hint(task.get("question", ""), user_answer, task.get("answer", ""))
                st.session_state.setdefault(task_key, {"hints": []})
                st.session_state[task_key]["hints"].append(hint)
                st.info(f"Подсказка: {hint}")
            log_user_action("incorrect_answer", {"task_key": task_key, "attempts": attempts})
        else:
            st.error("Все попытки исчерпаны.")
            st.info(f"**Правильный ответ:** {task.get('answer','')}", unsafe_allow_html=True)
            st.info(f"**Решение:** {task.get('solution','')}", unsafe_allow_html=True)
            if st.button("Следующее задание"):
                move_to_next_task()


def move_to_next_task():
    st.session_state.current_task_index += 1
    st.rerun()


def show_practice_completion(tutor: EnhancedAITutor, session: SessionManager):
    videos = session.get_videos()
    if not videos:
        st.info("Практика завершена.")
        return
    current_video = videos[session.get_current_video_index()]
    topic_key = f"{session.get_subject()}_{session.get_grade()}_{current_video['title']}"

    st.markdown('<div class="progress-card">', unsafe_allow_html=True)
    st.subheader("Практика завершена!")

    total = sum(len(st.session_state.practice_tasks.get(t, [])) for t in ["easy", "medium", "hard"])
    completed = len(st.session_state.completed_tasks)
    score = calculate_score(completed, total) if total else 0
    st.success(f"Выполнено {completed} из {total} заданий ({score:.0f}%)")

    session.save_practice_score(topic_key, completed, total)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Изучить новую тему"):
            if session.next_video():
                session.set_stage("video")
                for k in ["practice_tasks", "task_attempts", "completed_tasks", "current_task_type", "current_task_index"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
            else:
                st.info("Все темы курса пройдены!")
    with col2:
        if st.button("Вернуться к выбору курса"):
            session.set_stage("selection")
            for k in ["practice_tasks", "task_attempts", "completed_tasks", "current_task_type", "current_task_index"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    st.markdown(generate_progress_report(session.get_progress(), topic_key), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

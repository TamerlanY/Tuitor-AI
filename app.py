# app.py
import os
import json
import time
from datetime import datetime

import requests
import streamlit as st
import plotly.express as px

from config import (
    PLAYLISTS,
    APP_CONFIG,
    DEEPSEEK_CONFIG,
    UI_CONFIG,
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

# ──────────────────────────────────────────────────────────────────────────────
# set_page_config ДОЛЖЕН быть первым вызовом streamlit
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["initial_sidebar_state"],
)
# ──────────────────────────────────────────────────────────────────────────────

# === РЕЗОЛВИМ КЛЮЧИ/НАСТРОЙКИ ПОСЛЕ set_page_config ===
def _get_secret(name: str) -> str | None:
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)

YOUTUBE_API_KEY = _get_secret("YOUTUBE_API_KEY")
DEEPSEEK_API_KEY = _get_secret("DEEPSEEK_API_KEY")

# провайдер LLM: "deepseek" (по умолчанию) или "openai" (ChatGPT/Grok через совместимый API)
LLM_PROVIDER = (_get_secret("LLM_PROVIDER") or "deepseek").lower()

# для OpenAI-совместимых провайдеров (ChatGPT, Grok и т.п.)
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
OPENAI_BASE_URL = _get_secret("OPENAI_BASE_URL")  # можно оставить пустым для официального OpenAI

# DeepSeek может быть пустым, если используем другой провайдер
DEEPSEEK_ENABLED = bool(DEEPSEEK_API_KEY) or (LLM_PROVIDER != "deepseek")

if not YOUTUBE_API_KEY:
    st.error("Не задан YOUTUBE_API_KEY. Укажи его в .env или в Secrets.")
    st.stop()

# Проверка ключей в зависимости от провайдера
if LLM_PROVIDER == "deepseek":
    if not DEEPSEEK_API_KEY:
        st.warning("DEEPSEEK_API_KEY не задан. Генерация вопросов/подсказок может быть недоступна.")
else:
    if not OPENAI_API_KEY:
        st.warning("OPENAI_API_KEY не задан (для LLM_PROVIDER=openai). Генерация может быть недоступна.")

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
  .main-header {
    text-align:center; padding:2rem;
    background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);
    border-radius:10px; color:#fff; margin-bottom:2rem;
  }
  .progress-card { background:#fff; padding:1.5rem; border-radius:10px;
    box-shadow:0 2px 8px rgba(0,0,0,0.1); margin:1rem 0; }
  .task-card { background:#f8f9fa; padding:1.5rem; border-radius:8px;
    border-left:4px solid #007bff; margin:1rem 0; }
  .success-animation { animation:pulse 0.5s ease-in-out; }
  @keyframes pulse { 0%{transform:scale(1);} 50%{transform:scale(1.05);} 100%{transform:scale(1);} }
  .difficulty-badge { display:inline-block; padding:.3rem .8rem; border-radius:15px;
    font-size:.75rem; font-weight:600; text-transform:uppercase; margin-bottom:.5rem; }
  .easy{ background:#d4edda; color:#155724; }
  .medium{ background:#fff3cd; color:#856404; }
  .hard{ background:#f8d7da; color:#721c24; }
  .notebook-note{ background:#e9f7ef; padding:1rem; border-radius:8px; margin-bottom:1rem; border-left:4px solid #28a745; }
  .badge{ display:inline-block; padding:.25rem .5rem; border-radius:6px; font-size:.75rem; font-weight:600; }
  .badge-green{ background:#d1fae5; color:#065f46; } .badge-gray{ background:#e5e7eb; color:#374151; }
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# LLM клиент с backoff
def call_llm(prompt: str) -> dict:
    """
    Возвращает:
      - dict с JSON-ответом, если модель вернула JSON
      - {"content": "..."} если пришёл plain-text
      - {"error": "..."} при ошибке
    """
    # общий таймаут/ретраи
    retry_attempts = DEEPSEEK_CONFIG.get("retry_attempts", 4)
    timeout_s = DEEPSEEK_CONFIG.get("timeout", 60)
    temperature = DEEPSEEK_CONFIG.get("temperature", 0.7)
    max_tokens = DEEPSEEK_CONFIG.get("max_tokens", 1800)
    model = DEEPSEEK_CONFIG.get("model", "deepseek-chat")

    if LLM_PROVIDER == "deepseek":
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        url = "https://api.deepseek.com/v1/chat/completions"

        for attempt in range(retry_attempts):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
                if resp.status_code == 402:
                    st.warning("DeepSeek вернул 402 (недостаточно средств).")
                    return {"error": "402"}
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"content": content}
            except requests.exceptions.Timeout:
                if attempt == retry_attempts - 1:
                    st.error("Превышено время ожидания ответа от DeepSeek API")
                    return {"error": "timeout"}
                time.sleep(0.5 * (2 ** attempt))
            except requests.exceptions.HTTPError as e:
                if attempt == retry_attempts - 1:
                    st.error(f"Ошибка HTTP DeepSeek API: {e.response.status_code}")
                    return {"error": str(e)}
                time.sleep(0.5 * (2 ** attempt))
            except Exception as e:
                if attempt == retry_attempts - 1:
                    st.error(f"Ошибка API DeepSeek: {str(e)}")
                    return {"error": str(e)}
                time.sleep(0.5 * (2 ** attempt))
        return {"error": "unknown"}

    # OpenAI-совместимый провайдер (ChatGPT, Grok через совместимый endpoint и т.п.)
    try:
        from openai import OpenAI  # требует пакет openai
    except Exception:
        return {"error": "openai_sdk_missing"}

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)

    for attempt in range(retry_attempts):
        try:
            resp = client.chat.completions.create(
                model=(os.getenv("LLM_MODEL") or "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            content = resp.choices[0].message.content
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"content": content}
        except Exception as e:
            if attempt == retry_attempts - 1:
                st.error(f"Ошибка LLM провайдера: {e}")
                return {"error": str(e)}
            time.sleep(0.5 * (2 ** attempt))


# ──────────────────────────────────────────────────────────────────────────────
# Генераторы контента
def gen_theory_questions(topic: str, subject: str, grade: str, count: int):
    prompt = f"""
Сгенерируй {count} тестовых вопрос(ов) по теме "{topic}" для {grade}-го класса по предмету "{subject}".
Требования:
- Каждый вопрос проверяет ключевую идею текущей темы.
- 4 варианта ответа: A), B), C), D).
- Ровно один правильный вариант.
- Дай короткое объяснение почему правильный вариант верный.
- Формулы — только в LaTeX, например: \\(x^2+2x+1=0\\).

Верни строго ВАЛИДНЫЙ JSON без комментариев и многоточий:
{{
  "questions": [
    {{
      "question": "Текст вопроса с \\( ... \\) где нужно",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct_answer": "A",
      "explanation": "Краткое объяснение"
    }}
  ]
}}
"""
    return call_llm(prompt)


def gen_practice_tasks(topic: str, subject: str, grade: str, perf: float | None):
    adjustment = ""
    if perf is not None:
        if perf < 60:
            adjustment = "Сделай упор на базу и подробные объяснения."
        elif perf > 85:
            adjustment = "Добавь нестандартные и более сложные задачи."

    e = APP_CONFIG["tasks_per_difficulty"]["easy"]
    m = APP_CONFIG["tasks_per_difficulty"]["medium"]
    h = APP_CONFIG["tasks_per_difficulty"]["hard"]

    prompt = f"""
Составь практические задания по теме "{topic}" для {grade}-го класса по предмету "{subject}":
- {e} лёгких, {m} средних, {h} сложных.

{adjustment}

Для каждой задачи:
- Чёткое условие; формулы — в LaTeX (\\( ... \\)).
- Точный правильный ответ (текст/число; без LaTeX, например: "x >= 2, x < 3").
- Пошаговое решение (с LaTeX).
- Короткую подсказку (без LaTeX, не раскрывающую решение полностью).

Верни строго ВАЛИДНЫЙ JSON без многоточий:
{{
  "easy": [
    {{
      "question": "Условие ... с LaTeX",
      "answer": "Правильный ответ",
      "solution": "Пошаговое решение с LaTeX",
      "hint": "Короткая подсказка"
    }}
  ],
  "medium": [...],
  "hard": [...]
}}
"""
    return call_llm(prompt)


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогалки
def coerce_questions_to_count(qs: list[dict], need: int) -> list[dict]:
    """Если модель дала меньше вопросов — дозаполним валидными заглушками."""
    qs = [q for q in qs if isinstance(q, dict)]
    while len(qs) < need:
        idx = len(qs) + 1
        qs.append(
            {
                "question": f"Вопрос недоступен ({idx}). Нажмите «🔁 Попробовать снова».",
                "options": ["A) —", "B) —", "C) —", "D) —"],
                "correct_answer": "A",
                "explanation": "",
            }
        )
    # если дала больше — аккуратно обрежем
    return qs[:need]


def sanitize_mc_options(options: list) -> list[str]:
    """Гарантируем 4 строки формата 'X) ...'."""
    base = ["A) —", "B) —", "C) —", "D) —"]
    if not isinstance(options, list) or len(options) != 4:
        return base
    fixed = []
    letters = ["A", "B", "C", "D"]
    for i, opt in enumerate(options):
        text = str(opt).strip()
        # если модель не проставила "A) ", добавим
        if not text.startswith(f"{letters[i]})"):
            text = f"{letters[i]}) {text}"
        fixed.append(text)
    return fixed


# ──────────────────────────────────────────────────────────────────────────────
# YouTube + Tutor
class EnhancedAITutor:
    def __init__(self):
        self.youtube_api_key = YOUTUBE_API_KEY
        self.playlists = PLAYLISTS
        self.config = APP_CONFIG
        self.ui_config = UI_CONFIG

    def get_playlist_videos(self, playlist_id: str) -> list[dict]:
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
                rid = sn.get("resourceId", {}) or {}
                thumbs = sn.get("thumbnails", {}) or {}
                thumb = thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}
                vid = rid.get("videoId")
                if not vid:
                    continue
                videos.append(
                    {
                        "title": sn.get("title", "Без названия"),
                        "video_id": vid,
                        "description": (sn.get("description") or "")[:200]
                        + ("..." if len(sn.get("description") or "") > 200 else ""),
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


# ──────────────────────────────────────────────────────────────────────────────
# UI / страницы
def main():
    st.markdown('<div class="main-header"><h1>📚 AI Тьютор — персональное обучение</h1></div>', unsafe_allow_html=True)

    tutor = EnhancedAITutor()
    session = SessionManager()  # если используешь Supabase — оставь как было у тебя

    # Sidebar — выбор курса
    with st.sidebar:
        st.header("📖 Выбор курса")
        subjects = list(tutor.playlists.keys())
        selected_subject = st.selectbox("Предмет:", subjects, format_func=lambda x: f"{get_subject_emoji(x)} {x}")

        if selected_subject:
            grades = list(tutor.playlists[selected_subject].keys())
            selected_grade = st.selectbox("Класс:", grades)

            if selected_grade:
                session.set_course(selected_subject, selected_grade)
                playlist_id = tutor.playlists[selected_subject][selected_grade]

                if st.button("Начать обучение", type="primary"):
                    with st.spinner("Загрузка видео из плейлиста..."):
                        videos = tutor.get_playlist_videos(playlist_id)
                        if videos:
                            session.start_course(videos)
                            st.success(f"Загружено {len(videos)} видео")
                            st.rerun()
                        else:
                            st.error("Не удалось загрузить видео из плейлиста")

        st.markdown("---")
        st.header("📊 Ваш прогресс")
        progress_data = session.get_progress()
        st.metric("Пройдено тем", len(progress_data["completed_topics"]))

        chart = create_progress_chart_data(progress_data)
        if chart:
            st.plotly_chart(chart, use_container_width=True)

    # Роутинг
    stage = session.get_stage()
    if stage == "video":
        display_video_content(session)
    elif stage == "theory_test":
        show_theory_test(session)
    elif stage == "practice":
        show_practice_stage(session)
    else:
        st.info("👆 Выберите предмет и класс в боковой панели, затем нажмите «Начать обучение».")


def display_video_content(session: SessionManager):
    videos = session.get_videos()
    if not videos:
        st.warning("Видео из плейлиста не загружены. Попробуйте перезагрузить страницу.")
        return

    current_video = videos[session.get_current_video_index()]
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header(f"📺 {current_video['title']}")
        st.video(f"https://www.youtube.com/watch?v={current_video['video_id']}")
        if current_video.get("description"):
            with st.expander("Описание урока"):
                st.write(current_video["description"])

    with col2:
        st.markdown('<div class="progress-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Текущий урок")
        st.info(f"Урок {session.get_current_video_index() + 1} из {len(videos)}")
        st.progress((session.get_current_video_index() + 1) / len(videos))

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Готов к тесту", type="primary"):
                session.set_stage("theory_test")
                log_user_action("start_theory_test", {"video": current_video["title"]})
                st.rerun()
        with col_b:
            if st.button("Пересмотреть"):
                log_user_action("rewatch_video", {"video": current_video["title"]})
                st.rerun()

        if session.get_current_video_index() > 0:
            if st.button("← Предыдущий урок"):
                session.prev_video()
                st.rerun()
        if session.get_current_video_index() < len(videos) - 1:
            if st.button("Следующий урок →"):
                session.next_video()
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def show_theory_test(session: SessionManager):
    videos = session.get_videos()
    if not videos:
        st.warning("Нет видео для теста.")
        return

    current_video = videos[session.get_current_video_index()]
    topic = current_video["title"]
    subject = session.get_subject()
    grade = session.get_grade()
    topic_key = f"{subject}_{grade}_{topic}"

    st.header("📝 Теоретический тест")
    st.info(f"Тема: {topic}")

    need_q = int(APP_CONFIG.get("theory_questions_count", 10))

    def _retry():
        if "theory_questions" in st.session_state:
            del st.session_state["theory_questions"]
        if "theory_answers" in st.session_state:
            del st.session_state["theory_answers"]
        st.rerun()

    if "theory_questions" not in st.session_state:
        with st.spinner("Генерация вопросов..."):
            data = gen_theory_questions(topic, subject, grade, need_q)
            # поддержка ошибок
            if isinstance(data, dict) and data.get("error"):
                st.error("Не удалось сгенерировать вопросы. Попробуйте снова.")
                st.button("🔁 Попробовать снова", key="retry_top", on_click=_retry)
                # создаём пустые заглушки, чтобы UI не падал
                qs = coerce_questions_to_count([], need_q)
                st.session_state.theory_questions = qs
                st.session_state.theory_answers = {}
            else:
                raw_qs = (data or {}).get("questions", [])
                # нормализуем опции
                for q in raw_qs:
                    q["options"] = sanitize_mc_options(q.get("options", []))
                    q["question"] = str(q.get("question", "")).strip() or "Вопрос"
                    q["correct_answer"] = (str(q.get("correct_answer", "A")).strip() or "A")[:1].upper()
                    q["explanation"] = str(q.get("explanation", "")).strip()
                qs = coerce_questions_to_count(raw_qs, need_q)
                st.session_state.theory_questions = qs
                st.session_state.theory_answers = {}

    # Если модель прислала меньше и мы дозаполнили — предупредим и дадим retry
    have_real = sum(1 for q in st.session_state.theory_questions if "—" not in "".join(q.get("options", [])))
    if have_real < need_q:
        st.warning("Модель прислала меньше вопросов, чем нужно. Нажмите «Попробовать снова».")
        st.button("🔁 Попробовать снова", key="retry_bottom", on_click=lambda: _retry())

    # Рендер вопросов
    for i, q in enumerate(st.session_state.theory_questions):
        st.markdown('<div class="task-card">', unsafe_allow_html=True)
        st.markdown(f"**Вопрос {i+1}:** {q.get('question','')}", unsafe_allow_html=True)
        answer_key = f"theory_q_{i}"
        selected = st.radio(
            "Выберите ответ:",
            q.get("options", []),
            key=answer_key,
            index=None,
        )
        if selected:
            st.session_state.theory_answers[i] = selected[0]  # первая буква (A/B/C/D)
        st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Вернуться к видео"):
            session.clear_theory_data()
            session.set_stage("video")
            st.rerun()
    with col2:
        if st.button("Проверить ответы", type="primary"):
            if len(st.session_state.theory_answers) != len(st.session_state.theory_questions):
                st.error("Пожалуйста, ответьте на все вопросы.")
            else:
                show_theory_results(session, topic_key)


def show_theory_results(session: SessionManager, topic_key: str):
    qs = st.session_state.theory_questions
    answers = st.session_state.theory_answers

    st.markdown('<div class="progress-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Результаты тестирования")

    correct_count = 0
    for i, q in enumerate(qs):
        got = answers.get(i)
        expected = q.get("correct_answer", "A")
        if compare_answers(got, expected):
            correct_count += 1
            st.markdown('<div class="success-animation">', unsafe_allow_html=True)
            st.success(f"Вопрос {i+1}: Правильно!")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error(f"Вопрос {i+1}: Неправильно")
            exp = q.get("explanation", "")
            if exp:
                st.caption(f"Объяснение: {exp}")

    score = calculate_score(correct_count, len(qs))
    st.metric("Ваш результат", f"{correct_count}/{len(qs)} ({score:.0f}%)")
    session.save_theory_score(topic_key, score)

    pass_bar = APP_CONFIG.get("theory_pass_threshold", 60)
    if score < pass_bar:
        st.warning(f"Проходной порог: {pass_bar}%. Рекомендуем пересмотреть видео.")

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


def show_practice_stage(session: SessionManager):
    videos = session.get_videos()
    if not videos:
        st.warning("Нет видео для практики.")
        return

    current_video = videos[session.get_current_video_index()]
    topic = current_video["title"]
    subject = session.get_subject()
    grade = session.get_grade()
    topic_key = f"{subject}_{grade}_{topic}"

    st.header("💪 Практические задания")
    st.info(f"Тема: {topic}")

    st.markdown(
        """
<div class="notebook-note">
 📝 <b>Совет:</b> Для сложных задач используйте тетрадь.
 Для неравенств — <code>x >= 2</code> или <code>[2, inf)</code>.
 Для нескольких условий — <code>and</code> или <code>,</code>.
</div>
""",
        unsafe_allow_html=True,
    )

    if "practice_tasks" not in st.session_state:
        with st.spinner("Генерация заданий..."):
            theory_score = session.get_theory_score(topic)
            data = gen_practice_tasks(topic, subject, grade, theory_score)
            if isinstance(data, dict) and data.get("error"):
                st.error("Не удалось сгенерировать задания.")
                st.session_state.practice_tasks = {"easy": [], "medium": [], "hard": []}
            else:
                st.session_state.practice_tasks = data or {"easy": [], "medium": [], "hard": []}
            st.session_state.task_attempts = {}
            st.session_state.completed_tasks = []
            st.session_state.current_task_type = "easy"
            st.session_state.current_task_index = 0

    if any(len(st.session_state.practice_tasks.get(t, [])) for t in ["easy", "medium", "hard"]):
        show_current_task(session)
    else:
        st.error("Нет заданий. Попробуйте позже.")


def show_current_task(session: SessionManager):
    tutor_cfg = APP_CONFIG
    task_types = ["easy", "medium", "hard"]
    ttype = st.session_state.current_task_type
    idx = st.session_state.current_task_index
    tasks_of_type = st.session_state.practice_tasks.get(ttype, [])

    if idx >= len(tasks_of_type):
        ti = task_types.index(ttype)
        if ti < len(task_types) - 1:
            st.session_state.current_task_type = task_types[ti + 1]
            st.session_state.current_task_index = 0
            st.rerun()
        else:
            show_practice_completion(session)
            return

    task = tasks_of_type[idx]
    task_key = f"{ttype}_{idx}"

    total = sum(len(st.session_state.practice_tasks.get(t, [])) for t in task_types)
    done = len(st.session_state.completed_tasks)

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown('<div class="progress-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Прогресс")
        st.progress(done / total if total else 0)
        st.metric("Выполнено", f"{done}/{total}")
        label = UI_CONFIG["task_type_names"].get(ttype, ttype)
        st.markdown(f'<span class="difficulty-badge {ttype}">{label}</span>', unsafe_allow_html=True)
        st.markdown(f"**Задание:** {idx+1} из {len(tasks_of_type)}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col1:
        st.markdown(
            f'<div class="task-card"><span class="difficulty-badge {ttype}">{UI_CONFIG["task_type_names"].get(ttype, ttype)}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"### Задание {idx+1}")
        st.markdown(task.get("question", ""), unsafe_allow_html=True)

        user_answer = st.text_input("Ваш ответ:", key=f"answer_{task_key}")
        attempts = st.session_state.task_attempts.get(task_key, 0)
        max_att = tutor_cfg["max_attempts_per_task"]

        if attempts < max_att:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Проверить ответ", type="primary", key=f"check_{task_key}"):
                    check_answer(session, task, user_answer, task_key)
            with col_b:
                if st.button("Пропустить", key=f"skip_{task_key}"):
                    log_user_action("skip_task", {"task_key": task_key})
                    move_to_next_task()
        else:
            st.error(f"Все попытки ({max_att}) исчерпаны.")
            if task.get("answer"):
                st.info(f"**Правильный ответ:** {task['answer']}")
            if task.get("solution"):
                st.caption(f"Решение: {task['solution']}")
            if st.button("Следующее задание", key=f"next_after_limit_{task_key}"):
                move_to_next_task()

        # Подсказки
        if task_key in st.session_state and "hints" in st.session_state[task_key]:
            st.markdown("### 💡 Подсказки:")
            for hint in st.session_state[task_key]["hints"]:
                st.info(hint)

        st.markdown("</div>", unsafe_allow_html=True)


def check_answer(session: SessionManager, task: dict, user_answer: str, task_key: str):
    st.session_state.task_attempts[task_key] = st.session_state.task_attempts.get(task_key, 0) + 1
    attempts = st.session_state.task_attempts[task_key]
    max_attempts = APP_CONFIG["max_attempts_per_task"]

    is_correct = compare_answers(
        (user_answer or "").strip().lower(),
        (task.get("answer") or "").strip().lower(),
    )

    if is_correct:
        st.markdown('<div class="success-animation">', unsafe_allow_html=True)
        st.success("Правильно! Отличная работа.")
        st.markdown("</div>", unsafe_allow_html=True)
        if task_key not in st.session_state.completed_tasks:
            st.session_state.completed_tasks.append(task_key)
        log_user_action("correct_answer", {"task_key": task_key, "attempts": attempts})
        if st.button("Следующее задание", key=f"next_{task_key}"):
            move_to_next_task()
    else:
        if attempts < max_attempts:
            st.error(f"Неправильно. Попытка {attempts} из {max_attempts}")
            # делаем короткую подсказку
            hint = "Подумай, какой шаг в вычислениях мог быть сделан неверно."  # fallback
            try:
                hint_resp = call_llm(
                    f"""
Студент решал задачу: "{task.get('question','')}"
Правильный ответ: "{task.get('answer','')}"
Ответ студента: "{user_answer}"
Дай краткую подсказку (1–2 предложения), без LaTeX и без полного решения.
"""
                )
                if isinstance(hint_resp, dict) and "content" in hint_resp:
                    hint = str(hint_resp["content"]).strip() or hint
            except Exception:
                pass

            if task_key not in st.session_state:
                st.session_state[task_key] = {"hints": []}
            st.session_state[task_key]["hints"].append(hint)
            st.info(f"Подсказка: {hint}")
            log_user_action("incorrect_answer", {"task_key": task_key, "attempts": attempts})
        else:
            st.error("Все попытки исчерпаны.")
            if task.get("answer"):
                st.info(f"**Правильный ответ:** {task['answer']}")
            if task.get("solution"):
                st.caption(f"Решение: {task['solution']}")
            if st.button("Следующее задание", key=f"next_limit_{task_key}"):
                move_to_next_task()


def move_to_next_task():
    st.session_state.current_task_index += 1
    st.rerun()


def show_practice_completion(session: SessionManager):
    videos = session.get_videos()
    if not videos:
        st.info("Практика завершена.")
        return

    current_video = videos[session.get_current_video_index()]
    topic_key = f"{session.get_subject()}_{session.get_grade()}_{current_video['title']}"

    st.markdown('<div class="progress-card">', unsafe_allow_html=True)
    st.header("Практика завершена!")

    total = sum(len(st.session_state.practice_tasks.get(t, [])) for t in ["easy", "medium", "hard"])
    done = len(st.session_state.completed_tasks)
    score = calculate_score(done, total) if total else 0.0
    st.success(f"Выполнено {done} из {total} заданий ({score:.0f}%)")

    session.save_practice_score(topic_key, done, total)

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


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

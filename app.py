import os
import json
import re
from datetime import datetime

import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    sanitize_theory_questions,
)

# set_page_config — обязательно первым вызовом Streamlit
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["initial_sidebar_state"],
)

# === РЕЗОЛВИМ КЛЮЧИ ПОСЛЕ set_page_config ===
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


class EnhancedAITutor:
    def __init__(self):
        self.youtube_api_key = YOUTUBE_API_KEY
        self.deepseek_api_key = DEEPSEEK_API_KEY
        self.playlists = PLAYLISTS
        self.config = APP_CONFIG
        self.deepseek_config = DEEPSEEK_CONFIG
        self.ui_config = UI_CONFIG

    # ---------- YouTube ----------
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
                thumbs = sn.get("thumbnails", {}) or {}
                thumb = thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}
                video = {
                    "title": sn.get("title", "Без названия"),
                    "video_id": (sn.get("resourceId") or {}).get("videoId"),
                    "description": (sn.get("description") or "")[:200]
                    + ("..." if len(sn.get("description") or "") > 200 else ""),
                    "thumbnail": thumb.get("url", ""),
                    "published_at": sn.get("publishedAt", ""),
                }
                if video["video_id"]:
                    videos.append(video)
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

    # ---------- DeepSeek ----------
    def _call_deepseek_api(self, prompt, *, max_tokens=None, timeout=None):
        if not DEEPSEEK_ENABLED:
            return {"error": "deepseek_disabled"}
        headers = {"Authorization": f"Bearer {self.deepseek_api_key}", "Content-Type": "application/json"}
        data = {
            "model": self.deepseek_config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.deepseek_config["temperature"],
            "max_tokens": max_tokens or self.deepseek_config["max_tokens"],
        }
        for attempt in range(self.deepseek_config["retry_attempts"]):
            try:
                resp = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=timeout or self.deepseek_config["timeout"],
                )
                if resp.status_code == 402:
                    st.warning("DeepSeek вернул 402 (недостаточно средств). Генерация временно отключена.")
                    return {"error": "402"}
                resp.raise_for_status()
                result = resp.json()
                content = result["choices"][0]["message"]["content"]
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"content": content}
            except requests.exceptions.Timeout:
                if attempt == self.deepseek_config["retry_attempts"] - 1:
                    return {"error": "timeout"}
            except requests.exceptions.HTTPError as e:
                if attempt == self.deepseek_config["retry_attempts"] - 1:
                    return {"error": f"http_{e.response.status_code}"}
            except Exception as e:
                if attempt == self.deepseek_config["retry_attempts"] - 1:
                    return {"error": str(e)}

    # ------ Теория: генерация с «дозапросом» ------
    def _prompt_theory_block(self, topic, subject, grade, count):
        return f"""
Создай РОВНО {count} теоретических вопросов по теме "{topic}" для {grade}-го класса по предмету "{subject}".

Требования к каждому вопросу:
- Четкий вопрос (можно LaTeX формулы: \\( ... \\))
- Ровно 4 варианта ответа в формате: ["A) ...", "B) ...", "C) ...", "D) ..."]
- Один правильный ответ: "A" | "B" | "C" | "D"
- Короткое объяснение правильного ответа
- Не добавляй поясняющий текст вне JSON!

Верни строго ВАЛИДНЫЙ JSON:
{{
  "questions": [
    {{
      "question": "Текст вопроса с формулами при необходимости",
      "options": ["A) вариант1", "B) вариант2", "C) вариант3", "D) вариант4"],
      "correct_answer": "A",
      "explanation": "Короткое объяснение"
    }}
  ]
}}
""".strip()

    def _prompt_theory_topup(self, topic, subject, grade, count):
        # просим ДОБАВИТЬ ещё N вопросов — без лишнего текста
        return f"""
Сгенерируй ДОПОЛНИТЕЛЬНО РОВНО {count} теоретических вопросов по теме "{topic}" ({grade} класс, предмет "{subject}").
Верни ТОЛЬКО ВАЛИДНЫЙ JSON с полем "questions" — как в примере, без пояснений, без многоточий, без текста вне JSON.
Структура как прежде.
""".strip()

    def generate_theory_questions_with_topup(self, topic, subject, grade, total_needed):
        """
        1) Первая попытка — запросить total_needed вопросов.
        2) Если пришло меньше — догенерировать недостающее количество (1-2 раза).
        3) Санитайзинг + обрезка до total_needed.
        """
        all_qs = []

        # первая попытка
        p1 = self._prompt_theory_block(topic, subject, grade, total_needed)
        r1 = self._call_deepseek_api(
            p1,
            max_tokens=self.deepseek_config["max_tokens_theory"],
            timeout=self.deepseek_config["timeout_theory"],
        )
        if isinstance(r1, dict) and r1.get("questions"):
            all_qs.extend(r1["questions"])
        # если пришёл "content" — ничего не парсим, пропустим (потом fallback)

        # дозапрос недостающих
        retries = self.deepseek_config["theory_topup_retries"]
        for _ in range(retries):
            missing = total_needed - len(all_qs)
            if missing <= 0:
                break
            p2 = self._prompt_theory_topup(topic, subject, grade, missing)
            r2 = self._call_deepseek_api(
                p2,
                max_tokens=max(800, int(self.deepseek_config["max_tokens_theory"] * 0.5)),
                timeout=max(20, int(self.deepseek_config["timeout_theory"] * 0.7)),
            )
            if isinstance(r2, dict) and r2.get("questions"):
                all_qs.extend(r2["questions"])

        # санитайзим + отсекаем лишнее
        clean = sanitize_theory_questions(all_qs)
        if len(clean) > total_needed:
            clean = clean[:total_needed]
        return clean


# ========================= Основной поток =========================

def main():
    st.markdown('<div class="main-header"><h1>📚 AI Тьютор — Персональное обучение</h1></div>', unsafe_allow_html=True)

    # ---- USER ID для облачного прогресса ----
    st.sidebar.markdown("### 👤 Пользователь")
    user_id = st.sidebar.text_input("Идентификатор (для облака)", placeholder="email или ник")
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

    # Выбор курса
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
        st.info("👆 Выберите предмет и класс в боковой панели, затем нажмите «Начать обучение»")


def display_video_content(tutor, session):
    videos = session.get_videos()
    if not videos:
        st.warning("Видео из плейлиста не загружены. Попробуйте перезагрузить страницу.")
        return
    current_video = videos[session.get_current_video_index()]
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header(f"📺 {current_video['title']}")
        st.video(f"https://www.youtube.com/watch?v={current_video['video_id']}")
        if current_video['description']:
            with st.expander("Описание урока"):
                st.write(current_video['description'])
    with col2:
        st.markdown('<div class="progress-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Текущий урок")
        st.info(f"Урок {session.get_current_video_index() + 1} из {len(videos)}")
        st.progress((session.get_current_video_index() + 1) / len(videos))
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Готов к тесту", type="primary"):
                session.set_stage("theory_test")
                log_user_action("start_theory_test", {"video": current_video['title']})
                st.rerun()
        with col_btn2:
            if st.button("Пересмотреть"):
                log_user_action("rewatch_video", {"video": current_video['title']})
                st.rerun()
        if session.get_current_video_index() > 0:
            if st.button("← Предыдущий урок"):
                session.prev_video()
                log_user_action("previous_video", {"video_index": session.get_current_video_index()})
                st.rerun()
        if session.get_current_video_index() < len(videos) - 1:
            if st.button("Следующий урок →"):
                session.next_video()
                log_user_action("next_video", {"video_index": session.get_current_video_index()})
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


def show_theory_test(tutor, session):
    current_video = session.get_videos()[session.get_current_video_index()]
    st.header("📝 Тест по теории")
    st.info(f"Тема: {current_video['title']}")

    total_needed = APP_CONFIG["theory_questions_count"]
    min_needed = APP_CONFIG["theory_min_questions"]

    if "theory_questions" not in st.session_state:
        with st.spinner("Генерация вопросов..."):
            questions = tutor.generate_theory_questions_with_topup(
                current_video["title"],
                session.get_subject(),
                session.get_grade(),
                total_needed=total_needed,
            )
            # если пришло меньше минимума — показываем понятное сообщение
            if len(questions) < min_needed:
                err = "Не удалось получить достаточно вопросов от модели. Нажмите «Попробовать снова»."
                st.error(err)
                st.session_state.theory_questions = []
                st.session_state.theory_answers = {}
                return
            st.session_state.theory_questions = questions
            st.session_state.theory_answers = {}

    if st.session_state.theory_questions:
        for i, q in enumerate(st.session_state.theory_questions):
            st.markdown('<div class="task-card">', unsafe_allow_html=True)
            st.markdown(f"**Вопрос {i+1}:** {q.get('question','')}", unsafe_allow_html=True)

            options = q.get("options", ["A) —", "B) —", "C) —", "D) —"])
            answer_key = f"theory_q_{i}"
            selected = st.radio("Выберите ответ:", options, key=answer_key, index=None)
            if selected:
                st.session_state.theory_answers[i] = selected[0]  # буква A/B/C/D
            st.markdown('</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("← Вернуться к видео"):
                session.clear_theory_data()
                session.set_stage("video")
                log_user_action("return_to_video", {"video": current_video['title']})
                st.rerun()
        with col2:
            if st.button("Проверить ответы", type="primary"):
                if len(st.session_state.theory_answers) == len(st.session_state.theory_questions):
                    show_theory_results(tutor, session)
                else:
                    st.error("Пожалуйста, ответьте на все вопросы")
        with col3:
            if st.button("Попробовать снова"):
                session.clear_theory_data()
                st.rerun()


def show_theory_results(tutor, session):
    current_video = session.get_videos()[session.get_current_video_index()]
    topic_key = f"{session.get_subject()}_{session.get_grade()}_{current_video['title']}"

    st.markdown('<div class="progress-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Результаты тестирования")

    correct_count = 0
    total_questions = len(st.session_state.theory_questions)

    for i, q in enumerate(st.session_state.theory_questions):
        user_answer = st.session_state.theory_answers.get(i)
        correct_answer = q.get("correct_answer")

        if compare_answers(user_answer, correct_answer):
            correct_count += 1
            st.markdown('<div class="success-animation">', unsafe_allow_html=True)
            st.success(f"Вопрос {i+1}: Правильно!")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error(f"Вопрос {i+1}: Неправильно")
            st.markdown(f"**Объяснение:** {q.get('explanation','')}", unsafe_allow_html=True)

    score = calculate_score(correct_count, total_questions)
    st.metric("Ваш результат", f"{correct_count}/{total_questions} ({score:.0f}%)")

    session.save_theory_score(topic_key, score)
    if score < tutor.config["theory_pass_threshold"]:
        st.warning("Проходной порог — 60%. Рекомендуем пересмотреть видео для лучшего понимания темы.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Пересмотреть урок"):
            session.clear_theory_data()
            session.set_stage("video")
            log_user_action("rewatch_after_theory", {"video": current_video['title'], "score": score})
            st.rerun()
    with col2:
        if st.button("Начать практику", type="primary"):
            session.clear_theory_data()
            session.set_stage("practice")
            log_user_action("start_practice", {"video": current_video['title'], "theory_score": score})
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Практика (как было) ----------------

def show_practice_stage(tutor, session):
    current_video = session.get_videos()[session.get_current_video_index()]
    st.header("💪 Практические задания")
    st.info(f"Тема: {current_video['title']}")
    st.markdown(
        """
    <div class="notebook-note">
        📝 <b>Совет:</b> Для сложных задач используйте тетрадь. Введите конечный ответ.
        Для неравенств — <code>x >= 2</code> или <code>[2, inf)</code>. Для нескольких условий — <code>and</code> или <code>,</code>.
    </div>
    """,
        unsafe_allow_html=True,
    )

    if "practice_tasks" not in st.session_state:
        with st.spinner("Генерация заданий..."):
            theory_score = session.get_theory_score(current_video["title"])
            tasks_data = tutor._call_deepseek_api(
                f"""
Составь практические задачи по теме "{current_video['title']}" для {session.get_grade()}-го класса по предмету "{session.get_subject()}":
- {APP_CONFIG["tasks_per_difficulty"]["easy"]} легкие задачи
- {APP_CONFIG["tasks_per_difficulty"]["medium"]} средние задачи
- {APP_CONFIG["tasks_per_difficulty"]["hard"]} сложные задачи

Требования к каждой задаче:
- "question": условие (можно LaTeX)
- "answer": точный ответ (текст/число/интервалы, без LaTeX)
- "solution": краткое пошаговое решение (можно LaTeX)
- "hint": подсказка без LaTeX

Верни строго ВАЛИДНЫЙ JSON:
{{
  "easy": [{{"question":"...", "answer":"...", "solution":"...", "hint":"..."}} ],
  "medium": [{{"question":"...", "answer":"...", "solution":"...", "hint":"..."}} ],
  "hard": [{{"question":"...", "answer":"...", "solution":"...", "hint":"..."}} ]
}}
""",
                max_tokens=DEEPSEEK_CONFIG["max_tokens_practice"],
                timeout=DEEPSEEK_CONFIG["timeout_practice"],
            ) or {}

            if isinstance(tasks_data, dict) and tasks_data.get("content"):
                try:
                    tasks_data = json.loads(tasks_data["content"])
                except Exception:
                    tasks_data = {"easy": [], "medium": [], "hard": []}

            if isinstance(tasks_data, dict) and tasks_data.get("error"):
                st.error("Не удалось сгенерировать задания (DeepSeek недоступен).")
                tasks_data = {"easy": [], "medium": [], "hard": []}

            st.session_state.practice_tasks = tasks_data
            st.session_state.task_attempts = {}
            st.session_state.completed_tasks = []
            st.session_state.current_task_type = "easy"
            st.session_state.current_task_index = 0

    if any(len(st.session_state.practice_tasks.get(t, [])) for t in ["easy", "medium", "hard"]):
        show_current_task(tutor, session)
    else:
        st.error("Нет заданий. Попробуйте позже.")


def show_current_task(tutor, session):
    task_types = ["easy", "medium", "hard"]
    current_type = st.session_state.current_task_type
    current_index = st.session_state.current_task_index
    tasks_of_type = st.session_state.practice_tasks.get(current_type, [])

    if current_index >= len(tasks_of_type):
        current_type_index = task_types.index(current_type)
        if current_type_index < len(task_types) - 1:
            st.session_state.current_task_type = task_types[current_type_index + 1]
            st.session_state.current_task_index = 0
            st.rerun()
        else:
            show_practice_completion(tutor, session)
            return

    current_task = tasks_of_type[current_index]
    task_key = f"{current_type}_{current_index}"

    total_tasks = sum(len(st.session_state.practice_tasks.get(t, [])) for t in task_types)
    completed_tasks = len(st.session_state.completed_tasks)

    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown('<div class="progress-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Прогресс")
        st.progress(completed_tasks / total_tasks if total_tasks > 0 else 0)
        st.metric("Выполнено", f"{completed_tasks}/{total_tasks}")
        st.markdown(
            f'<span class="difficulty-badge {current_type}">{tutor.ui_config["task_type_names"][current_type]}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Задание:** {current_index + 1} из {len(tasks_of_type)}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col1:
        st.markdown(
            f'<div class="task-card"><span class="difficulty-badge {current_type}">{tutor.ui_config["task_type_names"][current_type]}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"### Задание {current_index + 1}")
        st.markdown(current_task.get("question", ""), unsafe_allow_html=True)

        user_answer = st.text_input("Ваш ответ:", key=f"answer_{task_key}")
        attempts = st.session_state.task_attempts.get(task_key, 0)
        max_attempts = tutor.config["max_attempts_per_task"]

        if attempts < max_attempts:
            col_check, col_skip = st.columns([1, 1])
            with col_check:
                if st.button("Проверить ответ", type="primary"):
                    if (user_answer or "").strip():
                        check_answer(tutor, session, current_task, user_answer, task_key)
                    else:
                        st.error("Введите ответ!")
            with col_skip:
                if st.button("Пропустить"):
                    log_user_action("skip_task", {"task_key": task_key})
                    move_to_next_task()
        else:
            st.error(f"Исчерпаны все попытки ({max_attempts})")
            st.markdown(f"**Правильный ответ:** {current_task.get('answer','')}", unsafe_allow_html=True)
            st.markdown(f"**Решение:** {current_task.get('solution','')}", unsafe_allow_html=True)
            if st.button("Следующее задание"):
                move_to_next_task()

        if task_key in st.session_state and "hints" in st.session_state[task_key]:
            st.markdown("### 💡 Подсказки:")
            for hint in st.session_state[task_key]["hints"]:
                st.info(hint)
        st.markdown("</div>", unsafe_allow_html=True)


def check_answer(tutor, session, task, user_answer, task_key):
    st.session_state.task_attempts[task_key] = st.session_state.task_attempts.get(task_key, 0) + 1
    attempts = st.session_state.task_attempts[task_key]
    max_attempts = tutor.config["max_attempts_per_task"]

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
                hint = "Подумай, какие свойства применяются к этой формуле."
                if DEEPSEEK_ENABLED:
                    try:
                        hint_resp = tutor._call_deepseek_api(
                            f"""Студент решал задачу: "{task.get('question','')}"
Правильный ответ: "{task.get('answer','')}"
Ответ студента: "{user_answer}"
Дай краткую подсказку (1-2 предложения) без LaTeX и без полного решения."""
                        )
                        if isinstance(hint_resp, dict) and "content" in hint_resp:
                            hint = hint_resp["content"]
                    except Exception:
                        pass
                if task_key not in st.session_state:
                    st.session_state[task_key] = {"hints": []}
                st.session_state[task_key]["hints"].append(hint)
                st.info(f"Подсказка: {hint}")
            log_user_action("incorrect_answer", {"task_key": task_key, "attempts": attempts})
        else:
            st.error("Все попытки исчерпаны.")
            st.markdown(f"**Правильный ответ:** {task.get('answer','')}", unsafe_allow_html=True)
            st.markdown(f"**Решение:** {task.get('solution','')}", unsafe_allow_html=True)
            if st.button("Следующее задание"):
                move_to_next_task()


def move_to_next_task():
    st.session_state.current_task_index += 1    # следующий индекс в рамках типа
    st.rerun()


def show_practice_completion(tutor, session):
    videos = session.get_videos()
    if not videos:
        st.info("Практика завершена.")
        return
    current_video = videos[session.get_current_video_index()]
    topic_key = f"{session.get_subject()}_{session.get_grade()}_{current_video['title']}"

    st.markdown('<div class="progress-card">', unsafe_allow_html=True)
    st.header("Практика завершена!")

    task_types = ["easy", "medium", "hard"]
    total_tasks = sum(len(st.session_state.practice_tasks.get(t, [])) for t in task_types)
    completed = len(st.session_state.completed_tasks)
    score = calculate_score(completed, total_tasks) if total_tasks else 0
    st.success(f"Выполнено {completed} из {total_tasks} заданий ({score:.0f}%)")

    session.save_practice_score(topic_key, completed, total_tasks)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Изучить новую тему"):
            if session.next_video():
                session.set_stage("video")
                for k in [
                    "practice_tasks",
                    "task_attempts",
                    "completed_tasks",
                    "current_task_type",
                    "current_task_index",
                ]:
                    if k in st.session_state:
                        del st.session_state[k]
                log_user_action("next_topic", {"video_index": session.get_current_video_index()})
                st.rerun()
            else:
                st.info("Все темы курса пройдены!")
    with col2:
        if st.button("Вернуться к выбору курса"):
            session.set_stage("selection")
            for k in [
                "practice_tasks",
                "task_attempts",
                "completed_tasks",
                "current_task_type",
                "current_task_index",
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            log_user_action("return_to_selection", {})
            st.rerun()

    st.markdown(generate_progress_report(session.get_progress(), topic_key), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

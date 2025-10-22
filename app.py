# app.py
import os
import json
import requests
from datetime import datetime

import streamlit as st
import plotly.express as px

from config import (
    PLAYLISTS, APP_CONFIG, DEEPSEEK_CONFIG, UI_CONFIG,
    SUPABASE_URL, SUPABASE_ANON_KEY  # можно оставить пустыми, если не используешь облако
)
from utils import (
    compare_answers, calculate_score, generate_progress_report,
    get_subject_emoji, SessionManager, create_progress_chart_data,
    log_user_action, sanitize_theory_questions  # важно!
)

# ─────────────────────────────
# 1) set_page_config ДОЛЖЕН быть первым Streamlit-вызовом
# ─────────────────────────────
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["initial_sidebar_state"],
)

# ─────────────────────────────
# 2) Резолв ключей после set_page_config
# ─────────────────────────────
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

try:
    if not YOUTUBE_API_KEY and hasattr(st, "secrets") and "YOUTUBE_API_KEY" in st.secrets:
        YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    if not DEEPSEEK_API_KEY and hasattr(st, "secrets") and "DEEPSEEK_API_KEY" in st.secrets:
        DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    pass

if not YOUTUBE_API_KEY:
    st.error("Не задан YOUTUBE_API_KEY. Укажи в .env или в Secrets.")
    st.stop()

DEEPSEEK_ENABLED = bool(DEEPSEEK_API_KEY)

# Удобные константы
TARGET_THEORY_Q = APP_CONFIG["theory_questions_count"]
PASS_THRESHOLD = APP_CONFIG.get("theory_pass_threshold", 60)

# ─────────────────────────────
# MathJax (формулы)
# ─────────────────────────────
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

# ─────────────────────────────
# CSS
# ─────────────────────────────
st.markdown(
    """
<style>
  .main-header {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    color: white;
    margin-bottom: 2rem;
  }
  .progress-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin: 1rem 0;
  }
  .task-card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid #007bff;
    margin: 1rem 0;
  }
  .success-animation { animation: pulse 0.5s ease-in-out; }
  @keyframes pulse { 0%{transform:scale(1);} 50%{transform:scale(1.05);} 100%{transform:scale(1);} }
  .difficulty-badge {
    display: inline-block; padding: 0.3rem 0.8rem; border-radius: 15px;
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;
  }
  .easy { background-color: #d4edda; color: #155724; }
  .medium { background-color: #fff3cd; color: #856404; }
  .hard { background-color: #f8d7da; color: #721c24; }
  .notebook-note {
    background-color: #e9f7ef; padding: 1rem; border-radius: 8px;
    margin-bottom: 1rem; border-left: 4px solid #28a745;
  }
  .correct {
    border-left: 4px solid #28a745 !important;
    background: #ecfdf5;
  }
  .wrong {
    border-left: 4px solid #ef4444 !important;
    background: #fef2f2;
  }
  .badge{ display:inline-block; padding:.25rem .5rem; border-radius:6px; font-size:.75rem; font-weight:600; }
  .badge-green{ background:#d1fae5; color:#065f46; } .badge-gray{ background:#e5e7eb; color:#374151; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────
# Core класс
# ─────────────────────────────
class EnhancedAITutor:
    def __init__(self):
        self.youtube_api_key = YOUTUBE_API_KEY
        self.deepseek_api_key = DEEPSEEK_API_KEY
        self.playlists = PLAYLISTS
        self.config = APP_CONFIG
        self.deepseek_config = DEEPSEEK_CONFIG
        self.ui_config = UI_CONFIG

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
                rid = sn.get("resourceId", {}) or {}
                thumbs = sn.get("thumbnails", {}) or {}
                thumb = thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}
                vid = rid.get("videoId")
                if not vid:
                    continue
                videos.append({
                    "title": sn.get("title") or "Без названия",
                    "video_id": vid,
                    "description": (sn.get("description") or "")[:200] + ("..." if len(sn.get("description") or "") > 200 else ""),
                    "thumbnail": thumb.get("url", ""),
                    "published_at": sn.get("publishedAt", ""),
                })
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

    # DeepSeek — универсальный вызов
    def _call_deepseek_api(self, prompt):
        if not DEEPSEEK_ENABLED:
            return {"error": "deepseek_disabled"}
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.deepseek_config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.deepseek_config["temperature"],
            "max_tokens": self.deepseek_config["max_tokens"],
        }
        for attempt in range(self.deepseek_config["retry_attempts"]):
            try:
                resp = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers, json=data, timeout=self.deepseek_config["timeout"]
                )
                if resp.status_code == 402:
                    st.warning("DeepSeek вернул 402 (недостаточно средств). Генерация временно отключена.")
                    return {"error": "402"}
                resp.raise_for_status()
                payload = resp.json()
                content = payload["choices"][0]["message"]["content"]
                # Пытаемся распарсить как JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"content": content}
            except requests.exceptions.Timeout:
                if attempt == self.deepseek_config["retry_attempts"] - 1:
                    st.error("Превышено время ожидания ответа от DeepSeek API")
                    return {"error": "timeout"}
            except requests.exceptions.HTTPError as e:
                if attempt == self.deepseek_config["retry_attempts"] - 1:
                    st.error(f"Ошибка HTTP DeepSeek API: {e.response.status_code}")
                    return {"error": str(e)}
            except Exception as e:
                if attempt == self.deepseek_config["retry_attempts"] - 1:
                    st.error(f"Ошибка API DeepSeek: {str(e)}")
                    return {"error": str(e)}

    # Теоретические вопросы — только теория (без уровней)
    def generate_theory_questions(self, topic, subject, grade, n_questions):
        prompt = f"""
Сгенерируй {n_questions} тестовых вопросов ПО ТЕМЕ "{topic}" для {grade}-го класса по предмету "{subject}".
Только вопросы по этой теме (не выходи за программу этого класса).
Каждый вопрос:
- один правильный ответ из 4 вариантов,
- варианты в формате: "A) ...", "B) ...", "C) ...", "D) ...",
- дай краткое объяснение правильного ответа,
- формулы в тексте — в LaTeX, например \\(x^2 + 2x + 1\\).

Верни строго ВАЛИДНЫЙ JSON без комментариев/многоточий:
{{
  "questions": [
    {{
      "question": "Текст вопроса c формулами LaTeX: \\(...\\)",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct_answer": "A",
      "explanation": "Краткое объяснение"
    }}
  ]
}}
"""
        return self._call_deepseek_api(prompt)

    # Практика — easy/medium/hard
    def generate_practice_tasks(self, topic, subject, grade, user_performance=None):
        adj = ""
        if user_performance is not None:
            if user_performance < 60:
                adj = "Сделай акцент на простые задания с подробными объяснениями."
            elif user_performance > 85:
                adj = "Добавь несколько нестандартных, повышенной сложности."
        prompt = f"""
Составь практические задания по теме "{topic}" для {grade}-го класса по предмету "{subject}":
- {self.config["tasks_per_difficulty"]["easy"]} лёгких,
- {self.config["tasks_per_difficulty"]["medium"]} средних,
- {self.config["tasks_per_difficulty"]["hard"]} сложных.

{adj}

Для каждой задачи верни:
- "question" — условие (формулы в LaTeX),
- "answer" — правильный ответ (текст/число, без LaTeX, например "x >= 2, x < 3"),
- "solution" — пошаговое решение (с LaTeX),
- "hint" — короткая подсказка (без LaTeX).

Верни строго валидный JSON без комментариев/многоточий:
{{
  "easy": [{{"question":"...","answer":"...","solution":"...","hint":"..."}} ],
  "medium": [{{"question":"...","answer":"...","solution":"...","hint":"..."}} ],
  "hard": [{{"question":"...","answer":"...","solution":"...","hint":"..."}} ]
}}
"""
        return self._call_deepseek_api(prompt)


# ─────────────────────────────
# UI
# ─────────────────────────────
def main():
    st.markdown('<div class="main-header"><h1>📚 AI Тьютор — персональное обучение</h1></div>', unsafe_allow_html=True)

    # Sidebar — user id (для облака)
    st.sidebar.markdown("### 👤 Пользователь")
    user_id = st.sidebar.text_input("Идентификатор (email/ник для облака)", placeholder="например, email или ник")
    sb_on = bool(
        (SUPABASE_URL or (hasattr(st, "secrets") and st.secrets.get("SUPABASE_URL"))) and
        (SUPABASE_ANON_KEY or (hasattr(st, "secrets") and st.secrets.get("SUPABASE_ANON_KEY")))
    )
    if user_id and sb_on:
        st.sidebar.markdown('<span class="badge badge-green">Supabase: подключено</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="badge badge-gray">Supabase: локальное хранение</span>', unsafe_allow_html=True)

    tutor = EnhancedAITutor()
    session = SessionManager(user_id=user_id if user_id else None)

    # Sidebar — выбор курса
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

    # Sidebar — прогресс
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
        st.info("👆 Выберите предмет и класс слева, затем нажмите «Начать обучение».")


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
        desc = current_video.get("description")
        if desc:
            with st.expander("Описание урока"):
                st.write(desc)
    with col2:
        st.markdown('<div class="progress-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Текущий урок")
        st.info(f"Урок {session.get_current_video_index() + 1} из {len(videos)}")
        st.progress((session.get_current_video_index() + 1) / len(videos))
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Готов к тесту", type="primary"):
                session.set_stage("theory_test")
                log_user_action("start_theory_test", {"video": current_video["title"]})
                st.rerun()
        with c2:
            if st.button("Пересмотреть"):
                log_user_action("rewatch_video", {"video": current_video["title"]})
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


# ─────────────────────────────
# Теория с «Попробовать снова»
# ─────────────────────────────
def show_theory_test(tutor, session):
    current_video = session.get_videos()[session.get_current_video_index()]
    st.header("📝 Тест по теории")
    st.info(f"Тема: {current_video['title']}")

    def _generate():
        data = tutor.generate_theory_questions(
            current_video["title"],
            session.get_subject(),
            session.get_grade(),
            TARGET_THEORY_Q
        )
        # Обработка 402/timeout/disabled
        if isinstance(data, dict) and data.get("error") in ("402", "deepseek_disabled", "timeout"):
            return []

        # Если пришёл «сырой» текст — пробуем распарсить JSON
        if isinstance(data, dict) and "content" in data:
            try:
                data = json.loads(data["content"])
            except Exception:
                data = {"questions": []}

        raw = (data or {}).get("questions", [])
        safe = sanitize_theory_questions(raw)

        # Дозаполним заглушками до нужного количества
        while len(safe) < TARGET_THEORY_Q:
            idx = len(safe) + 1
            safe.append({
                "question": f"Вопрос недоступен ({idx}). Нажмите «🔁 Попробовать снова».",
                "options": ["A) —", "B) —", "C) —", "D) —"],
                "correct_answer": "A",
                "explanation": "—",
            })
        return safe[:TARGET_THEORY_Q]

    def _retry():
        for k in ("theory_questions", "theory_answers"):
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    # Первая генерация
    if "theory_questions" not in st.session_state:
        with st.spinner("Генерация вопросов..."):
            st.session_state.theory_questions = _generate()
            st.session_state.theory_answers = {}

    questions = st.session_state.theory_questions or []

    # Проверим реальное число (без «Вопрос недоступен»)
    real_count = sum(1 for q in questions if not q["question"].startswith("Вопрос недоступен"))
    if real_count < TARGET_THEORY_Q:
        st.warning("Не удалось получить достаточно вопросов от модели.")
        if st.button("🔁 Попробовать снова"):
            _retry()

    # Рендер вопросов
    for i, q in enumerate(questions):
        st.markdown('<div class="task-card">', unsafe_allow_html=True)
        st.markdown(f"**Вопрос {i+1}:** {q.get('question', '')}", unsafe_allow_html=True)

        options = q.get("options", [])
        if not options or len(options) != 4:
            options = ["A) —", "B) —", "C) —", "D) —"]

        answer_key = f"theory_q_{i}"
        selected = st.radio("Выберите ответ:", options, key=answer_key, index=None)
        if selected:
            st.session_state.theory_answers[i] = selected[0]  # буква A/B/C/D

        st.markdown('</div>', unsafe_allow_html=True)

    # Кнопки
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("← Вернуться к видео"):
            session.clear_theory_data()
            session.set_stage("video")
            log_user_action("return_to_video", {"video": current_video["title"]})
            st.rerun()
    with c2:
        if st.button("🔁 Попробовать снова"):
            _retry()
    with c3:
        if st.button("Проверить ответы", type="primary"):
            if len(st.session_state.theory_answers) == len(questions):
                show_theory_results(tutor, session)
            else:
                st.error("Пожалуйста, ответьте на все вопросы.")


def show_theory_results(tutor, session):
    current_video = session.get_videos()[session.get_current_video_index()]
    topic_key = f"{session.get_subject()}_{session.get_grade()}_{current_video['title']}"

    st.markdown('<div class="progress-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Результаты тестирования")

    correct_count = 0
    total_questions = len(st.session_state.theory_questions)

    for i, q in enumerate(st.session_state.theory_questions):
        user_choice = st.session_state.theory_answers.get(i)
        correct_choice = q.get("correct_answer", "A")

        is_right = compare_answers(user_choice, correct_choice)

        # Подсветка карточки результата
        css_class = "correct" if is_right else "wrong"
        st.markdown(f'<div class="task-card {css_class}">', unsafe_allow_html=True)

        # Заголовок и вопрос
        st.markdown(f"**Вопрос {i+1}:** {q.get('question','')}", unsafe_allow_html=True)

        # Подсветка варианта: жирный + (правильный/ваш)
        opts = q.get("options", ["A) —","B) —","C) —","D) —"])
        def pretty_option(opt_text, letter):
            label = ""
            if letter == correct_choice:
                label = " ✅ правильный"
            if user_choice == letter and letter != correct_choice:
                label = " ❌ ваш ответ"
            if user_choice == letter and letter == correct_choice:
                label = " ✅ ваш ответ"
            if label:
                return f"**{opt_text}{label}**"
            return opt_text

        st.markdown(pretty_option(opts[0], "A"))
        st.markdown(pretty_option(opts[1], "B"))
        st.markdown(pretty_option(opts[2], "C"))
        st.markdown(pretty_option(opts[3], "D"))

        # Объяснение
        expl = q.get("explanation", "")
        if expl:
            st.markdown(f"**Объяснение:** {expl}", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if is_right:
            correct_count += 1

    score = calculate_score(correct_count, total_questions)
    st.metric("Ваш результат", f"{correct_count}/{total_questions} ({score:.0f}%)")

    session.save_theory_score(topic_key, score)

    if score < PASS_THRESHOLD:
        st.warning(f"Проходной порог: {PASS_THRESHOLD}%. Рекомендуем пересмотреть урок.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Пересмотреть урок"):
            session.clear_theory_data()
            session.set_stage("video")
            log_user_action("rewatch_after_theory", {"video": current_video["title"], "score": score})
            st.rerun()
    with c2:
        if st.button("Начать практику", type="primary"):
            session.clear_theory_data()
            session.set_stage("practice")
            log_user_action("start_practice", {"video": current_video["title"], "theory_score": score})
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────
# Практика
# ─────────────────────────────
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
            data = tutor.generate_practice_tasks(
                current_video["title"],
                session.get_subject(),
                session.get_grade(),
                theory_score,
            )
            if isinstance(data, dict) and "content" in data:
                try:
                    data = json.loads(data["content"])
                except Exception:
                    data = {"easy": [], "medium": [], "hard": []}
            if isinstance(data, dict) and data.get("error") in ("402", "deepseek_disabled", "timeout"):
                st.error("Не удалось сгенерировать задания (DeepSeek недоступен).")
                st.session_state.practice_tasks = {"easy": [], "medium": [], "hard": []}
            else:
                st.session_state.practice_tasks = data or {"easy": [], "medium": [], "hard": []}

            st.session_state.task_attempts = {}
            st.session_state.completed_tasks = []
            st.session_state.current_task_type = "easy"
            st.session_state.current_task_index = 0

    # Если вообще нет задач — сообщаем
    if not any(len(st.session_state.practice_tasks.get(t, [])) for t in ("easy", "medium", "hard")):
        st.error("Нет заданий. Попробуйте позже или пополните баланс DeepSeek.")
        return

    show_current_task(tutor, session)


def show_current_task(tutor, session):
    task_types = ["easy", "medium", "hard"]
    cur_type = st.session_state.current_task_type
    cur_index = st.session_state.current_task_index
    tasks = st.session_state.practice_tasks.get(cur_type, [])

    if cur_index >= len(tasks):
        t_idx = task_types.index(cur_type)
        if t_idx < len(task_types) - 1:
            st.session_state.current_task_type = task_types[t_idx + 1]
            st.session_state.current_task_index = 0
            st.rerun()
        else:
            show_practice_completion(tutor, session)
            return

    current = tasks[cur_index]
    tkey = f"{cur_type}_{cur_index}"

    total = sum(len(st.session_state.practice_tasks.get(t, [])) for t in task_types)
    done = len(st.session_state.completed_tasks)

    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown('<div class="progress-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Прогресс")
        st.progress(done / total if total else 0)
        st.metric("Выполнено", f"{done}/{total}")
        st.markdown(f'<span class="difficulty-badge {cur_type}">{tutor.ui_config["task_type_names"][cur_type]}</span>', unsafe_allow_html=True)
        st.markdown(f"**Задание:** {cur_index + 1} из {len(tasks)}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col1:
        st.markdown(f'<div class="task-card"><span class="difficulty-badge {cur_type}">{tutor.ui_config["task_type_names"][cur_type]}</span>', unsafe_allow_html=True)
        st.markdown(f"### Задание {cur_index + 1}")
        st.markdown(current.get("question", ""), unsafe_allow_html=True)

        user_answer = st.text_input("Ваш ответ:", key=f"answer_{tkey}")
        attempts = st.session_state.task_attempts.get(tkey, 0)
        max_attempts = tutor.config["max_attempts_per_task"]

        if attempts < max_attempts:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Проверить ответ", type="primary"):
                    if (user_answer or "").strip():
                        check_answer(tutor, session, current, user_answer, tkey)
                    else:
                        st.error("Введите ответ!")
            with c2:
                if st.button("Пропустить"):
                    log_user_action("skip_task", {"task_key": tkey})
                    move_to_next_task()
        else:
            st.error(f"Исчерпаны все попытки ({max_attempts})")
            st.markdown(f"**Правильный ответ:** {current.get('answer','')}")
            st.markdown(f"**Решение:** {current.get('solution','')}", unsafe_allow_html=True)
            if st.button("Следующее задание"):
                move_to_next_task()

        # Подсказки
        if tkey in st.session_state and "hints" in st.session_state[tkey]:
            st.markdown("### 💡 Подсказки:")
            for hint in st.session_state[tkey]["hints"]:
                st.info(hint)

        st.markdown("</div>", unsafe_allow_html=True)


def check_answer(tutor, session, task, user_answer, task_key):
    st.session_state.task_attempts[task_key] = st.session_state.task_attempts.get(task_key, 0) + 1
    attempts = st.session_state.task_attempts[task_key]
    max_attempts = tutor.config["max_attempts_per_task"]

    is_correct = compare_answers(
        (user_answer or "").strip().lower(),
        (task.get("answer") or "").strip().lower()
    )

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
            # Короткая подсказка: если DeepSeek доступен — попробуем; иначе fallback
            hint = "Подумай о ключевом свойстве, применимом в этой задаче."
            if DEEPSEEK_ENABLED:
                try:
                    resp = tutor._call_deepseek_api(f"""
Студент решал задачу: "{task.get('question','')}"
Правильный ответ: "{task.get('answer','')}"
Ответ студента: "{user_answer}"
Дай очень краткую подсказку (1-2 предложения) без LaTeX — где может быть ошибка.
""")
                    if isinstance(resp, dict) and "content" in resp:
                        hint = resp["content"]
                except Exception:
                    pass
            if task_key not in st.session_state:
                st.session_state[task_key] = {"hints": []}
            st.session_state[task_key]["hints"].append(hint)
            st.info(hint)
            log_user_action("incorrect_answer", {"task_key": task_key, "attempts": attempts})
        else:
            st.error("Все попытки исчерпаны.")
            st.markdown(f"**Правильный ответ:** {task.get('answer','')}")
            st.markdown(f"**Решение:** {task.get('solution','')}", unsafe_allow_html=True)
            if st.button("Следующее задание"):
                move_to_next_task()


def move_to_next_task():
    st.session_state.current_task_index += 1
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

    total_tasks = sum(len(st.session_state.practice_tasks.get(t, [])) for t in ["easy", "medium", "hard"])
    completed = len(st.session_state.completed_tasks)
    score = calculate_score(completed, total_tasks) if total_tasks else 0

    st.success(f"Выполнено {completed} из {total_tasks} заданий ({score:.0f}%)")
    session.save_practice_score(topic_key, completed, total_tasks)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Изучить новую тему"):
            if session.next_video():
                session.set_stage("video")
                # очистка состояния практики
                for k in ["practice_tasks", "task_attempts", "completed_tasks", "current_task_type", "current_task_index"]:
                    if k in st.session_state:
                        del st.session_state[k]
                log_user_action("next_topic", {"video_index": session.get_current_video_index()})
                st.rerun()
            else:
                st.info("Все темы курса пройдены!")
    with c2:
        if st.button("Вернуться к выбору курса"):
            session.set_stage("selection")
            for k in ["practice_tasks", "task_attempts", "completed_tasks", "current_task_type", "current_task_index"]:
                if k in st.session_state:
                    del st.session_state[k]
            log_user_action("return_to_selection", {})
            st.rerun()

    st.markdown(generate_progress_report(session.get_progress(), topic_key), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

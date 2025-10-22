import os
import re
import json
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import streamlit as st

# === Конфигурация и утилиты ===
from config import (
    PLAYLISTS, APP_CONFIG, DEEPSEEK_CONFIG, UI_CONFIG,
    SUPABASE_URL, SUPABASE_ANON_KEY  # ок, если нет в config — просто не используем
)
from utils import (
    compare_answers, calculate_score, generate_progress_report,
    get_subject_emoji, SessionManager, create_progress_chart_data,
    log_user_action
)

# ────────────────────────────────────────────────────────────────────────────────
# set_page_config ДОЛЖЕН быть самым первым streamlit-вызовом
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["initial_sidebar_state"],
)

# === РЕЗОЛВИМ API-ключи ПОСЛЕ set_page_config ===
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

# DeepSeek может отсутствовать — тогда генерацию аккуратно отключим
DEEPSEEK_ENABLED = bool(DEEPSEEK_API_KEY)

# ────────────────────────────────────────────────────────────────────────────────
# Подключение MathJax
st.markdown("""
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
<script>
    MathJax.Hub.Config({
        tex2jax: { inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']], processEscapes: true }
    });
    MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
</script>
""", unsafe_allow_html=True)

# CSS (чуть компактнее)
st.markdown("""
<style>
.main-header{ text-align:center; padding:2rem; background:linear-gradient(90deg,#667eea 0%,#764ba2 100%); border-radius:10px; color:#fff; margin-bottom:2rem; }
.progress-card{ background:#fff; padding:1.5rem; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,0.1); margin:1rem 0; }
.task-card{ background:#f8f9fa; padding:1.5rem; border-radius:8px; border-left:4px solid #007bff; margin:1rem 0; }
.success-animation{ animation:pulse .5s ease-in-out; }
@keyframes pulse{ 0%{transform:scale(1);} 50%{transform:scale(1.05);} 100%{transform:scale(1);} }
.difficulty-badge{ display:inline-block; padding:.3rem .8rem; border-radius:15px; font-size:.75rem; font-weight:600; text-transform:uppercase; margin-bottom:.5rem; }
.easy{ background:#d4edda; color:#155724; } .medium{ background:#fff3cd; color:#856404; } .hard{ background:#f8d7da; color:#721c24; }
.notebook-note{ background:#e9f7ef; padding:1rem; border-radius:8px; margin-bottom:1rem; border-left:4px solid #28a745; }
.badge{ display:inline-block; padding:.25rem .5rem; border-radius:6px; font-size:.75rem; font-weight:600; }
.badge-green{ background:#d1fae5; color:#065f46; } .badge-gray{ background:#e5e7eb; color:#374151; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# ФОЛБЭКИ (на случай таймаутов/лимитов) + кеш-обёртки

FALLBACK_QUESTIONS = {
    "questions": [
        {
            "question": "Как называется многочлен второй степени \\(ax^2 + bx + c\\)?",
            "options": ["A) Линейный", "B) Квадратичный", "C) Показательный", "D) Логарифмический"],
            "correct_answer": "B",
            "explanation": "Степень 2 → квадратичный многочлен.",
            "difficulty": "easy"
        },
        {
            "question": "Чему равна производная \\(x^2\\)?",
            "options": ["A) 2x", "B) x", "C) x^3", "D) 2"],
            "correct_answer": "A",
            "explanation": "Правило: \\((x^n)' = n x^{n-1}\\).",
            "difficulty": "easy"
        }
    ]
}

FALLBACK_TASKS = {
    "easy": [
        {
            "question": "Вычисли: \\(2^3 + 2^2\\)",
            "answer": "12",
            "solution": "Посчитаем: \\(2^3=8\\), \\(2^2=4\\), сумма \\(8+4=12\\).",
            "hint": "Возведи 2 в нужные степени и сложи."
        }
    ],
    "medium": [
        {
            "question": "Реши неравенство: \\(x - 3 \\ge 2\\)",
            "answer": "x>=5",
            "solution": "Перенесём 3: \\(x\\ge 5\\). Интервал: \\([5, \\inf)\\).",
            "hint": "Сложи к обеим частям 3."
        }
    ],
    "hard": [
        {
            "question": "Найди корни: \\(x^2-5x+6=0\\)",
            "answer": "2,3",
            "solution": "Факторизация: \\((x-2)(x-3)=0\\) → корни 2 и 3.",
            "hint": "Попробуй разложить на множители."
        }
    ]
}

@st.cache_data(ttl=3600, show_spinner=False)
def cached_generate_questions(topic, subject, grade, difficulty, model, temperature, max_tokens):
    return EnhancedAITutor().generate_adaptive_questions(topic, subject, grade, difficulty)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_generate_tasks(topic, subject, grade, perf_bucket, model, temperature, max_tokens):
    return EnhancedAITutor().generate_practice_tasks_enhanced(topic, subject, grade, perf_bucket)

# ────────────────────────────────────────────────────────────────────────────────
# Основной класс

class EnhancedAITutor:
    def __init__(self):
        self.youtube_api_key = YOUTUBE_API_KEY
        self.deepseek_api_key = DEEPSEEK_API_KEY
        self.playlists = PLAYLISTS
        self.config = APP_CONFIG
        self.deepseek_config = DEEPSEEK_CONFIG
        self.ui_config = UI_CONFIG

    def get_playlist_videos(self, playlist_id):
        """Загрузка видео из YouTube плейлиста (устойчивая версия)."""
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
                    "video_id": (sn.get("resourceId", {}) or {}).get("videoId"),
                    "description": (sn.get("description") or "")[:200] + ("..." if len(sn.get("description") or "") > 200 else ""),
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

    # ── DeepSeek вызов: устойчивый, с бэкоффом, строгим JSON и мягкими возвратами
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
            "max_tokens": self.deepseek_config["max_tokens"]
        }

        attempts = max(1, self.deepseek_config.get("retry_attempts", 2))
        base_read_timeout = max(8, int(self.deepseek_config.get("timeout", 15) / 2))

        for attempt in range(attempts):
            try:
                # Раздельные таймауты: connect=5s, read растёт: 8s, 12s, ...
                read_timeout = base_read_timeout + attempt * 4
                resp = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=(5, read_timeout)
                )

                if resp.status_code == 402:
                    # недостаточно средств — не шумим, а возвращаем маркер
                    return {"error": "402"}

                resp.raise_for_status()
                result = resp.json()
                content = result["choices"][0]["message"]["content"]

                # Стараемся распарсить JSON строго
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Попытка вытащить JSON-фрагмент из текста
                    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
                    if match:
                        try:
                            return json.loads(match.group(0))
                        except Exception:
                            pass
                    return {"content": content}

            except requests.exceptions.ReadTimeout:
                if attempt == attempts - 1:
                    return {"error": "timeout"}
            except requests.exceptions.Timeout:
                if attempt == attempts - 1:
                    return {"error": "timeout"}
            except requests.exceptions.HTTPError as e:
                if attempt == attempts - 1:
                    code = e.response.status_code if e.response is not None else "err"
                    return {"error": f"http_{code}"}
            except Exception as e:
                if attempt == attempts - 1:
                    return {"error": f"exception:{str(e)}"}

    def generate_adaptive_questions(self, topic, subject, grade, difficulty_level="medium"):
        difficulty_prompts = {
            "easy": "простые базовые вопросы для закрепления основ",
            "medium": "вопросы средней сложности для углубления понимания",
            "hard": "сложные вопросы для продвинутого изучения",
        }
        prompt = f"""
Создай {self.config["theory_questions_count"]} теоретических вопросов по теме "{topic}" для {grade}-го класса по предмету "{subject}".
Уровень сложности: {difficulty_prompts.get(difficulty_level, "средний")}.

Требования:
- Вопросы должны проверять понимание ключевых концепций
- Каждый вопрос с 4 вариантами ответа
- Один правильный ответ (буква A/B/C/D)
- Подробное объяснение правильного ответа
- Формулы должны быть в формате LaTeX (например, \\(x^2 + 2x + 1 = 0\\))

Верни результат строго в валидном JSON без комментариев и без '...':
{{
  "questions": [
    {{
      "question": "Текст вопроса с LaTeX: \\(...\\)",
      "options": ["A) вариант1", "B) вариант2", "C) вариант3", "D) вариант4"],
      "correct_answer": "A",
      "explanation": "Пояснение с формулами LaTeX: \\(...\\)",
      "difficulty": "{difficulty_level}"
    }}
  ]
}}
"""
        return self._call_deepseek_api(prompt)

    def generate_practice_tasks_enhanced(self, topic, subject, grade, user_performance=None):
        performance_adjustment = ""
        if user_performance is not None:
            if user_performance < 60:
                performance_adjustment = "Сделай акцент на более простые задачи с подробными объяснениями."
            elif user_performance > 85:
                performance_adjustment = "Добавь более сложные и нестандартные задачи."

        prompt = f"""
Составь практические задания по теме "{topic}" для {grade}-го класса по предмету "{subject}":
- {self.config["tasks_per_difficulty"]["easy"]} легкие задачи
- {self.config["tasks_per_difficulty"]["medium"]} средние задачи
- {self.config["tasks_per_difficulty"]["hard"]} сложные задачи

{performance_adjustment}

Для каждой задачи дай:
- Формулировку (с LaTeX, например, \\(x^2 + 2x + 1 = 0\\))
- Правильный ответ (текст/число, без LaTeX, например, "x >= 2, x < 3")
- Пошаговое решение (с LaTeX)
- Короткую подсказку (без LaTeX)

Верни строго валидный JSON (без комментариев и без '...'):
{{
  "easy": [
    {{
      "question": "Условие с LaTeX: \\(...\\)",
      "answer": "Правильный ответ",
      "solution": "Решение с LaTeX: \\(...\\)",
      "hint": "Подсказка без LaTeX"
    }}
  ],
  "medium": [
    {{
      "question": "Условие с LaTeX: \\(...\\)",
      "answer": "Правильный ответ",
      "solution": "Решение с LaTeX: \\(...\\)",
      "hint": "Подсказка без LaTeX"
    }}
  ],
  "hard": [
    {{
      "question": "Условие с LaTeX: \\(...\\)",
      "answer": "Правильный ответ",
      "solution": "Решение с LaTeX: \\(...\\)",
      "hint": "Подсказка без LaTeX"
    }}
  ]
}}
"""
        return self._call_deepseek_api(prompt)

# ────────────────────────────────────────────────────────────────────────────────
# РОУТИНГ И ЭКРАНЫ

def main():
    st.markdown('<div class="main-header"><h1>📚 AI Тьютор — Персональное обучение</h1></div>', unsafe_allow_html=True)

    # === Пользователь/облако прогресса (опционально) ===
    st.sidebar.markdown("### 👤 Пользователь")
    user_id = st.sidebar.text_input("Идентификатор (для облака)", placeholder="email или ник")

    # простой индикатор подключения Supabase (если ключи есть в config или secrets)
    sb_url = SUPABASE_URL if 'SUPABASE_URL' in globals() else None
    sb_key = SUPABASE_ANON_KEY if 'SUPABASE_ANON_KEY' in globals() else None
    try:
        if hasattr(st, "secrets"):
            sb_url = sb_url or st.secrets.get("SUPABASE_URL", None)
            sb_key = sb_key or st.secrets.get("SUPABASE_ANON_KEY", None)
    except Exception:
        pass
    sb_on = bool(sb_url and sb_key and user_id)

    if sb_on:
        st.sidebar.markdown('<span class="badge badge-green">Supabase: подключено</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="badge badge-gray">Supabase: локальное хранение</span>', unsafe_allow_html=True)

    tutor = EnhancedAITutor()
    # В utils.SessionManager должна быть поддержка user_id (иначе игнорируется)
    session = SessionManager(user_id=user_id if user_id else None)

    # === Выбор курса ===
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

    # === Прогресс ===
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Ваш прогресс")
    progress_data = session.get_progress()
    st.sidebar.metric("Пройдено тем", len(progress_data["completed_topics"]))
    chart_data = create_progress_chart_data(progress_data)
    if chart_data:
        st.sidebar.plotly_chart(chart_data, use_container_width=True)

    # === Роутинг ===
    stage = session.get_stage()
    if stage == 'video':
        display_video_content(tutor, session)
    elif stage == 'theory_test':
        show_theory_test(tutor, session)
    elif stage == 'practice':
        show_practice_stage(tutor, session)
    else:
        st.info("👆 Выберите предмет и класс в боковой панели, затем нажмите «Начать обучение».")

def display_video_content(tutor, session):
    videos = session.get_videos()
    if not videos:
        st.warning("Видео не загружены. Нажмите «Начать обучение» в сайдбаре.")
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
                session.set_stage('theory_test')
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

    if 'theory_questions' not in st.session_state:
        with st.spinner("Генерация вопросов..."):
            difficulty = session.get_adaptive_difficulty()
            # кешированная генерация
            qdata = cached_generate_questions(
                current_video['title'],
                session.get_subject(),
                session.get_grade(),
                difficulty,
                DEEPSEEK_CONFIG["model"],
                DEEPSEEK_CONFIG["temperature"],
                DEEPSEEK_CONFIG["max_tokens"],
            )

            parsed_ok = False
            if isinstance(qdata, dict) and "questions" in qdata:
                st.session_state.theory_questions = qdata.get("questions", [])
                parsed_ok = True
            elif isinstance(qdata, dict) and "content" in qdata:
                try:
                    match = re.search(r"\{.*\}", qdata["content"], flags=re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0))
                        st.session_state.theory_questions = parsed.get("questions", [])
                        parsed_ok = True
                except Exception:
                    parsed_ok = False

            if not parsed_ok:
                # Любая ошибка/таймаут/402 → мягкий фолбэк
                st.info("Показываю быстрый офлайн-набор вопросов (LLM не ответил вовремя).")
                st.session_state.theory_questions = FALLBACK_QUESTIONS["questions"]

            st.session_state.theory_answers = {}

    if st.session_state.theory_questions:
        for i, question in enumerate(st.session_state.theory_questions):
            diff = (question.get("difficulty") or "medium").lower()
            badge_text = tutor.ui_config["task_type_names"].get(diff, tutor.ui_config["task_type_names"]["medium"])
            st.markdown(
                f'<div class="task-card"><span class="difficulty-badge {diff}">{badge_text}</span>',
                unsafe_allow_html=True
            )
            st.markdown(f"**Вопрос {i+1}:** {question.get('question','')}", unsafe_allow_html=True)
            options = question.get('options', [])
            answer_key = f"theory_q_{i}"
            selected = st.radio("Выберите ответ:", options, key=answer_key, index=None)
            if selected:
                st.session_state.theory_answers[i] = selected[0]
            st.markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Вернуться к видео"):
                session.clear_theory_data()
                session.set_stage('video')
                log_user_action("return_to_video", {"video": current_video['title']})
                st.rerun()
        with col2:
            if st.button("Проверить ответы", type="primary"):
                if len(st.session_state.theory_answers) == len(st.session_state.theory_questions):
                    show_theory_results(tutor, session)
                else:
                    st.error("Пожалуйста, ответьте на все вопросы")
    else:
        st.error("Не удалось сгенерировать вопросы. Попробуйте снова.")

def show_theory_results(tutor, session):
    current_video = session.get_videos()[session.get_current_video_index()]
    topic_key = f"{session.get_subject()}_{session.get_grade()}_{current_video['title']}"

    st.markdown('<div class="progress-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Результаты тестирования")

    correct_count = 0
    total_questions = len(st.session_state.theory_questions)

    for i, question in enumerate(st.session_state.theory_questions):
        user_answer = st.session_state.theory_answers.get(i)
        correct_answer = question.get('correct_answer')
        if compare_answers(user_answer, correct_answer):
            correct_count += 1
            st.markdown('<div class="success-animation">', unsafe_allow_html=True)
            st.success(f"Вопрос {i+1}: Правильно!")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error(f"Вопрос {i+1}: Неправильно")
            st.info(f"**Объяснение:** {question.get('explanation','')}", unsafe_allow_html=True)

    score = calculate_score(correct_count, total_questions)
    st.metric("Ваш результат", f"{correct_count}/{total_questions} ({score:.0f}%)")

    # сохраняем строго по topic_key
    session.save_theory_score(topic_key, score)

    if score < tutor.config["theory_pass_threshold"]:
        st.warning("Рекомендуем пересмотреть видео для лучшего понимания темы")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Пересмотреть урок"):
            session.clear_theory_data()
            session.set_stage('video')
            log_user_action("rewatch_after_theory", {"video": current_video['title'], "score": score})
            st.rerun()
    with col2:
        if st.button("Начать практику", type="primary"):
            session.clear_theory_data()
            session.set_stage('practice')
            log_user_action("start_practice", {"video": current_video['title'], "theory_score": score})
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

def show_practice_stage(tutor, session):
    current_video = session.get_videos()[session.get_current_video_index()]
    st.header("💪 Практические задания")
    st.info(f"Тема: {current_video['title']}")

    st.markdown("""
    <div class="notebook-note">
        📝 <b>Совет:</b> Для сложных задач используйте тетрадь. Введите конечный ответ.
        Для неравенств — <code>x >= 2</code> или <code>[2, inf)</code>. Для нескольких условий — <code>and</code> или <code>,</code>.
    </div>
    """, unsafe_allow_html=True)

    if 'practice_tasks' not in st.session_state:
        with st.spinner("Генерация заданий..."):
            # адаптация по скору теории
            theory_score = session.get_theory_score(current_video['title'])
            perf_bucket = theory_score if theory_score is not None else -1

            tdata = cached_generate_tasks(
                current_video['title'],
                session.get_subject(),
                session.get_grade(),
                perf_bucket,
                DEEPSEEK_CONFIG["model"],
                DEEPSEEK_CONFIG["temperature"],
                DEEPSEEK_CONFIG["max_tokens"],
            )

            parsed_ok = False
            if isinstance(tdata, dict) and all(k in tdata for k in ("easy", "medium", "hard")):
                st.session_state.practice_tasks = tdata
                parsed_ok = True
            elif isinstance(tdata, dict) and "content" in tdata:
                try:
                    match = re.search(r"\{.*\}", tdata["content"], flags=re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0))
                        if all(k in parsed for k in ("easy", "medium", "hard")):
                            st.session_state.practice_tasks = parsed
                            parsed_ok = True
                except Exception:
                    parsed_ok = False

            if not parsed_ok:
                st.info("Показываю офлайн-набор заданий (LLM задержался).")
                st.session_state.practice_tasks = FALLBACK_TASKS

            st.session_state.task_attempts = {}
            st.session_state.completed_tasks = []
            st.session_state.current_task_type = 'easy'
            st.session_state.current_task_index = 0

    if any(len(st.session_state.practice_tasks.get(t, [])) for t in ['easy','medium','hard']):
        show_current_task(tutor, session)
    else:
        st.error("Нет заданий. Попробуйте позже.")

def show_current_task(tutor, session):
    task_types = ['easy', 'medium', 'hard']
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
            unsafe_allow_html=True
        )
        st.markdown(f"**Задание:** {current_index + 1} из {len(tasks_of_type)}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col1:
        st.markdown(
            f'<div class="task-card"><span class="difficulty-badge {current_type}">{tutor.ui_config["task_type_names"][current_type]}</span>',
            unsafe_allow_html=True
        )
        st.markdown(f"### Задание {current_index + 1}")
        st.markdown(current_task.get('question', ''), unsafe_allow_html=True)

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
            st.info(f"**Правильный ответ:** {current_task.get('answer','')}", unsafe_allow_html=True)
            st.info(f"**Решение:** {current_task.get('solution','')}", unsafe_allow_html=True)
            if st.button("Следующее задание"):
                move_to_next_task()

        # Подсказки (если были)
        if task_key in st.session_state and 'hints' in st.session_state[task_key]:
            st.markdown("### 💡 Подсказки:")
            for hint in st.session_state[task_key]['hints']:
                st.info(hint)

        st.markdown('</div>', unsafe_allow_html=True)

def check_answer(tutor, session, task, user_answer, task_key):
    st.session_state.task_attempts[task_key] = st.session_state.task_attempts.get(task_key, 0) + 1
    attempts = st.session_state.task_attempts[task_key]
    max_attempts = tutor.config["max_attempts_per_task"]

    is_correct = compare_answers((user_answer or "").strip().lower(), (task.get("answer") or "").strip().lower())

    if is_correct:
        st.markdown('<div class="success-animation">', unsafe_allow_html=True)
        st.success("Правильно! Отличная работа.")
        st.markdown('</div>', unsafe_allow_html=True)

        if task_key not in st.session_state.completed_tasks:
            st.session_state.completed_tasks.append(task_key)

        log_user_action("correct_answer", {"task_key": task_key, "attempts": attempts})

        if st.button("Следующее задание"):
            move_to_next_task()
    else:
        if attempts < max_attempts:
            st.error(f"Неправильно. Попытка {attempts} из {max_attempts}")
            # Лёгкий фолбэк-подсказки (без DeepSeek, чтобы не тормозить экран)
            hint = "Сопоставь выражение с известной формулой/свойством и проверь преобразования."
            try:
                if DEEPSEEK_ENABLED:
                    # Быстрый однопроходный запрос (без жёстких зависимостей)
                    hint_resp = EnhancedAITutor()._call_deepseek_api(f"""
Студент решал задачу: "{task.get('question','')}"
Правильный ответ: "{task.get('answer','')}"
Ответ студента: "{user_answer}"
Дай краткую подсказку (1-2 предложения) без LaTeX и без полного решения.
""")
                    if isinstance(hint_resp, dict) and 'content' in hint_resp and hint_resp['content'].strip():
                        hint = hint_resp['content'].strip()
            except Exception:
                pass

            if task_key not in st.session_state:
                st.session_state[task_key] = {'hints': []}
            st.session_state[task_key]['hints'].append(hint)
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

def show_practice_completion(tutor, session):
    videos = session.get_videos()
    if not videos:
        st.info("Практика завершена.")
        return

    current_video = videos[session.get_current_video_index()]
    topic_key = f"{session.get_subject()}_{session.get_grade()}_{current_video['title']}"

    st.markdown('<div class="progress-card">', unsafe_allow_html=True)
    st.header("Практика завершена!")

    task_types = ['easy', 'medium', 'hard']
    total_tasks = sum(len(st.session_state.practice_tasks.get(t, [])) for t in task_types)
    completed = len(st.session_state.completed_tasks)
    score = calculate_score(completed, total_tasks) if total_tasks else 0

    st.success(f"Выполнено {completed} из {total_tasks} заданий ({score:.0f}%)")

    # сохраняем прогресс по topic_key (единый ключ)
    session.save_practice_score(topic_key, completed, total_tasks)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Изучить новую тему"):
            if session.next_video():
                session.set_stage('video')
                for k in ["practice_tasks", "task_attempts", "completed_tasks", "current_task_type", "current_task_index"]:
                    if k in st.session_state:
                        del st.session_state[k]
                log_user_action("next_topic", {"video_index": session.get_current_video_index()})
                st.rerun()
            else:
                st.info("Все темы курса пройдены!")
    with col2:
        if st.button("Вернуться к выбору курса"):
            session.set_stage('selection')
            for k in ["practice_tasks", "task_attempts", "completed_tasks", "current_task_type", "current_task_index"]:
                if k in st.session_state:
                    del st.session_state[k]
            log_user_action("return_to_selection", {})
            st.rerun()

    st.markdown(generate_progress_report(session.get_progress(), topic_key), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

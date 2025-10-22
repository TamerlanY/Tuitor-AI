import streamlit as st
import requests
import json
import os
from datetime import datetime
import plotly.express as px
import pandas as pd

from config import PLAYLISTS, APP_CONFIG, DEEPSEEK_CONFIG, UI_CONFIG, SUPABASE_URL, SUPABASE_ANON_KEY
from utils import (
    compare_answers, calculate_score, generate_progress_report,
    get_subject_emoji, SessionManager, create_progress_chart_data,
    log_user_action
)

# ----------------------------- set_page_config ДОЛЖЕН быть первым -----------------------------
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
    st.error("Не задан YOUTUBE_API_KEY. Укажи в .env или в Secrets.")
    st.stop()

# DeepSeek может быть пустым — тогда генерацию отключим точечно
DEEPSEEK_ENABLED = bool(DEEPSEEK_API_KEY)

# ----------------------------- MathJax -----------------------------
st.markdown("""
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
<script>
    MathJax.Hub.Config({
        tex2jax: { inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']], processEscapes: true }
    });
    MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
</script>
""", unsafe_allow_html=True)

# ----------------------------- CSS -----------------------------
st.markdown("""
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
""", unsafe_allow_html=True)

# ----------------------------- helpers -----------------------------
def _strip_code_fences(text: str) -> str:
    """Убирает ```json ... ``` и подобные ограждения, если модель их вернула."""
    if not isinstance(text, str):
        return text
    t = text.strip()
    if t.startswith("```"):
        t = t.lstrip("`")
        # после среза префикса может остаться "json\n"
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()

def _safe_json_from_text(text: str) -> dict:
    """Пробует распарсить JSON, предварительно убрав code fences."""
    cleaned = _strip_code_fences(text)
    return json.loads(cleaned)

# ----------------------------- core class -----------------------------
class EnhancedAITutor:
    def __init__(self):
        self.youtube_api_key = YOUTUBE_API_KEY
        self.deepseek_api_key = DEEPSEEK_API_KEY
        self.playlists = PLAYLISTS
        self.config = APP_CONFIG
        self.deepseek_config = DEEPSEEK_CONFIG
        self.ui_config = UI_CONFIG

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
                res = sn.get("resourceId", {}) or {}
                thumbs = sn.get("thumbnails", {}) or {}
                thumb = thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}
                video = {
                    "title": sn.get("title", "Без названия"),
                    "video_id": res.get("videoId"),
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

    def _call_deepseek_api(self, prompt, expect_json=False):
        """
        Универсальный вызов DeepSeek.
        - expect_json=True включает response_format=json_object и строгий JSON-парсинг.
        - Возвращает dict. В случае не-JSON может вернуть {"content": "..."}.
        - При ошибке печатает статус и тело ответа (обрезанное), чтобы проще было дебажить.
        """
        if not DEEPSEEK_ENABLED:
            return {"error": "deepseek_disabled"}

        headers = {"Authorization": f"Bearer {self.deepseek_api_key}", "Content-Type": "application/json"}
        data = {
            "model": self.deepseek_config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.deepseek_config["temperature"],
            "max_tokens": self.deepseek_config["max_tokens"],
        }
        if expect_json:
            data["response_format"] = {"type": "json_object"}

        for attempt in range(self.deepseek_config["retry_attempts"]):
            try:
                resp = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers, json=data, timeout=self.deepseek_config["timeout"]
                )

                # Явно ловим 402 — нет баланса
                if resp.status_code == 402:
                    try:
                        body = resp.json()
                    except Exception:
                        body = {"raw": resp.text}
                    st.warning("DeepSeek вернул 402 (недостаточно средств).")
                    return {"error": "402", "body": body}

                if resp.status_code != 200:
                    body_text = resp.text[:2000]
                    st.error(f"DeepSeek HTTP {resp.status_code}. Тело ответа ниже:")
                    st.code(body_text)
                    return {"error": f"http_{resp.status_code}", "body": body_text}

                result = resp.json()
                content = result["choices"][0]["message"]["content"]

                if expect_json:
                    try:
                        return _safe_json_from_text(content)
                    except json.JSONDecodeError:
                        # отдаём сырец в ответе — его покажем на экране
                        return {"content": content}
                else:
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
                    return {"error": f"http_{e.response.status_code}"}
            except Exception as e:
                if attempt == self.deepseek_config["retry_attempts"] - 1:
                    st.error(f"Ошибка API DeepSeek: {str(e)}")
                    return {"error": str(e)}

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
- Один правильный ответ
- Подробное объяснение правильного ответа
- Формулы должны быть в формате LaTeX (например, \\(x^2 + 2x + 1 = 0\\))

Верни результат строго в валидном JSON (без комментариев и многоточий). НЕ используй '...':
{{
  "questions": [
    {{
      "question": "Текст вопроса, формулы в LaTeX: \\(...\\)",
      "options": ["A) вариант1", "B) вариант2", "C) вариант3", "D) вариант4"],
      "correct_answer": "A",
      "explanation": "Подробное объяснение с примерами, формулы в LaTeX: \\(...\\)",
      "difficulty": "{difficulty_level}"
    }}
  ]
}}
"""
        # просим строго JSON
        return self._call_deepseek_api(prompt, expect_json=True)

    def generate_practice_tasks_enhanced(self, topic, subject, grade, user_performance=None):
        performance_adjustment = ""
        if user_performance is not None:
            if user_performance < 60:
                performance_adjustment = "Сделай акцент на более простые задания с подробными объяснениями."
            elif user_performance > 85:
                performance_adjustment = "Включи более сложные и нестандартные задания."

        prompt = f"""
Составь практические задания по теме "{topic}" для {grade}-го класса по предмету "{subject}":
- {self.config["tasks_per_difficulty"]["easy"]} легкие задачи (базовый уровень)
- {self.config["tasks_per_difficulty"]["medium"]} средние задачи (стандартный уровень)  
- {self.config["tasks_per_difficulty"]["hard"]} сложные задачи (повышенный уровень)

{performance_adjustment}

Для каждой задачи предоставь:
- Четкую формулировку, формулы в формате LaTeX (например, \\(x^2 + 2x + 1 = 0\\))
- Правильный ответ (текст или число, без LaTeX, например, "x >= 2, x < 3")
- Пошаговое решение с формулами в LaTeX
- Полезную подсказку (не раскрывающую полное решение, без LaTeX)

Верни результат строго в ВАЛИДНОМ JSON (без комментариев и многоточий). НЕ используй '...'.
Каждый массив должен содержать реальные объекты задач:
{{
  "easy": [
    {{
      "question": "Условие задачи с LaTeX: \\(...\\)",
      "answer": "Правильный ответ",
      "solution": "Пошаговое решение с LaTeX: \\(...\\)",
      "hint": "Короткая подсказка без LaTeX"
    }}
  ],
  "medium": [
    {{
      "question": "Условие задачи с LaTeX: \\(...\\)",
      "answer": "Правильный ответ",
      "solution": "Пошаговое решение с LaTeX: \\(...\\)",
      "hint": "Короткая подсказка без LaTeX"
    }}
  ],
  "hard": [
    {{
      "question": "Условие задачи с LaTeX: \\(...\\)",
      "answer": "Правильный ответ",
      "solution": "Пошаговое решение с LaTeX: \\(...\\)",
      "hint": "Короткая подсказка без LaTeX"
    }}
  ]
}}
"""
        return self._call_deepseek_api(prompt, expect_json=True)

# ----------------------------- UI Flow -----------------------------
def main():
    st.markdown('<div class="main-header"><h1>📚 AI Тьютор - Персональное обучение</h1></div>', unsafe_allow_html=True)

    # ---- USER ID для облачного прогресса ----
    st.sidebar.markdown("### 👤 Пользователь")
    user_id = st.sidebar.text_input("Идентификатор (для облака)", placeholder="например, email или ник")
    sb_on = bool((SUPABASE_URL or (hasattr(st, "secrets") and st.secrets.get("SUPABASE_URL"))) and
                 (SUPABASE_ANON_KEY or (hasattr(st, "secrets") and st.secrets.get("SUPABASE_ANON_KEY"))))
    if user_id and sb_on:
        st.sidebar.markdown('<span class="badge badge-green">Supabase: подключено</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="badge badge-gray">Supabase: локальное хранение</span>', unsafe_allow_html=True)

    # ---- Диагностика DeepSeek ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 Диагностика LLM")
    if st.sidebar.button("Проверить DeepSeek"):
        if not DEEPSEEK_ENABLED:
            st.sidebar.error("DEEPSEEK_API_KEY не задан (генерация выключена).")
        else:
            test_prompt = """
Верни строго валидный JSON:
{
  "ok": true,
  "msg": "ping"
}
"""
            t = EnhancedAITutor()
            resp = t._call_deepseek_api(test_prompt, expect_json=True)
            if isinstance(resp, dict) and resp.get("ok") is True:
                st.sidebar.success(f"DeepSeek OK: {resp}")
            else:
                st.sidebar.error("DeepSeek ответил не-JSON или ошибкой. Подробности ниже (в основной области):")
                st.write("Ответ диагностики:", resp)

    tutor = EnhancedAITutor()
    session = SessionManager(user_id=user_id or None)

    # Боковая панель — выбор курса
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

    # Прогресс блок
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Ваш прогресс")
    progress_data = session.get_progress()
    st.sidebar.metric("Пройдено тем", len(progress_data["completed_topics"]))
    chart_data = create_progress_chart_data(progress_data)
    if chart_data:
        st.sidebar.plotly_chart(chart_data, use_container_width=True)

    # Роутинг
    stage = session.get_stage()
    if stage == 'video':
        display_video_content(tutor, session)
    elif stage == 'theory_test':
        show_theory_test(tutor, session)
    elif stage == 'practice':
        show_practice_stage(tutor, session)
    else:
        st.info("👆 Выберите предмет и класс в боковой панели, затем нажмите 'Начать обучение'")

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
            questions_data = tutor.generate_adaptive_questions(
                current_video['title'], session.get_subject(), session.get_grade(), difficulty
            )

            # Явные ошибки API и подробности
            if isinstance(questions_data, dict) and questions_data.get("error"):
                err = questions_data.get("error")
                st.error(f"Не удалось сгенерировать вопросы. Код: {err}")
                body = questions_data.get("body")
                content = questions_data.get("content")

                if body:
                    st.info("Тело ответа DeepSeek (обрезано):")
                    st.code(str(body)[:2000])
                if content:
                    st.info("Содержимое контента (обрезано):")
                    st.code(str(content)[:2000])

                st.session_state.theory_questions = []
            else:
                # Парсинг результата
                if isinstance(questions_data, dict) and 'questions' in questions_data:
                    st.session_state.theory_questions = questions_data.get('questions', [])
                elif isinstance(questions_data, dict) and 'content' in questions_data:
                    try:
                        parsed = _safe_json_from_text(questions_data['content'])
                        st.session_state.theory_questions = parsed.get('questions', [])
                    except json.JSONDecodeError:
                        st.error("Модель прислала не-JSON. Ниже — сырой ответ для отладки:")
                        st.code(questions_data['content'][:1200])
                        st.session_state.theory_questions = []
                else:
                    st.session_state.theory_questions = []

            st.session_state.theory_answers = {}

    if st.session_state.theory_questions:
        for i, question in enumerate(st.session_state.theory_questions):
            diff = (question.get("difficulty") or "medium").lower()
            badge_text = tutor.ui_config["task_type_names"].get(diff, tutor.ui_config["task_type_names"]["medium"])
            st.markdown(f'<div class="task-card"><span class="difficulty-badge {diff}">{badge_text}</span>', unsafe_allow_html=True)
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
            theory_score = session.get_theory_score(current_video['title'])
            tasks_data = tutor.generate_practice_tasks_enhanced(
                current_video['title'], session.get_subject(), session.get_grade(), theory_score
            )
            if isinstance(tasks_data, dict) and 'content' in tasks_data:
                try:
                    tasks_data = _safe_json_from_text(tasks_data['content'])
                except Exception:
                    tasks_data = {"easy": [], "medium": [], "hard": []}
            if isinstance(tasks_data, dict) and tasks_data.get("error") in ("402", "deepseek_disabled"):
                st.error("Не удалось сгенерировать задания (DeepSeek недоступен).")
                st.session_state.practice_tasks = {"easy": [], "medium": [], "hard": []}
            else:
                st.session_state.practice_tasks = tasks_data
            st.session_state.task_attempts = {}
            st.session_state.completed_tasks = []
            st.session_state.current_task_type = 'easy'
            st.session_state.current_task_index = 0

    if any(len(st.session_state.practice_tasks.get(t, [])) for t in ['easy','medium','hard']):
        show_current_task(tutor, session)
    else:
        st.error("Нет заданий. Попробуйте позже или пополните баланс DeepSeek.")

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
        st.markdown(f'<span class="difficulty-badge {current_type}">{tutor.ui_config["task_type_names"][current_type]}</span>', unsafe_allow_html=True)
        st.markdown(f"**Задание:** {current_index + 1} из {len(tasks_of_type)}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col1:
        st.markdown(f'<div class="task-card"><span class="difficulty-badge {current_type}">{tutor.ui_config["task_type_names"][current_type]}</span>', unsafe_allow_html=True)
        st.markdown(f"### Задание {current_index + 1}")
        st.markdown(current_task.get('question', ''), unsafe_allow_html=True)
        user_answer = st.text_input("Ваш ответ:", key=f"answer_{task_key}")
        attempts = st.session_state.task_attempts.get(task_key, 0)
        max_attempts = tutor.config["max_attempts_per_task"]
        if attempts < max_attempts:
            col_check, col_skip = st.columns([1, 1])
            with col_check:
                if st.button("Проверить ответ", type="primary"):
                    if user_answer.strip():
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
            with st.spinner("Получаю подсказку..."):
                hint = "Подумай, какие свойства применяются к этой формуле."
                if DEEPSEEK_ENABLED:
                    try:
                        # лёгкий промпт для подсказки
                        hint_resp = tutor._call_deepseek_api(f"""
Студент решал задачу: "{task.get('question','')}"
Правильный ответ: "{task.get('answer','')}"
Ответ студента: "{user_answer}"
Дай краткую подсказку (1-2 предложения) без LaTeX.
""")
                        if isinstance(hint_resp, dict) and 'content' in hint_resp:
                            hint = hint_resp['content']
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

if __name__ == "__main__":
    main()

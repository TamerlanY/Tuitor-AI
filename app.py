import os
import json
from datetime import datetime

import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # (может пригодиться)

# ======= Конфиги (без ключей) =======
from config import PLAYLISTS, APP_CONFIG, DEEPSEEK_CONFIG, UI_CONFIG
from utils import (
    compare_answers, calculate_score, generate_progress_report,
    get_subject_emoji, SessionManager, create_progress_chart_data,
    log_user_action
)

# -------- set_page_config ДОЛЖЕН быть самым первым вызовом Streamlit --------
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["initial_sidebar_state"],
)

# === РЕЗОЛВ КЛЮЧЕЙ: .env/OS → st.secrets ===
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
try:
    if (not YOUTUBE_API_KEY) and hasattr(st, "secrets") and "YOUTUBE_API_KEY" in st.secrets:
        YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    if (not DEEPSEEK_API_KEY) and hasattr(st, "secrets") and "DEEPSEEK_API_KEY" in st.secrets:
        DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    pass

# Жёсткая проверка YouTube (без него даже видео не загрузить)
if not YOUTUBE_API_KEY:
    st.error("Не задан YOUTUBE_API_KEY. Задай его в .env или в Secrets на хостинге.")
    st.stop()

# Подключение MathJax
st.markdown("""
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
<script>
  MathJax.Hub.Config({
    tex2jax: {inlineMath: [['\\(','\\)']], displayMath: [['\\[','\\]']], processEscapes: true}
  });
  MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
</script>
""", unsafe_allow_html=True)

# CSS
st.markdown("""
<style>
  .main-header{ text-align:center; padding:2rem; background:linear-gradient(90deg,#667eea,#764ba2); border-radius:10px; color:#fff; margin-bottom:2rem;}
  .progress-card{ background:#fff; padding:1.5rem; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.1); margin:1rem 0;}
  .task-card{ background:#f8f9fa; padding:1.5rem; border-radius:8px; border-left:4px solid #007bff; margin:1rem 0;}
  .success-animation{ animation:pulse .5s ease-in-out;}
  @keyframes pulse{0%{transform:scale(1)}50%{transform:scale(1.05)}100%{transform:scale(1)}}
  .difficulty-badge{ display:inline-block; padding:.3rem .8rem; border-radius:15px; font-size:.75rem; font-weight:600; text-transform:uppercase; margin-bottom:.5rem;}
  .easy{background:#d4edda;color:#155724}.medium{background:#fff3cd;color:#856404}.hard{background:#f8d7da;color:#721c24}
  .notebook-note{ background:#e9f7ef; padding:1rem; border-radius:8px; margin-bottom:1rem; border-left:4px solid #28a745}
  .muted{opacity:.7}
</style>
""", unsafe_allow_html=True)


class EnhancedAITutor:
    def __init__(self):
        self.youtube_api_key = YOUTUBE_API_KEY
        self.deepseek_api_key = DEEPSEEK_API_KEY
        self.playlists = PLAYLISTS
        self.config = APP_CONFIG
        self.deepseek_config = DEEPSEEK_CONFIG
        self.ui_config = UI_CONFIG

        # флаг доступности LLM
        self.deepseek_enabled = bool(self.deepseek_api_key)
        # позволим отключать LLM глобально при ошибке 402
        if "deepseek_enabled" not in st.session_state:
            st.session_state.deepseek_enabled = self.deepseek_enabled
        self.deepseek_enabled = st.session_state.deepseek_enabled

    # ---------- YouTube ----------
    def get_playlist_videos(self, playlist_id):
        """Безопасная загрузка плейлиста YouTube (устойчиво к отсутствующим превью)."""
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
                thumb_obj = thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}
                thumb_url = thumb_obj.get("url", "")
                desc = sn.get("description", "") or ""
                if len(desc) > 200:
                    desc = desc[:200] + "..."
                videos.append({
                    "title": sn.get("title", "Без названия"),
                    "video_id": sn.get("resourceId", {}).get("videoId", ""),
                    "description": desc,
                    "thumbnail": thumb_url,
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

    # ---------- LLM вызовы ----------
    def _llm_unavailable_questions(self):
        return {"questions": []}

    def _llm_unavailable_tasks(self):
        return {"easy": [], "medium": [], "hard": []}

    def _call_deepseek_api(self, prompt):
        """Единая обёртка c обработкой 402 (Payment Required)."""
        if not self.deepseek_enabled:
            return {"error": "llm_disabled"}

        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json",
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
                    headers=headers,
                    json=data,
                    timeout=self.deepseek_config["timeout"]
                )
                # Специальная ветка для 402
                if resp.status_code == 402:
                    st.warning("DeepSeek вернул 402 (нет кредитов/подписки). Генерация будет отключена до обновления ключа.")
                    st.session_state.deepseek_enabled = False
                    self.deepseek_enabled = False
                    return {"error": "payment_required"}

                resp.raise_for_status()
                result = resp.json()
                content = result["choices"][0]["message"]["content"]
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
        if not self.deepseek_enabled:
            return self._llm_unavailable_questions()

        difficulty_prompts = {
            "easy": "простые базовые вопросы для закрепления основ",
            "medium": "вопросы средней сложности для углубления понимания",
            "hard": "сложные вопросы для продвинутого изучения"
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

Верни результат строго в валидном JSON (без комментариев и многоточий):
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
        return self._call_deepseek_api(prompt)

    def generate_practice_tasks_enhanced(self, topic, subject, grade, user_performance=None):
        if not self.deepseek_enabled:
            return self._llm_unavailable_tasks()

        perf = ""
        if user_performance is not None:
            if user_performance < 60:
                perf = "Сделай акцент на более простые задания с подробными объяснениями."
            elif user_performance > 85:
                perf = "Включи более сложные и нестандартные задания."

        prompt = f"""
Составь практические задания по теме "{topic}" для {grade}-го класса по предмету "{subject}":
- {self.config["tasks_per_difficulty"]["easy"]} легкие задачи
- {self.config["tasks_per_difficulty"]["medium"]} средние задачи
- {self.config["tasks_per_difficulty"]["hard"]} сложные задачи

{perf}

Для каждой задачи укажи:
- формулировку (с LaTeX)
- ответ (без LaTeX, например "x >= 2, x < 3")
- пошаговое решение (с LaTeX)
- короткую подсказку (без LaTeX)

Верни строго валидный JSON (без '...'):
{{
  "easy": [{{"question":"...","answer":"...","solution":"...","hint":"..."}}],
  "medium": [{{"question":"...","answer":"...","solution":"...","hint":"..."}}],
  "hard": [{{"question":"...","answer":"...","solution":"...","hint":"..."}}]
}}
"""
        return self._call_deepseek_api(prompt)

    def get_hint(self, question, user_answer, correct_answer):
        if not self.deepseek_enabled:
            return "Подсказки отключены (нет DEEPSEEK_API_KEY или 402)."

        prompt = f"""
Студент решал задачу: "{question}"
Правильный ответ: "{correct_answer}"
Ответ студента: "{user_answer}"

Дай краткую подсказку (1-2 предложения), без раскрытия полного решения и без LaTeX.
Если студент написал слова вместо символов (например 'больше или равно'), укажи, что нужно использовать >=, <= и т.п.
"""
        resp = self._call_deepseek_api(prompt)
        if isinstance(resp, dict) and "content" in resp:
            return resp["content"]
        return "Попробуйте ещё раз; проверьте запись условий и символов (>=, <=, <, >)."


# =================== UI ===================

def main():
    st.markdown('<div class="main-header"><h1>📚 AI Тьютор — персональное обучение</h1></div>', unsafe_allow_html=True)
    tutor = EnhancedAITutor()
    session = SessionManager()

    # Sidebar
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
        chart_data = create_progress_chart_data(progress_data)
        if chart_data:
            st.plotly_chart(chart_data, use_container_width=True)

        st.markdown("---")
        # статус LLM
        if not tutor.deepseek_enabled:
            st.markdown("🧠 LLM: **отключён** (нет ключа или 402). Доступно: видео и учёт прогресса.", unsafe_allow_html=True)
        else:
            st.markdown("🧠 LLM: **включён**", unsafe_allow_html=True)

    # Main
    stage = session.get_stage()
    if stage == "video":
        display_video_content(tutor, session)
    elif stage == "theory_test":
        if tutor.deepseek_enabled:
            show_theory_test(tutor, session)
        else:
            st.info("Генерация теста недоступна (LLM отключён). Вернитесь к видео.")
    elif stage == "practice":
        if tutor.deepseek_enabled:
            show_practice_stage(tutor, session)
        else:
            st.info("Генерация практики недоступна (LLM отключён). Вернитесь к видео.")
    else:
        st.info("👆 Выберите предмет и класс в боковой панели, затем нажмите «Начать обучение».")


def display_video_content(tutor, session):
    videos = session.get_videos()
    if not videos:
        st.warning("Видео из плейлиста не загружены. Попробуйте перезагрузить страницу.")
        return

    current_video = videos[session.get_current_video_index()]
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header(f"📺 {current_video['title']}")
        if current_video.get("video_id"):
            st.video(f"https://www.youtube.com/watch?v={current_video['video_id']}")
        else:
            st.info("Видео-ID не найден. Откройте ролик вручную на YouTube.")
        if current_video.get("description"):
            with st.expander("Описание урока"):
                st.write(current_video["description"])

    with col2:
        st.markdown('<div class="progress-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Текущий урок")
        st.info(f"Урок {session.get_current_video_index() + 1} из {len(videos)}")
        st.progress((session.get_current_video_index() + 1) / len(videos))

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            disabled = not st.session_state.get("deepseek_enabled", True)
            if st.button("Готов к тесту", type="primary", disabled=disabled):
                session.set_stage("theory_test")
                log_user_action("start_theory_test", {"video": current_video["title"]})
                st.rerun()
            if disabled:
                st.caption("LLM отключён — тест недоступен.")

        with col_btn2:
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


def show_theory_test(tutor, session):
    current_video = session.get_videos()[session.get_current_video_index()]
    st.header("📝 Тест по теории")
    st.info(f"Тема: {current_video['title']}")

    if "theory_questions" not in st.session_state:
        with st.spinner("Генерация вопросов..."):
            difficulty = session.get_adaptive_difficulty()
            data = tutor.generate_adaptive_questions(
                current_video["title"], session.get_subject(), session.get_grade(), difficulty
            )
            if isinstance(data, dict) and "content" in data:
                try:
                    data = json.loads(data["content"])
                except Exception:
                    data = {"questions": []}
            st.session_state.theory_questions = data.get("questions", [])
            st.session_state.theory_answers = {}

    if not st.session_state.theory_questions:
        st.error("Не удалось сгенерировать вопросы (возможно, LLM отключён или 402).")
        if st.button("← Вернуться к видео"):
            session.clear_theory_data()
            session.set_stage("video")
            st.rerun()
        return

    for i, q in enumerate(st.session_state.theory_questions):
        diff = (q.get("difficulty") or "medium").lower()
        badge_text = tutor.ui_config["task_type_names"].get(diff, tutor.ui_config["task_type_names"]["medium"])
        st.markdown(
            f'<div class="task-card"><span class="difficulty-badge {diff}">{badge_text}</span>',
            unsafe_allow_html=True
        )
        st.markdown(f"**Вопрос {i+1}:** {q.get('question','')}", unsafe_allow_html=True)
        opts = q.get("options", [])
        selected = st.radio("Выберите ответ:", opts, key=f"theory_q_{i}", index=None)
        if selected:
            st.session_state.theory_answers[i] = selected[0]
        st.markdown('</div>', unsafe_allow_html=True)

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


def show_theory_results(tutor, session):
    current_video = session.get_videos()[session.get_current_video_index()]
    topic_key = f"{session.get_subject()}_{session.get_grade()}_{current_video['title']}"

    st.markdown('<div class="progress-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Результаты тестирования")

    correct = 0
    total = len(st.session_state.theory_questions)
    for i, q in enumerate(st.session_state.theory_questions):
        ua = st.session_state.theory_answers.get(i)
        ca = q.get("correct_answer")
        if compare_answers(ua, ca):
            correct += 1
            st.markdown('<div class="success-animation">', unsafe_allow_html=True)
            st.success(f"Вопрос {i+1}: Правильно!")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error(f"Вопрос {i+1}: Неправильно")
            st.info(f"**Объяснение:** {q.get('explanation','')}", unsafe_allow_html=True)

    score = calculate_score(correct, total)
    st.metric("Ваш результат", f"{correct}/{total} ({score:.0f}%)")
    session.save_theory_score(topic_key, score)

    if score < tutor.config["theory_pass_threshold"]:
        st.warning("Рекомендуем пересмотреть видео для лучшего понимания темы")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Пересмотреть урок"):
            session.clear_theory_data()
            session.set_stage("video")
            st.rerun()
    with col2:
        disabled = not st.session_state.get("deepseek_enabled", True)
        if st.button("Начать практику", type="primary", disabled=disabled):
            session.clear_theory_data()
            session.set_stage("practice")
            st.rerun()
        if disabled:
            st.caption("LLM отключён — практика недоступна.")
    st.markdown('</div>', unsafe_allow_html=True)


def show_practice_stage(tutor, session):
    current_video = session.get_videos()[session.get_current_video_index()]
    st.header("💪 Практические задания")
    st.info(f"Тема: {current_video['title']}")

    st.markdown("""
    <div class="notebook-note">
      📝 <b>Совет:</b> Введите итоговый ответ. Для неравенств пишите <code>x >= 2</code>, <code>[2, inf)</code>, 
      или <code>x >= 2, x < 5</code>. Используйте символы, а не слова.
    </div>
    """, unsafe_allow_html=True)

    if "practice_tasks" not in st.session_state:
        with st.spinner("Генерация заданий..."):
            theory_score = session.get_theory_score(current_video["title"])
            data = tutor.generate_practice_tasks_enhanced(
                current_video["title"], session.get_subject(), session.get_grade(), theory_score
            )
            if isinstance(data, dict) and "content" in data:
                try:
                    data = json.loads(data["content"])
                except Exception:
                    data = {"easy": [], "medium": [], "hard": []}
            st.session_state.practice_tasks = data
        st.session_state.task_attempts = {}
        st.session_state.completed_tasks = []
        st.session_state.current_task_type = "easy"
        st.session_state.current_task_index = 0

    if not any(st.session_state.practice_tasks.get(t, []) for t in ["easy", "medium", "hard"]):
        st.error("Не удалось сгенерировать задания (возможно, LLM отключён или 402).")
        if st.button("← Вернуться к видео"):
            session.clear_practice_data()
            session.set_stage("video")
            st.rerun()
        return

    show_current_task(tutor, session)


def show_current_task(tutor, session):
    task_types = ["easy", "medium", "hard"]
    ct = st.session_state.current_task_type
    idx = st.session_state.current_task_index
    tasks = st.session_state.practice_tasks.get(ct, [])

    if idx >= len(tasks):
        t_index = task_types.index(ct)
        if t_index < len(task_types) - 1:
            st.session_state.current_task_type = task_types[t_index + 1]
            st.session_state.current_task_index = 0
            st.rerun()
        else:
            show_practice_completion(tutor, session)
            return

    task = tasks[idx]
    tkey = f"{ct}_{idx}"

    total_tasks = sum(len(st.session_state.practice_tasks.get(t, [])) for t in task_types)
    completed = len(st.session_state.completed_tasks)

    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown('<div class="progress-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Прогресс")
        st.progress(completed / total_tasks if total_tasks else 0)
        st.metric("Выполнено", f"{completed}/{total_tasks}")
        st.markdown(f'<span class="difficulty-badge {ct}">{tutor.ui_config["task_type_names"][ct]}</span>', unsafe_allow_html=True)
        st.markdown(f"**Задание:** {idx + 1} из {len(tasks)}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col1:
        st.markdown(f'<div class="task-card"><span class="difficulty-badge {ct}">{tutor.ui_config["task_type_names"][ct]}</span>', unsafe_allow_html=True)
        st.markdown(f"### Задание {idx + 1}")
        st.markdown(task.get("question", ""), unsafe_allow_html=True)

        user_answer = st.text_input("Ваш ответ:", key=f"answer_{tkey}")
        attempts = st.session_state.task_attempts.get(tkey, 0)
        max_attempts = tutor.config["max_attempts_per_task"]

        if attempts < max_attempts:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Проверить ответ", type="primary"):
                    if (user_answer or "").strip():
                        check_answer(tutor, session, task, user_answer, tkey)
                    else:
                        st.error("Введите ответ!")
            with col_b:
                if st.button("Пропустить"):
                    log_user_action("skip_task", {"task_key": tkey})
                    move_to_next_task()
        else:
            st.error(f"Исчерпаны все попытки ({max_attempts})")
            st.info(f"**Правильный ответ:** {task.get('answer','')}", unsafe_allow_html=True)
            st.info(f"**Решение:** {task.get('solution','')}", unsafe_allow_html=True)
            if st.button("Следующее задание"):
                move_to_next_task()

        # подсказки
        if tkey in st.session_state and "hints" in st.session_state[tkey]:
            st.markdown("### 💡 Подсказки:")
            for hint in st.session_state[tkey]["hints"]:
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
                hint = tutor.get_hint(task.get("question", ""), user_answer, task.get("answer", ""))
                if task_key not in st.session_state:
                    st.session_state[task_key] = {"hints": []}
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

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Изучить новую тему"):
            if session.next_video():
                session.set_stage("video")
                session.clear_practice_data()
                log_user_action("next_topic", {"video_index": session.get_current_video_index()})
                st.rerun()
            else:
                st.info("Все темы курса пройдены!")

    with col2:
        if st.button("Вернуться к выбору курса"):
            session.set_stage("selection")
            session.clear_practice_data()
            log_user_action("return_to_selection", {})
            st.rerun()

    st.markdown(generate_progress_report(session.get_progress(), topic_key), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

# app.py
import os
import json
from datetime import datetime

import requests
import streamlit as st
import plotly.express as px  # noqa: F401 (используется в utils)
import pandas as pd         # noqa: F401 (используется в utils)

from config import PLAYLISTS, APP_CONFIG, DEEPSEEK_CONFIG, UI_CONFIG
from utils import (
    compare_answers, calculate_score, generate_progress_report,
    get_subject_emoji, SessionManager, create_progress_chart_data,
    log_user_action
)

# set_page_config — ДОЛЖЕН быть самым первым вызовом Streamlit
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["initial_sidebar_state"],
)

# ==== РЕЗОЛВИМ API-ключи ПОСЛЕ set_page_config ====
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

# ==== MathJax ====
st.markdown("""
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
<script>
  MathJax.Hub.Config({
    tex2jax: { inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']], processEscapes: true }
  });
  MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
</script>
""", unsafe_allow_html=True)

# ==== CSS ====
st.markdown("""
<style>
.main-header { text-align:center; padding:2rem; background:linear-gradient(90deg,#667eea 0%,#764ba2 100%); border-radius:10px; color:#fff; margin-bottom:2rem; }
.progress-card { background:#fff; padding:1.2rem 1.5rem; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.08); margin:1rem 0; }
.task-card { background:#f8f9fa; padding:1.2rem 1.5rem; border-radius:10px; border-left:4px solid #007bff; margin:1rem 0; }
.success-animation { animation:pulse .5s ease-in-out; }
@keyframes pulse { 0%{transform:scale(1);} 50%{transform:scale(1.02);} 100%{transform:scale(1);} }
.difficulty-badge{ display:inline-block; padding:.25rem .6rem; border-radius:12px; font-size:.75rem; font-weight:600; text-transform:uppercase; margin-bottom:.5rem; }
.easy{ background:#d4edda; color:#155724; } .medium{ background:#fff3cd; color:#856404; } .hard{ background:#f8d7da; color:#721c24; }
.badge{ display:inline-block; padding:.2rem .5rem; border-radius:6px; font-size:.72rem; font-weight:600; }
.badge-green{ background:#d1fae5; color:#065f46; } .badge-gray{ background:#e5e7eb; color:#374151; }
.answer { padding:.5rem .75rem; border-radius:8px; margin:.25rem 0; }
.answer.correct { background:#e6ffed; border:1px solid #12b886; }
.answer.incorrect { background:#ffe8e8; border:1px solid #fa5252; }
</style>
""", unsafe_allow_html=True)


# ============== БЭКЕНД: DeepSeek клиент ==============
class EnhancedAITutor:
    def __init__(self):
        self.youtube_api_key = YOUTUBE_API_KEY
        self.deepseek_api_key = DEEPSEEK_API_KEY
        self.playlists = PLAYLISTS
        self.config = APP_CONFIG
        self.deepseek_config = DEEPSEEK_CONFIG
        self.ui_config = UI_CONFIG

    # ---- YouTube ----
    def get_playlist_videos(self, playlist_id: str):
        if not (isinstance(playlist_id, str) and (playlist_id.startswith("PL") or playlist_id.startswith("UU") or playlist_id.startswith("P"))):
            st.error(f"Неверный формат ID плейлиста: {playlist_id}")
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
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            videos = []
            for item in data.get("items", []):
                sn = item.get("snippet", {}) or {}
                thumbs = sn.get("thumbnails", {}) or {}
                thumb = thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}
                vid = sn.get("resourceId", {}).get("videoId")
                if not vid:
                    continue
                videos.append({
                    "title": sn.get("title", "Без названия"),
                    "video_id": vid,
                    "description": (sn.get("description") or "")[:280] + ("..." if len(sn.get("description") or "") > 280 else ""),
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
            st.error(f"Ошибка при загрузке видео: {e}")
            log_user_action("playlist_error", {"error": str(e), "playlist_id": playlist_id})
            return []

    # ---- DeepSeek: устойчивый вызов с ретраями ----
    def _call_deepseek_api(self, prompt: str, *, max_tokens: int):
        if not DEEPSEEK_ENABLED:
            return {"error": "deepseek_disabled"}

        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        headers = {"Authorization": f"Bearer {self.deepseek_api_key}", "Content-Type": "application/json"}
        data = {
            "model": self.deepseek_config.get("model", "deepseek-chat"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(self.deepseek_config.get("temperature", 0.5)),
            "max_tokens": int(max_tokens),
        }

        retry = Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["POST"]),
        )
        adapter = HTTPAdapter(max_retries=retry)
        sess = requests.Session()
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)

        try:
            resp = sess.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=(10, 60),  # connect 10s, read 60s
            )
            if resp.status_code == 402:
                return {"error": "402"}
            resp.raise_for_status()
            result = resp.json()
            content = result["choices"][0]["message"]["content"]

            # Попробуем JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"content": content}

        except requests.exceptions.Timeout:
            return {"error": "timeout"}
        except requests.exceptions.HTTPError as e:
            return {"error": f"http_{getattr(e.response, 'status_code', 'unknown')}"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        finally:
            sess.close()

    # ---- Теоретические вопросы: РОВНО N, без уровней ----
    def generate_theory_questions(self, topic: str, subject: str, grade: str, questions_count: int):
        prompt = f"""
Сгенерируй РОВНО {questions_count} теоретических вопросов по теме "{topic}" ({grade} класс, предмет "{subject}").

Требования к каждому вопросу:
- 4 варианта ответа строго в формате: "A) ...", "B) ...", "C) ...", "D) ..."
- Ровно один правильный вариант среди A/B/C/D
- Короткое объяснение, формулы в LaTeX: \\( ... \\) или \\[ ... \\]
- Формулировки короткие и по теме

Верни СТРОГО ВАЛИДНЫЙ JSON без комментариев и многоточий:
{{
  "questions": [
    {{
      "question": "Текст вопроса с LaTeX при необходимости: \\(...\\)",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct_answer": "A",
      "explanation": "Краткое объяснение: \\(...\\)"
    }}
  ]
}}
"""
        # для 10 вопросов этого хватает
        return self._call_deepseek_api(prompt, max_tokens=2000)

    # ---- Практика: easy/medium/hard ----
    def generate_practice_tasks(self, topic: str, subject: str, grade: str, user_performance: float | None):
        perf = ""
        if user_performance is not None:
            if user_performance < 60:
                perf = "Сделай акцент на простые задачи с очень понятными пояснениями."
            elif user_performance > 85:
                perf = "Добавь более сложные, нестандартные задачи."
        t_easy = APP_CONFIG["tasks_per_difficulty"]["easy"]
        t_med  = APP_CONFIG["tasks_per_difficulty"]["medium"]
        t_hard = APP_CONFIG["tasks_per_difficulty"]["hard"]

        prompt = f"""
Составь практические задачи по теме "{topic}" для {grade}-го класса по предмету "{subject}":
- {t_easy} лёгких,
- {t_med} средних,
- {t_hard} сложных.

{perf}

Для каждой задачи верни:
- "question": текст условия (с LaTeX при необходимости)
- "answer": правильный ответ (текст/число; для неравенств формат типа "x >= 2, x < 5")
- "solution": пошаговое объяснение (с LaTeX)
- "hint": короткая подсказка без LaTeX

Верни СТРОГО ВАЛИДНЫЙ JSON, без многоточий и комментариев:
{{
  "easy": [ {{ "question":"...", "answer":"...", "solution":"...", "hint":"..." }} ],
  "medium": [ {{ "question":"...", "answer":"...", "solution":"...", "hint":"..." }} ],
  "hard": [ {{ "question":"...", "answer":"...", "solution":"...", "hint":"..." }} ]
}}
"""
        # для полного блока практики — больше токенов
        return self._call_deepseek_api(prompt, max_tokens=2200)

    # ---- Короткая подсказка (малый max_tokens) ----
    def get_hint(self, question, user_answer, correct_answer):
        prompt = f"""
Студент решал задачу: "{question}"
Правильный ответ: "{correct_answer}"
Ответ студента: "{user_answer}"

Дай очень короткую подсказку (1-2 предложения), чтобы навести на верный путь, без LaTeX и без полного решения.
Если формат записан неверно (например, вместо ">=" написано "больше или равно"), обязательно укажи на это.
"""
        resp = self._call_deepseek_api(prompt, max_tokens=300)
        if isinstance(resp, dict) and "content" in resp and resp["content"].strip():
            return resp["content"].strip()
        return "Подумай о свойствах выражений и корректном формате записи ответа."


# ============== UI / ЛОГИКА ПРИЛОЖЕНИЯ ==============
def main():
    st.markdown('<div class="main-header"><h1>📚 AI Тьютор — персональное обучение</h1></div>', unsafe_allow_html=True)

    tutor = EnhancedAITutor()
    session = SessionManager()

    # ---- Sidebar: выбор курса ----
    st.sidebar.header("📖 Выбор курса")
    subjects = list(tutor.playlists.keys())
    subject = st.sidebar.selectbox("Предмет:", subjects, format_func=lambda x: f"{get_subject_emoji(x)} {x}")
    if subject:
        grades = list(tutor.playlists[subject].keys())
        grade = st.sidebar.selectbox("Класс:", grades)
        if grade:
            session.set_course(subject, grade)
            playlist_id = tutor.playlists[subject][grade]
            if st.sidebar.button("Начать обучение", type="primary"):
                with st.spinner("Загрузка видео из плейлиста..."):
                    videos = tutor.get_playlist_videos(playlist_id)
                    if videos:
                        session.start_course(videos)
                        st.success(f"Загружено {len(videos)} видео.")
                        st.rerun()
                    else:
                        st.error("Не удалось загрузить видео из плейлиста.")

    st.sidebar.markdown("---")
    st.sidebar.header("📊 Ваш прогресс")
    p = session.get_progress()
    st.sidebar.metric("Пройдено тем", len(p["completed_topics"]))
    chart = create_progress_chart_data(p)
    if chart:
        st.sidebar.plotly_chart(chart, use_container_width=True)

    # Небольшая диагностика DeepSeek
    with st.sidebar.expander("⚙️ Диагностика LLM"):
        if not DEEPSEEK_ENABLED:
            st.warning("DeepSeek: ключ не задан")
        else:
            if st.button("Проверить доступность"):
                ping = tutor._call_deepseek_api('{"ping": "ok"}', max_tokens=64)
                st.write(ping)

    # ---- Роутинг ----
    stage = session.get_stage()
    if stage == "video":
        display_video_content(tutor, session)
    elif stage == "theory_test":
        show_theory_test(tutor, session)
    elif stage == "practice":
        show_practice_stage(tutor, session)
    else:
        st.info("👆 Выберите предмет и класс в боковой панели, затем нажмите «Начать обучение».")


def display_video_content(tutor: EnhancedAITutor, session: SessionManager):
    videos = session.get_videos()
    if not videos:
        st.warning("Видео из плейлиста не загружены.")
        return
    current = videos[session.get_current_video_index()]
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader(f"📺 {current['title']}")
        st.video(f"https://www.youtube.com/watch?v={current['video_id']}")
        if current.get("description"):
            with st.expander("Описание урока"):
                st.write(current["description"])
    with c2:
        st.markdown('<div class="progress-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Текущий урок")
        st.info(f"Урок {session.get_current_video_index() + 1} из {len(videos)}")
        st.progress((session.get_current_video_index() + 1) / len(videos))
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Готов к тесту", type="primary"):
                session.set_stage("theory_test")
                log_user_action("start_theory_test", {"video": current["title"]})
                st.rerun()
        with b2:
            if st.button("Пересмотреть"):
                log_user_action("rewatch_video", {"video": current["title"]})
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
    st.caption(f"Тема: {current_video['title']}")

    # Генерация 1 раз
    if "theory_questions" not in st.session_state:
        with st.spinner("Генерация вопросов..."):
            qn = int(APP_CONFIG.get("theory_questions_count", 5))
            data = tutor.generate_theory_questions(
                topic=current_video["title"],
                subject=session.get_subject(),
                grade=session.get_grade(),
                questions_count=qn,
            )

        # Ошибки DeepSeek
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            if err == "timeout":
                st.error("DeepSeek не ответил вовремя (timeout). Попробуйте снова.")
            elif err in ("402", "deepseek_disabled"):
                st.error("Генерация недоступна (нет средств/ключа).")
            else:
                st.error(f"Не удалось сгенерировать вопросы: {err}")
            if st.button("Попробовать снова"):
                if "theory_questions" in st.session_state:
                    del st.session_state["theory_questions"]
                st.rerun()
            return

        if isinstance(data, dict) and "content" in data:
            try:
                data = json.loads(data["content"])
            except Exception:
                data = {"questions": []}

        questions = (data or {}).get("questions", [])
        # Ровно N
        qn = int(APP_CONFIG.get("theory_questions_count", 5))
        questions = questions[:qn]
        if len(questions) < qn:
            st.warning("Модель прислала меньше вопросов, чем нужно. Нажмите «Попробовать снова».")
            if st.button("Попробовать снова"):
                if "theory_questions" in st.session_state:
                    del st.session_state["theory_questions"]
                st.rerun()
            return

        # Нормализация опций (на случай, если пришли без «A) » и т.п.)
        for q in questions:
            opts = q.get("options", [])
            if len(opts) == 4:
                letters = ["A", "B", "C", "D"]
                q["options"] = [o if o.strip().lower().startswith(tuple([f"{x.lower()})" for x in letters]))
                                else f"{letters[i]}) {o}" for i, o in enumerate(opts)]
        st.session_state.theory_questions = questions
        st.session_state.theory_answers = {}

    # Рендер
    qs = st.session_state.theory_questions
    for i, q in enumerate(qs):
        st.markdown('<div class="task-card">', unsafe_allow_html=True)
        st.markdown(f"**Вопрос {i+1}:** {q.get('question','')}", unsafe_allow_html=True)
        options = q.get("options", ["A) —", "B) —", "C) —", "D) —"])
        selected = st.radio("Выберите ответ:", options, index=None, key=f"theory_q_{i}")
        if selected:
            st.session_state.theory_answers[i] = selected[0].upper()
        st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Вернуться к видео"):
            session.clear_theory_data()
            session.set_stage("video")
            st.rerun()
    with c2:
        if st.button("Проверить ответы", type="primary"):
            if len(st.session_state.theory_answers) == len(st.session_state.theory_questions):
                show_theory_results(tutor, session)
            else:
                st.error("Пожалуйста, ответьте на все вопросы.")


def show_theory_results(tutor: EnhancedAITutor, session: SessionManager):
    current_video = session.get_videos()[session.get_current_video_index()]
    topic_key = f"{session.get_subject()}_{session.get_grade()}_{current_video['title']}"
    st.markdown('<div class="progress-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Результаты тестирования")

    correct = 0
    total = len(st.session_state.theory_questions)
    for i, q in enumerate(st.session_state.theory_questions):
        user = st.session_state.theory_answers.get(i, "")
        corr = (q.get("correct_answer") or "").strip()[:1].upper()
        ok = compare_answers(user, corr)

        if ok:
            correct += 1
            st.markdown(f'<div class="answer correct">Вопрос {i+1}: Правильно!</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="answer incorrect">Вопрос {i+1}: Неправильно</div>', unsafe_allow_html=True)
            exp = q.get("explanation", "")
            if exp:
                st.markdown(f"**Объяснение:** {exp}", unsafe_allow_html=True)

    score = calculate_score(correct, total)
    st.metric("Ваш результат", f"{correct}/{total} ({score:.0f}%)")

    # Сохраняем
    session.save_theory_score(topic_key, score)

    # Порог
    pass_thr = APP_CONFIG.get("theory_pass_threshold", 60)
    if score < pass_thr:
        st.warning(f"Проходной порог {pass_thr}%. Рекомендуем пересмотреть видео и пройти тест снова.")
    else:
        st.success("Порог пройден! Можно переходить к практике.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Пересмотреть урок"):
            session.clear_theory_data()
            session.set_stage("video")
            st.rerun()
    with c2:
        # Кнопка доступна всегда, но можно заблокировать если хочешь строго после порога:
        if st.button("Начать практику", type="primary"):
            session.clear_theory_data()
            session.set_stage("practice")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def show_practice_stage(tutor: EnhancedAITutor, session: SessionManager):
    current_video = session.get_videos()[session.get_current_video_index()]
    st.subheader("💪 Практические задания")
    st.caption(f"Тема: {current_video['title']}")

    st.markdown("""
<div class="task-card">
  📝 <b>Совет:</b> Для сложных задач записывайте решение на черновике.<br>
  Для неравенств используйте символы: <code>&gt;=, &lt;=, &gt;, &lt;</code> и интервалы вида <code>[2, inf)</code>.<br>
  Для нескольких условий — <code>and</code> или запятые.
</div>
""", unsafe_allow_html=True)

    # Генерация 1 раз
    if "practice_tasks" not in st.session_state:
        with st.spinner("Генерация заданий..."):
            theory_score = session.get_theory_score(current_video["title"])
            data = tutor.generate_practice_tasks(
                topic=current_video["title"],
                subject=session.get_subject(),
                grade=session.get_grade(),
                user_performance=theory_score
            )
        # Ошибки
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            if err == "timeout":
                st.error("DeepSeek не ответил вовремя (timeout). Попробуйте снова.")
            elif err in ("402", "deepseek_disabled"):
                st.error("Генерация недоступна (нет средств/ключа).")
            else:
                st.error(f"Не удалось сгенерировать задания: {err}")
            if st.button("Попробовать снова"):
                if "practice_tasks" in st.session_state:
                    del st.session_state["practice_tasks"]
                st.rerun()
            return

        if isinstance(data, dict) and "content" in data:
            try:
                data = json.loads(data["content"])
            except Exception:
                data = {"easy": [], "medium": [], "hard": []}

        st.session_state.practice_tasks = {
            "easy":   data.get("easy", []),
            "medium": data.get("medium", []),
            "hard":   data.get("hard", []),
        }
        st.session_state.task_attempts = {}
        st.session_state.completed_tasks = []
        st.session_state.current_task_type = "easy"
        st.session_state.current_task_index = 0

    # Если задач нет
    if not any(len(st.session_state.practice_tasks.get(t, [])) for t in ["easy", "medium", "hard"]):
        st.error("Нет заданий. Попробуйте снова.")
        return

    show_current_task(tutor, session)


def show_current_task(tutor: EnhancedAITutor, session: SessionManager):
    task_types = ["easy", "medium", "hard"]
    cur_type = st.session_state.current_task_type
    cur_idx = st.session_state.current_task_index
    tasks = st.session_state.practice_tasks.get(cur_type, [])

    # Переходы
    if cur_idx >= len(tasks):
        type_i = task_types.index(cur_type)
        if type_i < len(task_types) - 1:
            st.session_state.current_task_type = task_types[type_i + 1]
            st.session_state.current_task_index = 0
            st.rerun()
        else:
            show_practice_completion(tutor, session)
            return

    task = tasks[cur_idx]
    task_key = f"{cur_type}_{cur_idx}"

    total_tasks = sum(len(st.session_state.practice_tasks.get(t, [])) for t in task_types)
    completed = len(st.session_state.completed_tasks)

    c1, c2 = st.columns([3, 1])
    with c2:
        st.markdown('<div class="progress-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Прогресс")
        st.progress(completed / total_tasks if total_tasks else 0)
        st.metric("Выполнено", f"{completed}/{total_tasks}")
        st.markdown(f'<span class="difficulty-badge {cur_type}">{UI_CONFIG["task_type_names"][cur_type]}</span>', unsafe_allow_html=True)
        st.markdown(f"**Задание:** {cur_idx + 1} из {len(tasks)}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c1:
        st.markdown(f'<div class="task-card"><span class="difficulty-badge {cur_type}">{UI_CONFIG["task_type_names"][cur_type]}</span>', unsafe_allow_html=True)
        st.markdown(f"### Задание {cur_idx + 1}")
        st.markdown(task.get("question", ""), unsafe_allow_html=True)

        ans = st.text_input("Ваш ответ:", key=f"ans_{task_key}")
        attempts = st.session_state.task_attempts.get(task_key, 0)
        max_attempts = APP_CONFIG["max_attempts_per_task"]

        if attempts < max_attempts:
            b1, b2 = st.columns(2)
            with b1:
                if st.button("Проверить ответ", type="primary"):
                    if ans.strip():
                        check_answer(tutor, session, task, ans, task_key)
                    else:
                        st.error("Введите ответ.")
            with b2:
                if st.button("Пропустить"):
                    log_user_action("skip_task", {"task_key": task_key})
                    move_to_next_task()
        else:
            st.error(f"Попытки исчерпаны ({max_attempts}).")
            st.markdown(f"**Правильный ответ:** {task.get('answer','')}", unsafe_allow_html=True)
            st.markdown(f"**Решение:** {task.get('solution','')}", unsafe_allow_html=True)
            if st.button("Следующее задание"):
                move_to_next_task()

        # Подсказки, если уже есть
        if task_key in st.session_state and "hints" in st.session_state[task_key]:
            st.markdown("### 💡 Подсказки")
            for h in st.session_state[task_key]["hints"]:
                st.info(h)

        st.markdown("</div>", unsafe_allow_html=True)


def check_answer(tutor: EnhancedAITutor, session: SessionManager, task: dict, user_answer: str, task_key: str):
    st.session_state.task_attempts[task_key] = st.session_state.task_attempts.get(task_key, 0) + 1
    attempts = st.session_state.task_attempts[task_key]
    max_attempts = APP_CONFIG["max_attempts_per_task"]

    is_ok = compare_answers((user_answer or "").strip().lower(), (task.get("answer") or "").strip().lower())

    if is_ok:
        st.markdown('<div class="answer correct">Правильно! Отличная работа.</div>', unsafe_allow_html=True)
        if task_key not in st.session_state.completed_tasks:
            st.session_state.completed_tasks.append(task_key)
        log_user_action("correct_answer", {"task_key": task_key, "attempts": attempts})
        if st.button("Следующее задание"):
            move_to_next_task()
    else:
        if attempts < max_attempts:
            st.markdown('<div class="answer incorrect">Неправильно.</div>', unsafe_allow_html=True)
            # Подсказка (DeepSeek — малый max_tokens)
            hint = "Подумай про свойства выражений и корректный формат ответа."
            if DEEPSEEK_ENABLED:
                try:
                    hint = tutor.get_hint(task.get("question", ""), user_answer, task.get("answer", ""))
                except Exception:
                    pass
            if task_key not in st.session_state:
                st.session_state[task_key] = {"hints": []}
            st.session_state[task_key]["hints"].append(hint)
            st.info(hint)
            log_user_action("incorrect_answer", {"task_key": task_key, "attempts": attempts})
        else:
            st.markdown('<div class="answer incorrect">Все попытки исчерпаны.</div>', unsafe_allow_html=True)
            st.markdown(f"**Правильный ответ:** {task.get('answer','')}", unsafe_allow_html=True)
            st.markdown(f"**Решение:** {task.get('solution','')}", unsafe_allow_html=True)
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
    current = videos[session.get_current_video_index()]
    topic_key = f"{session.get_subject()}_{session.get_grade()}_{current['title']}"

    st.markdown('<div class="progress-card">', unsafe_allow_html=True)
    st.subheader("Практика завершена!")
    total = sum(len(st.session_state.practice_tasks.get(t, [])) for t in ["easy", "medium", "hard"])
    done = len(st.session_state.completed_tasks)
    score = calculate_score(done, total) if total else 0
    st.success(f"Выполнено {done} из {total} заданий ({score:.0f}%).")

    session.save_practice_score(topic_key, done, total)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Изучить новую тему"):
            if session.next_video():
                session.set_stage("video")
                session.clear_practice_data()
                st.rerun()
            else:
                st.info("Все темы курса пройдены!")
    with c2:
        if st.button("Вернуться к выбору курса"):
            session.set_stage("selection")
            session.clear_practice_data()
            st.rerun()

    st.markdown(generate_progress_report(session.get_progress(), topic_key), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

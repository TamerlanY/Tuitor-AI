# app.py
import os
import json
import re
from datetime import datetime

import requests
import streamlit as st
import plotly.express as px
import pandas as pd

from config import (
    PLAYLISTS,
    APP_CONFIG,
    DEEPSEEK_CONFIG,
    UI_CONFIG,
    # опционально — если есть в config, не обязательно использовать здесь
    # SUPABASE_URL, SUPABASE_ANON_KEY
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

# -----------------------
# set_page_config — первым!
# -----------------------
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["initial_sidebar_state"],
)

# ==== РЕЗОЛВИМ КЛЮЧИ ====
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
try:
    if (not YOUTUBE_API_KEY) and hasattr(st, "secrets") and "YOUTUBE_API_KEY" in st.secrets:
        YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    if (not DEEPSEEK_API_KEY) and hasattr(st, "secrets") and "DEEPSEEK_API_KEY" in st.secrets:
        DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    pass

# YouTube обязателен для загрузки плейлистов
if not YOUTUBE_API_KEY:
    st.error("Не задан YOUTUBE_API_KEY. Укажи его в .env или в Secrets.")
    st.stop()

DEEPSEEK_ENABLED = bool(DEEPSEEK_API_KEY)

# MathJax (для формул в markdown)
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
.right { text-align: right; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Класс-обёртка над API/LLM
# =========================
class EnhancedAITutor:
    def __init__(self):
        self.youtube_api_key = YOUTUBE_API_KEY
        self.deepseek_api_key = DEEPSEEK_API_KEY
        self.playlists = PLAYLISTS
        self.config = APP_CONFIG
        self.deepseek_config = DEEPSEEK_CONFIG
        self.ui_config = UI_CONFIG

    # --------- YouTube ----------
    def get_playlist_videos(self, playlist_id: str):
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
            r = requests.get(url, params=params, timeout=12)
            r.raise_for_status()
            data = r.json()
            videos = []
            for item in data.get("items", []):
                sn = item.get("snippet", {}) or {}
                thumbs = sn.get("thumbnails", {}) or {}
                thumb = thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}
                title = sn.get("title") or "Без названия"
                video_id = (sn.get("resourceId") or {}).get("videoId")
                if not video_id:
                    continue
                desc = sn.get("description") or ""
                short_desc = (desc[:200] + "...") if len(desc) > 200 else desc
                videos.append(
                    {
                        "title": title,
                        "video_id": video_id,
                        "description": short_desc,
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

    # --------- DeepSeek ----------
    def _call_deepseek_api(self, prompt: str, *, max_tokens: int | None = None):
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
                    timeout=self.deepseek_config["timeout"],
                )
                if resp.status_code == 402:
                    # баланс/подписка
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

    # === Теория: всегда РОВНО N вопросов (батчами + валидация + заглушки) ===
    def generate_theory_questions(self, topic: str, subject: str, grade: str, questions_count: int):
        def make_prompt(n):
            return f"""
Сгенерируй РОВНО {n} тестовых вопросов по теме "{topic}" ({grade} класс, предмет "{subject}").
Требования:
- Каждый вопрос с 4 вариантами ответа строго в формате: "A) …", "B) …", "C) …", "D) …"
- Ровно один правильный вариант, укажи букву в поле "correct_answer" (A/B/C/D)
- Короткое объяснение (можно с LaTeX \\( ... \\))
- Содержательно строго по теме и по уровню класса

Верни СТРОГО ВАЛИДНЫЙ JSON (без комментариев/многоточий):
{{
  "questions": [
    {{
      "question": "Текст вопроса",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct_answer": "A",
      "explanation": "Краткое объяснение"
    }}
  ]
}}
"""

        def validate_and_normalize(items):
            ok = []
            letters = ["A", "B", "C", "D"]
            for q in items:
                try:
                    question = str(q.get("question", "")).strip()
                    options = q.get("options") or []
                    if len(options) != 4:
                        continue
                    fixed_opts = []
                    for i, opt in enumerate(options):
                        opt = str(opt or "").strip()
                        # принудительно префикс A)/B)/C)/D)
                        if not opt.lower().startswith(f"{letters[i].lower()})"):
                            opt = f"{letters[i]}) {opt}"
                        fixed_opts.append(opt)
                    ca = str(q.get("correct_answer", "")).strip()[:1].upper()
                    if ca not in letters:
                        continue
                    exp = str(q.get("explanation", "")).strip()
                    ok.append(
                        {
                            "question": question if question else "Вопрос недоступен.",
                            "options": fixed_opts,
                            "correct_answer": ca,
                            "explanation": exp if exp else "Объяснение недоступно.",
                        }
                    )
                except Exception:
                    continue
            return ok

        need = int(questions_count)
        acc: list[dict] = []
        batch_size = 5
        max_retries_per_batch = 2

        while len(acc) < need:
            to_get = min(batch_size, need - len(acc))
            tries = 0
            got_batch: list[dict] = []
            while tries <= max_retries_per_batch and len(got_batch) < to_get:
                resp = self._call_deepseek_api(make_prompt(to_get), max_tokens=1200)
                if isinstance(resp, dict) and resp.get("error"):
                    # не удалось — прервём этот батч, дальше дозаполним заглушками
                    break
                if isinstance(resp, dict) and "content" in resp:
                    try:
                        resp = json.loads(resp["content"])
                    except Exception:
                        resp = {"questions": []}
                items = (resp or {}).get("questions", []) or []
                got_batch = validate_and_normalize(items)
                tries += 1

            acc.extend(got_batch)
            if not got_batch:
                break  # чтобы не зациклиться — добьём заглушками

        # заполняем placeholders до нужного количества
        while len(acc) < need:
            idx = len(acc) + 1
            acc.append(
                {
                    "question": f"Вопрос недоступен (заглушка) по теме «{topic}».",
                    "options": ["A) —", "B) —", "C) —", "D) —"],
                    "correct_answer": "A",
                    "explanation": "Объяснение недоступно.",
                }
            )

        return {"questions": acc[:need]}

    # === Практика: генерируем набор задач по уровням ===
    def generate_practice_tasks_enhanced(self, topic, subject, grade, user_performance=None):
        perf = ""
        if user_performance is not None:
            if user_performance < 60:
                perf = "Сделай акцент на более простые задания с подробными объяснениями."
            elif user_performance > 85:
                perf = "Добавь более сложные и нестандартные задачи."

        cnt_easy = self.config["tasks_per_difficulty"]["easy"]
        cnt_medium = self.config["tasks_per_difficulty"]["medium"]
        cnt_hard = self.config["tasks_per_difficulty"]["hard"]

        prompt = f"""
Составь практические задания по теме "{topic}" для {grade}-го класса по предмету "{subject}":
- {cnt_easy} лёгкие,
- {cnt_medium} средние,
- {cnt_hard} сложные.

{perf}

Для каждой задачи:
- Чёткое условие (LaTeX \\( ... \\) допускается)
- Поле "answer": правильный ответ (строка/число, БЕЗ LaTeX, например "x >= 2, x < 3")
- Поле "solution": краткое пошаговое объяснение (можно с LaTeX)
- Поле "hint": короткая подсказка без LaTeX

Верни СТРОГО ВАЛИДНЫЙ JSON (без многоточий/комментариев):
{{
  "easy": [{{"question":"...","answer":"...","solution":"...","hint":"..."}}, ...],
  "medium": [...],
  "hard": [...]
}}
"""
        return self._call_deepseek_api(prompt, max_tokens=1800)


# ==================
# Приложение Streamlit
# ==================
def main():
    st.markdown(
        '<div class="main-header"><h1>📚 AI Тьютор — персональное обучение</h1></div>',
        unsafe_allow_html=True,
    )

    tutor = EnhancedAITutor()
    # опционально — пользовательский идентификатор (если в utils есть облачное хранение)
    with st.sidebar:
        st.markdown("### 👤 Пользователь")
        user_id = st.text_input("Идентификатор (email/ник)", placeholder="например, sister_01")

    session = SessionManager(user_id=user_id if user_id else None)

    # ========== Боковая панель: выбор курса ==========
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
                    with st.spinner("Загрузка видео..."):
                        videos = tutor.get_playlist_videos(playlist_id)
                        if videos:
                            session.start_course(videos)
                            st.success(f"Загружено видео: {len(videos)}")
                            st.rerun()
                        else:
                            st.error("Не удалось загрузить видео из плейлиста.")

        st.markdown("---")
        st.header("📊 Ваш прогресс")
        progress_data = session.get_progress()
        st.metric("Пройдено тем", len(progress_data.get("completed_topics", [])))
        chart = create_progress_chart_data(progress_data)
        if chart:
            st.plotly_chart(chart, use_container_width=True)

    # ========== Роутинг основных экранов ==========
    stage = session.get_stage()
    if stage == "video":
        display_video_content(tutor, session)
    elif stage == "theory_test":
        show_theory_test(tutor, session)
    elif stage == "practice":
        show_practice_stage(tutor, session)
    else:
        st.info("👆 Выберите предмет и класс слева и нажмите «Начать обучение».")


def display_video_content(tutor: EnhancedAITutor, session: SessionManager):
    videos = session.get_videos()
    if not videos:
        st.warning("Видео из плейлиста не загружены. Нажмите «Начать обучение» слева.")
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
                log_user_action("previous_video", {"idx": session.get_current_video_index()})
                st.rerun()

        if session.get_current_video_index() < len(videos) - 1:
            if st.button("Следующий урок →"):
                session.next_video()
                log_user_action("next_video", {"idx": session.get_current_video_index()})
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


def show_theory_test(tutor: EnhancedAITutor, session: SessionManager):
    current_video = session.get_videos()[session.get_current_video_index()]
    st.header("📝 Тест по теории")
    st.info(f"Тема: {current_video['title']}")

    if "theory_questions" not in st.session_state:
        with st.spinner("Генерация вопросов..."):
            qn = int(APP_CONFIG.get("theory_questions_count", 5))
            data = tutor.generate_theory_questions(
                topic=current_video["title"],
                subject=session.get_subject(),
                grade=session.get_grade(),
                questions_count=qn,
            )
            # если DeepSeek упал — всё равно вернётся список с заглушками
            if isinstance(data, dict) and "content" in data:
                try:
                    data = json.loads(data["content"])
                except Exception:
                    data = {"questions": []}
            questions = (data or {}).get("questions", [])
            st.session_state.theory_questions = questions[:qn]
            st.session_state.theory_answers = {}

    if not st.session_state.theory_questions:
        st.error("Не удалось сгенерировать вопросы. Попробуйте снова.")
        return

    for i, q in enumerate(st.session_state.theory_questions):
        st.markdown('<div class="task-card">', unsafe_allow_html=True)
        st.markdown(f"**Вопрос {i+1}:** {q.get('question','')}")
        options = q.get("options") or ["A) —", "B) —", "C) —", "D) —"]
        answer_key = f"theory_q_{i}"
        selected = st.radio("Выберите ответ:", options, key=answer_key, index=None)
        if selected:
            # берём букву
            st.session_state.theory_answers[i] = (selected or "A)")[0]
        st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Вернуться к видео"):
            session.clear_theory_data()
            session.set_stage("video")
            log_user_action("return_to_video", {"video": current_video["title"]})
            st.rerun()
    with c2:
        if st.button("Проверить ответы", type="primary"):
            # убедимся, что на всё ответили — но не стопорим, просто предупредим
            if len(st.session_state.theory_answers) < len(st.session_state.theory_questions):
                st.warning("Вы ответили не на все вопросы — считаю только отвеченные.")
            show_theory_results(tutor, session)


def show_theory_results(tutor: EnhancedAITutor, session: SessionManager):
    current_video = session.get_videos()[session.get_current_video_index()]
    topic_key = f"{session.get_subject()}_{session.get_grade()}_{current_video['title']}"

    st.markdown('<div class="progress-card">', unsafe_allow_html=True)
    st.subheader("📊 Результаты тестирования")

    correct_count = 0
    total_questions = len(st.session_state.theory_questions)

    for i, q in enumerate(st.session_state.theory_questions):
        user_answer = st.session_state.theory_answers.get(i)
        correct_answer = (q.get("correct_answer") or "A").strip()[:1].upper()
        is_ok = compare_answers(user_answer or "", correct_answer or "A")

        if is_ok:
            correct_count += 1
            st.markdown('<div class="success-animation">', unsafe_allow_html=True)
            st.success(f"Вопрос {i+1}: Правильно! ✅")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error(f"Вопрос {i+1}: Неправильно ❌")
            expl = q.get("explanation", "")
            if expl:
                st.markdown(f"**Объяснение:** {expl}")

    score = calculate_score(correct_count, total_questions)
    st.metric("Ваш результат", f"{correct_count}/{total_questions} ({score:.0f}%)")
    session.save_theory_score(topic_key, score)

    if score < tutor.config["theory_pass_threshold"]:
        st.warning("Рекомендуем пересмотреть видео для лучшего понимания темы.")

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

    st.markdown('</div>', unsafe_allow_html=True)


def show_practice_stage(tutor: EnhancedAITutor, session: SessionManager):
    current_video = session.get_videos()[session.get_current_video_index()]
    st.header("💪 Практические задания")
    st.info(f"Тема: {current_video['title']}")

    st.markdown(
        """
<div class="notebook-note">
📝 <b>Совет:</b> Для сложных задач используйте тетрадь и вводите конечный ответ.<br/>
Для неравенств: <code>x >= 2</code>, интервалы: <code>[2, inf)</code>, несколько условий — <code>and</code> или запятая.
</div>
""",
        unsafe_allow_html=True,
    )

    if "practice_tasks" not in st.session_state:
        with st.spinner("Генерация заданий..."):
            theory_score = session.get_theory_score(current_video["title"])  # utils сам соберёт topic_key
            data = tutor.generate_practice_tasks_enhanced(
                topic=current_video["title"],
                subject=session.get_subject(),
                grade=session.get_grade(),
                user_performance=theory_score,
            )
            if isinstance(data, dict) and "content" in data:
                try:
                    data = json.loads(data["content"])
                except Exception:
                    data = {"easy": [], "medium": [], "hard": []}
            if isinstance(data, dict) and data.get("error") in ("402", "deepseek_disabled", "timeout"):
                st.error("Не удалось сгенерировать задания (LLM недоступен).")
                st.session_state.practice_tasks = {"easy": [], "medium": [], "hard": []}
            else:
                st.session_state.practice_tasks = data

            st.session_state.task_attempts = {}
            st.session_state.completed_tasks = []
            st.session_state.current_task_type = "easy"
            st.session_state.current_task_index = 0

    if any(len(st.session_state.practice_tasks.get(t, [])) for t in ["easy", "medium", "hard"]):
        show_current_task(tutor, session)
    else:
        st.error("Нет заданий. Попробуйте позже.")


def show_current_task(tutor: EnhancedAITutor, session: SessionManager):
    task_types = ["easy", "medium", "hard"]
    cur_type = st.session_state.current_task_type
    cur_idx = st.session_state.current_task_index

    tasks_of_type = st.session_state.practice_tasks.get(cur_type, [])
    if cur_idx >= len(tasks_of_type):
        ti = task_types.index(cur_type)
        if ti < len(task_types) - 1:
            st.session_state.current_task_type = task_types[ti + 1]
            st.session_state.current_task_index = 0
            st.rerun()
        else:
            show_practice_completion(tutor, session)
            return

    task = tasks_of_type[cur_idx]
    task_key = f"{cur_type}_{cur_idx}"

    total = sum(len(st.session_state.practice_tasks.get(t, [])) for t in task_types)
    done = len(st.session_state.completed_tasks)

    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown('<div class="progress-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Прогресс")
        st.progress(done / total if total else 0)
        st.metric("Выполнено", f"{done}/{total}")
        st.markdown(
            f'<span class="difficulty-badge {cur_type}">{UI_CONFIG["task_type_names"][cur_type]}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Задание:** {cur_idx + 1} из {len(tasks_of_type)}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col1:
        st.markdown(
            f'<div class="task-card"><span class="difficulty-badge {cur_type}">{UI_CONFIG["task_type_names"][cur_type]}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"### Задание {cur_idx + 1}")
        st.markdown(task.get("question", ""))

        user_answer = st.text_input("Ваш ответ:", key=f"ans_{task_key}")
        attempts = st.session_state.task_attempts.get(task_key, 0)
        max_att = APP_CONFIG["max_attempts_per_task"]

        if attempts < max_att:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Проверить ответ", type="primary"):
                    if (user_answer or "").strip():
                        check_answer(tutor, session, task, user_answer, task_key)
                    else:
                        st.error("Введите ответ.")
            with c2:
                if st.button("Пропустить"):
                    log_user_action("skip_task", {"task_key": task_key})
                    move_to_next_task()
        else:
            st.error(f"Исчерпаны все попытки ({max_att}).")
            st.markdown(f"**Правильный ответ:** {task.get('answer','')}")
            st.markdown(f"**Решение:** {task.get('solution','')}")
            if st.button("Следующее задание"):
                move_to_next_task()

        # подсказки
        hints_bucket = st.session_state.get(task_key, {}).get("hints", [])
        if hints_bucket:
            st.markdown("### 💡 Подсказки")
            for h in hints_bucket:
                st.info(h)

        st.markdown('</div>', unsafe_allow_html=True)


def check_answer(tutor: EnhancedAITutor, session: SessionManager, task: dict, user_answer: str, task_key: str):
    st.session_state.task_attempts[task_key] = st.session_state.task_attempts.get(task_key, 0) + 1
    attempts = st.session_state.task_attempts[task_key]
    max_att = APP_CONFIG["max_attempts_per_task"]

    is_ok = compare_answers((user_answer or "").strip().lower(), (task.get("answer") or "").strip().lower())

    if is_ok:
        st.markdown('<div class="success-animation">', unsafe_allow_html=True)
        st.success("Правильно! Отличная работа.")
        st.markdown('</div>', unsafe_allow_html=True)
        if task_key not in st.session_state.completed_tasks:
            st.session_state.completed_tasks.append(task_key)
        log_user_action("correct_answer", {"task_key": task_key, "attempts": attempts})
        if st.button("Следующее задание"):
            move_to_next_task()
    else:
        if attempts < max_att:
            st.error(f"Неправильно. Попытка {attempts} из {max_att}.")
            # Короткая подсказка: либо локальная, либо через LLM, если доступен
            hint = "Подумайте о свойствах выражения/формулы и проверьте формат ответа."
            if DEEPSEEK_ENABLED:
                try:
                    hint_resp = tutor._call_deepseek_api(
                        f"""
Студент решал задачу: "{task.get('question','')}"
Правильный ответ: "{task.get('answer','')}"
Ответ студента: "{user_answer}"
Дай 1-2 предложения подсказки (без LaTeX), где ошибка и куда смотреть, не раскрывая полный ответ.
""",
                        max_tokens=200,
                    )
                    if isinstance(hint_resp, dict) and "content" in hint_resp:
                        hint = str(hint_resp["content"]).strip()
                except Exception:
                    pass

            bucket = st.session_state.get(task_key, {"hints": []})
            bucket["hints"].append(hint)
            st.session_state[task_key] = bucket
            st.info(hint)

            log_user_action("incorrect_answer", {"task_key": task_key, "attempts": attempts})
        else:
            st.error("Все попытки исчерпаны.")
            st.markdown(f"**Правильный ответ:** {task.get('answer','')}")
            st.markdown(f"**Решение:** {task.get('solution','')}")
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
    st.header("Практика завершена!")

    total = sum(len(st.session_state.practice_tasks.get(t, [])) for t in ["easy", "medium", "hard"])
    done = len(st.session_state.completed_tasks)
    score = calculate_score(done, total) if total else 0

    st.success(f"Выполнено {done} из {total} заданий ({score:.0f}%)")
    session.save_practice_score(topic_key, done, total)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Изучить новую тему"):
            if session.next_video():
                session.set_stage("video")
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

    # Отчёт
    st.markdown(generate_progress_report(session.get_progress(), topic_key), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

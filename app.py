import os
import json
import requests
import streamlit as st

from config import (
    PLAYLISTS, APP_CONFIG, DEEPSEEK_CONFIG, UI_CONFIG,
    # Эти два могут отсутствовать в config.py — не критично
    # просто оставим импорт, а если их нет — обойдёмся локальным режимом
    # SUPABASE_URL, SUPABASE_ANON_KEY
)

from utils import (
    compare_answers, calculate_score, generate_progress_report,
    get_subject_emoji, SessionManager, create_progress_chart_data,
    log_user_action, diagnose_mistake
)

# ---------- set_page_config ДОЛЖЕН быть первым вызовом ----------
st.set_page_config(
    page_title=UI_CONFIG.get("page_title", "AI Тьютор"),
    page_icon=UI_CONFIG.get("page_icon", "📚"),
    layout=UI_CONFIG.get("layout", "wide"),
    initial_sidebar_state=UI_CONFIG.get("initial_sidebar_state", "expanded"),
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

DEEPSEEK_ENABLED = bool(DEEPSEEK_API_KEY)

# ---------- MathJax ----------
st.markdown("""
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
<script>
    MathJax.Hub.Config({
        tex2jax: { inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']], processEscapes: true }
    });
    MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
</script>
""", unsafe_allow_html=True)

# ---------- CSS ----------
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

/* Подсветка вариантов ответа в теории */
.choice { padding: .35rem .6rem; border-radius: 6px; margin:.15rem 0; display:inline-block; }
.choice-correct { background:#d1fae5; color:#065f46; }   /* зеленый */
.choice-wrong   { background:#fee2e2; color:#991b1b; }   /* красный */
</style>
""", unsafe_allow_html=True)


# ======================== МОДУЛЬ LLM =========================
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
        if not (isinstance(playlist_id, str) and playlist_id.startswith(("PL", "UU", "VL"))):
            st.error(f"Неверный формат ID плейлиста: {playlist_id}.")
            log_user_action("invalid_playlist_id", {"playlist_id": playlist_id})
            return []

        url = "https://www.googleapis.com/youtube/v3/playlistItems"
        params = {
            "part": "snippet,contentDetails",
            "playlistId": playlist_id,
            "maxResults": int(self.config.get("youtube_max_results", 50)),
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
                video_id = (sn.get("resourceId") or {}).get("videoId")
                if not video_id:
                    continue
                desc = sn.get("description") or ""
                videos.append({
                    "title": sn.get("title", "Без названия"),
                    "video_id": video_id,
                    "description": (desc[:200] + "…") if len(desc) > 200 else desc,
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

    # ---- DeepSeek Base Call ----
    def _call_deepseek_api(self, prompt: str):
        if not DEEPSEEK_ENABLED:
            return {"error": "deepseek_disabled"}
        headers = {"Authorization": f"Bearer {self.deepseek_api_key}", "Content-Type": "application/json"}
        data = {
            "model": self.deepseek_config.get("model", "deepseek-chat"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(self.deepseek_config.get("temperature", 0.7)),
            "max_tokens": int(self.deepseek_config.get("max_tokens", 4000)),
        }
        for attempt in range(int(self.deepseek_config.get("retry_attempts", 3))):
            try:
                resp = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=int(self.deepseek_config.get("timeout", 30)),
                )
                if resp.status_code == 402:
                    st.warning("DeepSeek вернул 402 (недостаточно средств). Генерация временно отключена.")
                    return {"error": "402"}
                resp.raise_for_status()
                result = resp.json()
                content = result["choices"][0]["message"]["content"]
                # Попробуем JSON, иначе вернём текст
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"content": content}
            except requests.exceptions.Timeout:
                if attempt == int(self.deepseek_config.get("retry_attempts", 3)) - 1:
                    st.error("Превышено время ожидания ответа от DeepSeek API")
                    return {"error": "timeout"}
            except requests.exceptions.HTTPError as e:
                if attempt == int(self.deepseek_config.get("retry_attempts", 3)) - 1:
                    st.error(f"Ошибка HTTP DeepSeek API: {e.response.status_code}")
                    return {"error": str(e)}
            except Exception as e:
                if attempt == int(self.deepseek_config.get("retry_attempts", 3)) - 1:
                    st.error(f"Ошибка API DeepSeek: {str(e)}")
                    return {"error": str(e)}

    # ---- Теория: строго N вопросов, без easy/medium/hard ----
    def generate_theory_questions(self, topic: str, subject: str, grade: str, questions_count: int):
        prompt = f"""
Создай ровно {questions_count} теоретических вопросов по теме "{topic}" для {grade}-го класса по предмету "{subject}".

Требования:
- Каждый вопрос с ровно 4 вариантами ответа формата "A) ...", "B) ...", "C) ...", "D) ..."
- Ровно один правильный вариант (A/B/C/D)
- К каждому вопросу — короткое и ясное объяснение
- Формулы только в LaTeX: \\( ... \\) для inline, \\[ ... \\] для блочных
- Строго ВАЛИДНЫЙ JSON. Никаких многоточий/комментариев.

Верни строго такой JSON (заполни содержимым):
{{
  "questions": [
    {{
      "question": "Текст вопроса с LaTeX при необходимости: \\(...\\)",
      "options": ["A) вариант", "B) вариант", "C) вариант", "D) вариант"],
      "correct_answer": "A",
      "explanation": "Краткое объяснение с LaTeX при необходимости: \\(...\\)"
    }}
  ]
}}
"""
        return self._call_deepseek_api(prompt)

    # ---- Практика ----
    def generate_practice_tasks(self, topic: str, subject: str, grade: str, user_performance: float | None):
        perf = ""
        if user_performance is not None:
            if user_performance < 60:
                perf = "Сделай акцент на более простые задачи и добавь детальные подсказки."
            elif user_performance > 85:
                perf = "Добавь больше нестандартных/повышенных по сложности задач."
        prompt = f"""
Составь практические задания по теме "{topic}" для {grade}-го класса по предмету "{subject}":
- {self.config["tasks_per_difficulty"]["easy"]} задачи уровня easy
- {self.config["tasks_per_difficulty"]["medium"]} задачи уровня medium
- {self.config["tasks_per_difficulty"]["hard"]} задачи уровня hard

{perf}

Для каждой задачи верни:
- "question": формулировка с LaTeX при необходимости
- "answer": точный правильный ответ (текст/число/интервалы) БЕЗ LaTeX, например "x >= 2, x < 3"
- "solution": краткое пошаговое объяснение c LaTeX
- "hint": короткая подсказка (без LaTeX)

Верни строго валидный JSON без многоточий:
{{
  "easy":   [{{"question":"...","answer":"...","solution":"...","hint":"..."}}],
  "medium": [{{"question":"...","answer":"...","solution":"...","hint":"..."}}],
  "hard":   [{{"question":"...","answer":"...","solution":"...","hint":"..."}}]
}}
"""
        return self._call_deepseek_api(prompt)


# ======================== ВСПОМОГАТЕЛЬНЫЕ ЭКРАНЫ =========================
def main():
    st.markdown('<div class="main-header"><h1>📚 AI Тьютор — Персональное обучение</h1></div>', unsafe_allow_html=True)

    tutor = EnhancedAITutor()

    # ---- Идентификатор пользователя (для облака, если подключишь БД) ----
    st.sidebar.markdown("### 👤 Пользователь")
    user_id = st.sidebar.text_input("Идентификатор (email/ник) для облачного прогресса", value="")

    # Session manager (локальное хранение; user_id просто сохраняем для будущего)
    session = SessionManager(user_id=user_id if user_id else None)

    # ---- Боковая панель: выбор курса ----
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

    # ---- Прогресс ----
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Ваш прогресс")
    progress_data = session.get_progress()
    st.sidebar.metric("Пройдено тем", len(progress_data.get("completed_topics", [])))
    chart_fig = create_progress_chart_data(progress_data)
    if chart_fig:
        st.sidebar.plotly_chart(chart_fig, use_container_width=True)

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
        st.warning("Видео из плейлиста не загружены. Попробуйте перезагрузить страницу.")
        return
    current_video = videos[session.get_current_video_index()]

    col1, col2 = st.columns([2, 1], vertical_alignment="top")
    with col1:
        st.header(f"📺 {current_video['title']}")
        st.video(f"https://www.youtube.com/watch?v={current_video['video_id']}")
        if current_video.get('description'):
            with st.expander("Описание урока"):
                st.write(current_video['description'])

    with col2:
        st.markdown('<div class="progress-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Текущий урок")
        st.info(f"Урок {session.get_current_video_index() + 1} из {len(videos)}")
        st.progress((session.get_current_video_index() + 1) / max(1, len(videos)))

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


def show_theory_test(tutor: EnhancedAITutor, session: SessionManager):
    current_video = session.get_videos()[session.get_current_video_index()]
    st.header("📝 Тест по теории")
    st.info(f"Тема: {current_video['title']}")

    if 'theory_questions' not in st.session_state:
        with st.spinner("Генерация вопросов…"):
            qn = int(APP_CONFIG.get("theory_questions_count", 5))
            data = tutor.generate_theory_questions(
                topic=current_video['title'],
                subject=session.get_subject(),
                grade=session.get_grade(),
                questions_count=qn
            )
            # Обработка ошибок DeepSeek
            if isinstance(data, dict) and data.get("error") in ("402", "deepseek_disabled", "timeout"):
                st.error("Не удалось сгенерировать вопросы (DeepSeek недоступен). Попробуйте позже.")
                st.session_state.theory_questions = []
            else:
                if isinstance(data, dict) and 'content' in data:
                    try:
                        data = json.loads(data['content'])
                    except Exception:
                        data = {"questions": []}
                questions = data.get("questions", [])
                # Гарантируем ровно qn вопросов (если пришло больше/меньше)
                questions = questions[:qn]
                while len(questions) < qn:
                    questions.append({
                        "question": "Вопрос недоступен.",
                        "options": ["A) —", "B) —", "C) —", "D) —"],
                        "correct_answer": "A",
                        "explanation": "Объяснение недоступно."
                    })
                st.session_state.theory_questions = questions
            st.session_state.theory_answers = {}

    if st.session_state.theory_questions:
        for i, q in enumerate(st.session_state.theory_questions):
            st.markdown('<div class="task-card">', unsafe_allow_html=True)
            st.markdown(f"**Вопрос {i+1}:** {q.get('question','')}", unsafe_allow_html=True)
            options = q.get('options', [])
            selected = st.radio("Выберите ответ:", options, key=f"theory_q_{i}", index=None)
            if selected:
                # храним «букву» (A/B/C/D)
                st.session_state.theory_answers[i] = (selected or "")[:1]
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


def show_theory_results(tutor: EnhancedAITutor, session: SessionManager):
    current_video = session.get_videos()[session.get_current_video_index()]
    topic_key = f"{session.get_subject()}_{session.get_grade()}_{current_video['title']}"

    st.markdown('<div class="progress-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Результаты тестирования")

    correct_count = 0
    total_questions = len(st.session_state.theory_questions)

    for i, q in enumerate(st.session_state.theory_questions):
        user_ans = st.session_state.theory_answers.get(i)
        correct = (q.get('correct_answer') or "").strip()[:1]
        options = q.get("options", [])
        is_ok = compare_answers(user_ans, correct)

        if is_ok:
            correct_count += 1
            st.markdown('<div class="success-animation">', unsafe_allow_html=True)
            st.success(f"Вопрос {i+1}: Правильно!")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error(f"Вопрос {i+1}: Неправильно")

        # Подсветка всех вариантов
        out = []
        for opt in options:
            label = (opt or "").strip()
            opt_letter = label[:1] if label else ""
            klass = ""
            if opt_letter == correct:
                klass = "choice choice-correct"
            elif user_ans and opt_letter == (user_ans[:1] if isinstance(user_ans, str) else user_ans):
                klass = "choice choice-wrong"
            if klass:
                out.append(f'<span class="{klass}">{label}</span>')
            else:
                out.append(f"{label}")
        st.markdown("<br>".join(out), unsafe_allow_html=True)

        exp = q.get('explanation', '')
        if exp:
            st.markdown(f"**Объяснение:** {exp}", unsafe_allow_html=True)

    score = calculate_score(correct_count, total_questions)
    st.metric("Ваш результат", f"{correct_count}/{total_questions} ({score:.0f}%)")
    session.save_theory_score(topic_key, score)

    if score < tutor.config.get("theory_pass_threshold", 60):
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


def show_practice_stage(tutor: EnhancedAITutor, session: SessionManager):
    current_video = session.get_videos()[session.get_current_video_index()]

    st.header("💪 Практические задания")
    st.info(f"Тема: {current_video['title']}")

    st.markdown("""
<div class="notebook-note">
📝 <b>Совет:</b> Для сложных задач используйте черновик. Введите конечный ответ.
Для неравенств — <code>x >= 2</code> или <code>[2, inf)</code>. Для нескольких условий — <code>and</code> или <code>,</code>.
</div>
""", unsafe_allow_html=True)

    if 'practice_tasks' not in st.session_state:
        with st.spinner("Генерация заданий…"):
            theory_score = session.get_theory_score(current_video['title'])
            data = tutor.generate_practice_tasks(
                topic=current_video['title'],
                subject=session.get_subject(),
                grade=session.get_grade(),
                user_performance=theory_score
            )
            if isinstance(data, dict) and data.get("error") in ("402", "deepseek_disabled", "timeout"):
                st.error("Не удалось сгенерировать задания (DeepSeek недоступен).")
                st.session_state.practice_tasks = {"easy": [], "medium": [], "hard": []}
            else:
                if isinstance(data, dict) and "content" in data:
                    try:
                        data = json.loads(data["content"])
                    except Exception:
                        data = {"easy": [], "medium": [], "hard": []}
                st.session_state.practice_tasks = {
                    "easy": data.get("easy", []),
                    "medium": data.get("medium", []),
                    "hard": data.get("hard", []),
                }
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
    current_type = st.session_state.current_task_type
    current_index = st.session_state.current_task_index
    tasks_of_type = st.session_state.practice_tasks.get(current_type, [])

    if current_index >= len(tasks_of_type):
        curr_idx = task_types.index(current_type)
        if curr_idx < len(task_types) - 1:
            st.session_state.current_task_type = task_types[curr_idx + 1]
            st.session_state.current_task_index = 0
            st.rerun()
        else:
            show_practice_completion(tutor, session)
            return

    current_task = tasks_of_type[current_index]
    task_key = f"{current_type}_{current_index}"

    total_tasks = sum(len(st.session_state.practice_tasks.get(t, [])) for t in task_types)
    completed_tasks = len(st.session_state.completed_tasks)

    col1, col2 = st.columns([3, 1], vertical_alignment="top")
    with col2:
        st.markdown('<div class="progress-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Прогресс")
        st.progress(completed_tasks / max(1, total_tasks))
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
        st.markdown(current_task.get("question", ""), unsafe_allow_html=True)

        user_answer = st.text_input("Ваш ответ:", key=f"answer_{task_key}")
        attempts = st.session_state.task_attempts.get(task_key, 0)
        max_attempts = int(tutor.config.get("max_attempts_per_task", 3))

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
            if current_task.get("answer"):
                st.markdown(f"**Правильный ответ:** {current_task.get('answer','')}", unsafe_allow_html=True)
            if current_task.get("solution"):
                st.markdown(f"**Решение:** {current_task.get('solution','')}", unsafe_allow_html=True)
            if st.button("Следующее задание"):
                move_to_next_task()

        # Накапливаемые подсказки
        if task_key in st.session_state and 'hints' in st.session_state[task_key]:
            st.markdown("### 💡 Подсказки:")
            for h in st.session_state[task_key]['hints']:
                st.info(h)

        st.markdown('</div>', unsafe_allow_html=True)


def check_answer(tutor: EnhancedAITutor, session: SessionManager, task: dict, user_answer: str, task_key: str):
    st.session_state.task_attempts[task_key] = st.session_state.task_attempts.get(task_key, 0) + 1
    attempts = st.session_state.task_attempts[task_key]
    max_attempts = int(tutor.config.get("max_attempts_per_task", 3))

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

            # 1) Локальная «диагностика ошибки»
            diag = diagnose_mistake(user_answer, task.get("answer", ""))
            if task_key not in st.session_state:
                st.session_state[task_key] = {'hints': []}
            st.session_state[task_key]['hints'].append(diag)
            st.info(f"Подсказка: {diag}")

            # 2) Доп. короткая подсказка от LLM (если доступен)
            if DEEPSEEK_ENABLED:
                with st.spinner("Получаю дополнительную подсказку..."):
                    try:
                        hint_resp = tutor._call_deepseek_api(f"""
Студент решал задачу: "{task.get('question','')}"
Правильный ответ: "{task.get('answer','')}"
Ответ студента: "{user_answer}"
Дай очень краткую подсказку (1 предложение) без LaTeX, укажи, где именно возможная ошибка (знак, формат или вычисление).
""")
                        if isinstance(hint_resp, dict) and 'content' in hint_resp:
                            st.session_state[task_key]['hints'].append(hint_resp['content'])
                            st.info(f"Подсказка: {hint_resp['content']}")
                    except Exception:
                        pass

            log_user_action("incorrect_answer", {"task_key": task_key, "attempts": attempts})
        else:
            st.error("Все попытки исчерпаны.")
            if task.get("answer"):
                st.markdown(f"**Правильный ответ:** {task.get('answer','')}", unsafe_allow_html=True)
            if task.get("solution"):
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

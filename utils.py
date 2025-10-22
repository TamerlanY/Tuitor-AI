# utils.py
import os
import json
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from config import APP_CONFIG, UI_CONFIG, SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_TABLE

# ============== helpers ==============

def compare_answers(user_answer, correct_answer):
    user_answer = str(user_answer or "").strip().lower()
    correct_answer = str(correct_answer or "").strip().lower()

    def replace_textual_operators(text: str) -> str:
        text = text.replace("больше или равно", ">=")
        text = text.replace("меньше или равно", "<=")
        text = text.replace("больше", ">")
        text = text.replace("меньше", "<")
        return text

    user_answer = replace_textual_operators(user_answer)
    correct_answer = replace_textual_operators(correct_answer)

    # Вариант «A) …»
    if len(user_answer) >= 1 and user_answer[0] in "abcd" and (")" in user_answer or "." in user_answer):
        user_answer = user_answer[0]
    if len(correct_answer) >= 1 and correct_answer[0] in "abcd" and (")" in correct_answer or "." in correct_answer):
        correct_answer = correct_answer[0]

    def normalize_answer(a: str) -> str:
        a = re.sub(r"\s+", "", a)
        a = a.replace("infinity", "inf")
        a = a.replace("−", "-")
        return a

    user_answer = normalize_answer(user_answer)
    correct_answer = normalize_answer(correct_answer)

    if any(op in user_answer for op in [">=", "<=", ">", "<"]):
        up = re.split(r"(?:and|or|,|;)", user_answer)
        cp = re.split(r"(?:and|or|,|;)", correct_answer)
        up = sorted([normalize_answer(p) for p in up if p])
        cp = sorted([normalize_answer(p) for p in cp if p])
        return up == cp

    if any(c in user_answer for c in ["[", "]", "(", ")"]):
        return user_answer == correct_answer

    if "," in user_answer or "," in correct_answer:
        return set(user_answer.split(",")) == set(correct_answer.split(","))

    if "/" in user_answer or "/" in correct_answer:
        try:
            u = eval(user_answer)
            c = eval(correct_answer)
            return abs(float(u) - float(c)) < 1e-6
        except Exception:
            pass

    return user_answer == correct_answer or (len(correct_answer) > 0 and user_answer == correct_answer[0])

def calculate_score(correct: int, total: int) -> float:
    return (correct / total * 100) if total > 0 else 0.0

def generate_progress_report(progress_data, topic_key):
    info = progress_data.get("scores", {}).get(topic_key, {})
    rows = ["<h3>📈 Отчёт о прогрессе</h3>", "<ul>"]
    if "theory_score" in info:
        rows.append(f"<li>Теория: {info['theory_score']:.0f}%</li>")
    if "practice_completed" in info:
        p = calculate_score(info.get("practice_completed", 0), info.get("practice_total", 1))
        rows.append(f"<li>Практика: {info.get('practice_completed',0)}/{info.get('practice_total',0)} ({p:.0f}%)</li>")
    rows.append(f"<li>Дата: {info.get('date','N/A')}</li>")
    rows.append("</ul>")
    return "\n".join(rows)

def get_subject_emoji(subject):
    return {
        "Алгебра": "🔢",
        "Геометрия": "📐",
        "Физика": "⚛️",
        "Химия": "🧪",
        "Английский язык": "🇬🇧",
    }.get(subject, "📚")

def create_progress_chart_data(progress_data):
    scores = progress_data.get("scores", {})
    if not scores:
        return None
    rows = []
    for topic_key, info in scores.items():
        subject, grade, topic = topic_key.split("_", 2)
        theory = info.get("theory_score", 0)
        practice = calculate_score(info.get("practice_completed", 0), info.get("practice_total", 1))
        rows.append({
            "Тема": f"{subject} {grade} — {topic[:30]}{'...' if len(topic) > 30 else ''}",
            "Теория (%)": theory,
            "Практика (%)": practice,
            "Дата": info.get("date", "N/A"),
        })
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="Тема", y=["Теория (%)", "Практика (%)"], barmode="group", title="Прогресс по темам", height=320)
    fig.update_layout(yaxis_title="%", legend_title="Тип", margin=dict(t=40, b=60))
    return fig

def log_user_action(action, details):
    entry = {"timestamp": datetime.now().isoformat(), "action": action, "details": details}
    try:
        with open("user_actions.log", "a", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    except Exception:
        pass

# ============== Session / Progress (локально или в облаке) ==============

class SessionManager:
    """
    Хранение прогресса:
    - если заданы SUPABASE_URL/ANON_KEY и есть user_id => облако (таблица SUPABASE_TABLE)
    - иначе => локально в progress.json
    """
    def __init__(self, user_id: str | None = None):
        self.user_id = (user_id or "").strip() or None
        self.progress_file = APP_CONFIG["progress_file"]

        # Локальные state-поля
        if "progress" not in st.session_state:
            st.session_state.progress = self.load_progress()
        if "current_stage" not in st.session_state:
            st.session_state.current_stage = "selection"
        if "videos" not in st.session_state:
            st.session_state.videos = []
        if "current_video_index" not in st.session_state:
            st.session_state.current_video_index = 0
        if "selected_subject" not in st.session_state:
            st.session_state.selected_subject = None
        if "selected_grade" not in st.session_state:
            st.session_state.selected_grade = None

    # ---- Supabase helpers ----
    @property
    def _cloud_enabled(self) -> bool:
        return bool(SUPABASE_URL and SUPABASE_ANON_KEY and self.user_id)

    def _get_supabase(self):
        try:
            from supabase import create_client
            url = SUPABASE_URL
            key = SUPABASE_ANON_KEY
            if not (url and key):
                return None
            return create_client(url, key)
        except Exception:
            return None

    # ---- Загрузка / Сохранение ----
    def load_progress(self):
        if self._cloud_enabled:
            sb = self._get_supabase()
            if sb is not None:
                try:
                    # ожидаем схему таблицы: user_id (text, PK/unique), data (jsonb)
                    res = sb.table(SUPABASE_TABLE).select("data").eq("user_id", self.user_id).maybe_single().execute()
                    row = (res.data or {}) if isinstance(res.data, dict) else (res.data[0] if res.data else None)
                    if row and "data" in row and isinstance(row["data"], dict):
                        return row["data"]
                except Exception as e:
                    st.warning(f"⚠️ Не удалось прочитать облачный прогресс: {e}")
        # fallback локально
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"completed_topics": [], "scores": {}}

    def save_progress(self):
        if self._cloud_enabled:
            sb = self._get_supabase()
            if sb is not None:
                try:
                    payload = {"user_id": self.user_id, "data": st.session_state.progress}
                    # upsert по user_id
                    sb.table(SUPABASE_TABLE).upsert(payload, on_conflict="user_id").execute()
                    return
                except Exception as e:
                    st.warning(f"⚠️ Не удалось сохранить в облако: {e}")
        # fallback локально
        try:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(st.session_state.progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"❌ Ошибка сохранения прогресса: {e}")

    # ---- Остальные методы ----
    def set_course(self, subject, grade):
        st.session_state.selected_subject = subject
        st.session_state.selected_grade = grade

    def get_subject(self):
        return st.session_state.selected_subject

    def get_grade(self):
        return st.session_state.selected_grade

    def start_course(self, videos):
        st.session_state.videos = videos
        completed_titles = [
            t.split("_", 2)[-1]
            for t in st.session_state.progress["completed_topics"]
            if t.startswith(f"{self.get_subject()}_{self.get_grade()}_")
        ]
        start_index = 0
        for i, v in enumerate(videos):
            if v["title"] not in completed_titles:
                start_index = i
                break
        st.session_state.current_video_index = start_index
        st.session_state.current_stage = "video"

    def get_videos(self):
        return st.session_state.videos

    def get_current_video_index(self):
        return st.session_state.current_video_index

    def prev_video(self):
        if st.session_state.current_video_index > 0:
            st.session_state.current_video_index -= 1

    def next_video(self):
        if st.session_state.current_video_index < len(st.session_state.videos) - 1:
            st.session_state.current_video_index += 1
            return True
        return False

    def set_stage(self, stage):
        st.session_state.current_stage = stage

    def get_stage(self):
        return st.session_state.current_stage

    def get_progress(self):
        return st.session_state.progress

    def save_theory_score(self, topic_key, score):
        if topic_key not in st.session_state.progress["scores"]:
            st.session_state.progress["scores"][topic_key] = {}
        st.session_state.progress["scores"][topic_key]["theory_score"] = score
        st.session_state.progress["scores"][topic_key]["date"] = datetime.now().isoformat()
        self.save_progress()

    def save_practice_score(self, topic_key, completed, total):
        if topic_key not in st.session_state.progress["completed_topics"]:
            st.session_state.progress["completed_topics"].append(topic_key)
        if topic_key not in st.session_state.progress["scores"]:
            st.session_state.progress["scores"][topic_key] = {}
        st.session_state.progress["scores"][topic_key]["practice_completed"] = completed
        st.session_state.progress["scores"][topic_key]["practice_total"] = total
        st.session_state.progress["scores"][topic_key]["date"] = datetime.now().isoformat()
        self.save_progress()

    def get_theory_score(self, video_title):
        topic_key = f"{self.get_subject()}_{self.get_grade()}_{video_title}"
        return st.session_state.progress["scores"].get(topic_key, {}).get("theory_score", None)

    def get_adaptive_difficulty(self):
        current_video = self.get_videos()[self.get_current_video_index()]
        theory_score = self.get_theory_score(current_video["title"])
        if theory_score is None:
            return "medium"
        elif theory_score < 60:
            return "easy"
        elif theory_score > 85:
            return "hard"
        return "medium"

    def clear_theory_data(self):
        for k in ["theory_questions", "theory_answers"]:
            if k in st.session_state:
                del st.session_state[k]

    def clear_practice_data(self):
        for k in ["practice_tasks", "task_attempts", "completed_tasks", "current_task_type", "current_task_index"]:
            if k in st.session_state:
                del st.session_state[k]

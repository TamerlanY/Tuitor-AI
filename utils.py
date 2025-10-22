import os
import json
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from config import APP_CONFIG, UI_CONFIG, SUPABASE_URL, SUPABASE_ANON_KEY

# ─────────────────────────────────────────────
# Опциональный Supabase
# ─────────────────────────────────────────────
_supabase = None
if (SUPABASE_URL and SUPABASE_ANON_KEY) or (hasattr(st, "secrets") and st.secrets.get("SUPABASE_URL") and st.secrets.get("SUPABASE_ANON_KEY")):
    try:
        from supabase import create_client
        _url = SUPABASE_URL or st.secrets.get("SUPABASE_URL")
        _key = SUPABASE_ANON_KEY or st.secrets.get("SUPABASE_ANON_KEY")
        _supabase = create_client(_url, _key)
    except Exception:
        _supabase = None


def compare_answers(user_answer, correct_answer):
    """
    Сравнивает ответ пользователя с правильным, учитывая числа, множества, неравенства и формат A/B/C/D.
    """
    if user_answer is None or correct_answer is None:
        return False

    user_answer = str(user_answer).strip().lower()
    correct_answer = str(correct_answer).strip().lower()

    # Нормализация текстовых операторов
    def repl_ops(text):
        text = text.replace("больше или равно", ">=")
        text = text.replace("меньше или равно", "<=")
        text = text.replace("больше", ">")
        text = text.replace("меньше", "<")
        return text

    user_answer = repl_ops(user_answer)
    correct_answer = repl_ops(correct_answer)

    # Только буквы A/B/C/D для теории
    if user_answer in ["a", "b", "c", "d"] and correct_answer in ["a", "b", "c", "d"]:
        return user_answer == correct_answer

    # Удаляем пробелы/скобки
    def norm(a):
        a = re.sub(r"\s+", "", a)
        a = a.replace("infinity", "inf")
        a = re.sub(r"[()]+", "", a)
        return a

    user_answer = norm(user_answer)
    correct_answer = norm(correct_answer)

    # Неравенства
    if any(op in user_answer for op in [">=", "<=", ">", "<"]):
        up = sorted([norm(p) for p in re.split(r"(?:and|or|,|;)", user_answer) if p])
        cp = sorted([norm(p) for p in re.split(r"(?:and|or|,|;)", correct_answer) if p])
        return up == cp

    # Интервалы
    if any(c in user_answer for c in ["[", "]", "(", ")"]):
        return user_answer.replace(" ", "") == correct_answer.replace(" ", "")

    # Множества через запятую
    if "," in user_answer or "," in correct_answer:
        return set(user_answer.split(",")) == set(correct_answer.split(","))

    # Дроби (1/2 == 0.5)
    if "/" in user_answer or "/" in correct_answer:
        try:
            u = eval(user_answer)
            c = eval(correct_answer)
            return abs(u - c) < 1e-6
        except Exception:
            pass

    return user_answer == correct_answer or (correct_answer and user_answer == correct_answer[0])


def calculate_score(correct, total):
    return (correct / total * 100) if total > 0 else 0


def generate_progress_report(progress_data, topic_key):
    report = "<h3>📈 Отчет о прогрессе</h3><ul>"
    sc = progress_data.get("scores", {}).get(topic_key, {})
    if "theory_score" in sc:
        report += f"<li>Теория: {sc['theory_score']:.0f}%</li>"
    if "practice_completed" in sc:
        p = calculate_score(sc.get("practice_completed", 0), sc.get("practice_total", 1))
        report += f"<li>Практика: {sc.get('practice_completed',0)}/{sc.get('practice_total',0)} ({p:.0f}%)</li>"
    report += f"<li>Дата: {sc.get('date','N/A')}</li></ul>"
    return report


def get_subject_emoji(subject):
    emojis = {"Алгебра": "🔢", "Геометрия": "📐", "Физика": "⚛️", "Химия": "🧪", "Английский язык": "🇬🇧"}
    return emojis.get(subject, "📚")


class SessionManager:
    """
    Управление состоянием и прогрессом. Если указан user_id и доступен Supabase — сохраняем в облако.
    Иначе — локальный progress.json.
    """
    def __init__(self, user_id: str = None):
        self.progress_file = APP_CONFIG["progress_file"]
        self.user_id = user_id
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

    # ───── сохранение/загрузка прогресса ─────
    def _load_cloud(self):
        if not (_supabase and self.user_id):
            return None
        try:
            res = _supabase.table("user_progress").select("progress").eq("user_id", self.user_id).single().execute()
            data = (res.data or {}).get("progress")
            return data
        except Exception:
            return None

    def _save_cloud(self, data: dict):
        if not (_supabase and self.user_id):
            return False
        try:
            payload = {"user_id": self.user_id, "progress": data, "updated_at": datetime.utcnow().isoformat()}
            _supabase.table("user_progress").upsert(payload, on_conflict="user_id").execute()
            return True
        except Exception:
            return False

    def load_progress(self):
        # сначала пробуем облако
        cloud = self._load_cloud()
        if cloud is not None:
            return cloud
        # локально
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"completed_topics": [], "scores": {}}

    def save_progress(self):
        # облако
        if self._save_cloud(st.session_state.progress):
            return
        # локально
        try:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(st.session_state.progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"❌ Ошибка сохранения прогресса: {str(e)}")

    # ───── курс и стадия ─────
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

    # ───── очки ─────
    def save_theory_score(self, topic_key, score):
        st.session_state.progress["scores"].setdefault(topic_key, {})
        st.session_state.progress["scores"][topic_key]["theory_score"] = score
        st.session_state.progress["scores"][topic_key]["date"] = datetime.now().isoformat()
        self.save_progress()

    def save_practice_score(self, topic_key, completed, total):
        if topic_key not in st.session_state.progress["completed_topics"]:
            st.session_state.progress["completed_topics"].append(topic_key)
        st.session_state.progress["scores"].setdefault(topic_key, {})
        st.session_state.progress["scores"][topic_key]["practice_completed"] = completed
        st.session_state.progress["scores"][topic_key]["practice_total"] = total
        st.session_state.progress["scores"][topic_key]["date"] = datetime.now().isoformat()
        self.save_progress()

    def get_theory_score(self, video_title):
        topic_key = f"{self.get_subject()}_{self.get_grade()}_{video_title}"
        return st.session_state.progress["scores"].get(topic_key, {}).get("theory_score", None)

    def get_adaptive_difficulty(self):
        # для практики (оставим как есть)
        current_video = self.get_videos()[self.get_current_video_index()]
        ts = self.get_theory_score(current_video["title"])
        if ts is None:
            return "medium"
        if ts < 60:
            return "easy"
        if ts > 85:
            return "hard"
        return "medium"

    # ───── чистка ─────
    def clear_theory_data(self):
        for key in ["theory_questions", "theory_answers"]:
            if key in st.session_state:
                del st.session_state[key]

    def clear_practice_data(self):
        for key in ["practice_tasks", "task_attempts", "completed_tasks", "current_task_type", "current_task_index"]:
            if key in st.session_state:
                del st.session_state[key]


def create_progress_chart_data(progress_data):
    scores = progress_data.get("scores", {})
    if not scores:
        return None

    data = []
    for topic_key, score_info in scores.items():
        subject, grade, topic = topic_key.split("_", 2)
        theory_score = score_info.get("theory_score", 0)
        practice_score = calculate_score(score_info.get("practice_completed", 0), score_info.get("practice_total", 1))
        data.append(
            {
                "Тема": f"{subject} {grade} — {topic[:20]}...",
                "Теория (%)": theory_score,
                "Практика (%)": practice_score,
                "Дата": score_info.get("date", "N/A"),
            }
        )

    df = pd.DataFrame(data)
    fig = px.bar(
        df,
        x="Тема",
        y=["Теория (%)", "Практика (%)"],
        barmode="group",
        title="Прогресс по темам",
        height=300,
    )
    fig.update_layout(yaxis_title="Результат (%)", legend_title="Тип", margin=dict(t=50, b=50))
    return fig


def log_user_action(action, details):
    log_entry = {"timestamp": datetime.now().isoformat(), "action": action, "details": details}
    log_file = "user_actions.log"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")
    except Exception:
        pass

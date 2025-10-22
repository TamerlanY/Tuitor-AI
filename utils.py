import os
import json
import pandas as pd
import plotly.express as px
from datetime import datetime
from config import APP_CONFIG, UI_CONFIG, SUPABASE_URL, SUPABASE_ANON_KEY
import re
import streamlit as st

# --- OPTIONAL: Supabase client ---
SUPABASE = None
try:
    # сначала пробуем secrets (Streamlit Cloud), потом env
    sb_url = st.secrets.get("SUPABASE_URL", None) if hasattr(st, "secrets") else None
    sb_key = st.secrets.get("SUPABASE_ANON_KEY", None) if hasattr(st, "secrets") else None
    sb_url = sb_url or SUPABASE_URL
    sb_key = sb_key or SUPABASE_ANON_KEY
    if sb_url and sb_key:
        from supabase import create_client
        SUPABASE = create_client(sb_url, sb_key)
except Exception:
    SUPABASE = None

SUPABASE_TABLE = "user_progress"  # создадим такую таблицу

def compare_answers(user_answer, correct_answer):
    """Гибкое сравнение ответов (текст, интервалы, неравенства, множества, дроби)."""
    user_answer = str(user_answer or "").strip().lower()
    correct_answer = str(correct_answer or "").strip().lower()

    def replace_textual_operators(text):
        text = text.replace("больше или равно", ">=")
        text = text.replace("меньше или равно", "<=")
        text = text.replace("больше", ">")
        text = text.replace("меньше", "<")
        return text

    user_answer = replace_textual_operators(user_answer)
    correct_answer = replace_textual_operators(correct_answer)

    def normalize_answer(answer):
        answer = re.sub(r'\s+', '', answer)
        answer = answer.replace('infinity', 'inf')
        answer = re.sub(r'[()]+', '', answer)
        return answer

    user_answer = normalize_answer(user_answer)
    correct_answer = normalize_answer(correct_answer)

    if any(op in user_answer for op in ['>=', '<=', '>', '<']):
        user_parts = re.split(r'(?:and|or|,|;)', user_answer)
        correct_parts = re.split(r'(?:and|or|,|;)', correct_answer)
        user_parts = sorted([normalize_answer(p) for p in user_parts if p])
        correct_parts = sorted([normalize_answer(p) for p in correct_parts if p])
        return user_parts == correct_parts

    if any(c in user_answer for c in ['[', ']', '(', ')']):
        user_answer = user_answer.replace(' ', '')
        correct_answer = correct_answer.replace(' ', '')
        return user_answer == correct_answer

    if ',' in user_answer or ',' in correct_answer:
        user_set = set([s for s in user_answer.split(',') if s])
        correct_set = set([s for s in correct_answer.split(',') if s])
        return user_set == correct_set

    if '/' in user_answer:
        try:
            user_val = eval(user_answer)
            correct_val = eval(correct_answer)
            return abs(user_val - correct_val) < 1e-6
        except Exception:
            pass

    return user_answer == correct_answer or (correct_answer and user_answer == correct_answer[0])

def calculate_score(correct, total):
    return (correct / total * 100) if total > 0 else 0

def generate_progress_report(progress_data, topic_key):
    report = "<h3>📈 Отчет о прогрессе</h3><ul>"
    topic_scores = progress_data.get("scores", {}).get(topic_key, {})
    if "theory_score" in topic_scores:
        report += f"<li>Теория: {topic_scores['theory_score']:.0f}%</li>"
    if "practice_completed" in topic_scores:
        pc = topic_scores['practice_completed']
        pt = topic_scores.get('practice_total', 1)
        report += f"<li>Практика: {pc}/{pt} ({calculate_score(pc, pt):.0f}%)</li>"
    report += f"<li>Дата: {topic_scores.get('date', 'N/A')}</li>"
    report += "</ul>"
    return report

def get_subject_emoji(subject):
    emojis = {"Алгебра": "🔢", "Геометрия": "📐", "Физика": "⚛️", "Химия": "🧪", "Английский язык": "🇬🇧"}
    return emojis.get(subject, "📚")

class SessionManager:
    """Управление прогрессом: Supabase (если есть user_id и ключи), иначе локальный файл."""
    def __init__(self, user_id=None):
        self.progress_file = APP_CONFIG["progress_file"]
        self.user_id = user_id
        if 'progress' not in st.session_state:
            st.session_state.progress = self.load_progress()
        if 'current_stage' not in st.session_state:
            st.session_state.current_stage = 'selection'
        if 'videos' not in st.session_state:
            st.session_state.videos = []
        if 'current_video_index' not in st.session_state:
            st.session_state.current_video_index = 0
        if 'selected_subject' not in st.session_state:
            st.session_state.selected_subject = None
        if 'selected_grade' not in st.session_state:
            st.session_state.selected_grade = None

    # ---------- Supabase helpers ----------
    def _sb_enabled(self):
        return (self.user_id is not None) and bool(self.user_id.strip()) and (SUPABASE is not None)

    def load_progress(self):
        # Supabase → progress.json fallback
        if self._sb_enabled():
            try:
                resp = SUPABASE.table(SUPABASE_TABLE).select("progress").eq("user_id", self.user_id).maybe_single().execute()
                row = resp.data
                if row and isinstance(row, dict) and row.get("progress"):
                    return row["progress"]
                return {"completed_topics": [], "scores": {}}
            except Exception as e:
                st.warning(f"⚠️ Не удалось прочитать Supabase, используем локальный файл. ({e})")

        # локальный файл
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"completed_topics": [], "scores": {}}

    def save_progress(self):
        # Пишем в Supabase, если можно
        if self._sb_enabled():
            try:
                payload = {
                    "user_id": self.user_id,
                    "progress": st.session_state.progress,
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                }
                SUPABASE.table(SUPABASE_TABLE).upsert(payload, on_conflict="user_id").execute()
                return
            except Exception as e:
                st.warning(f"⚠️ Не удалось сохранить Supabase, сохраняем локально. ({e})")

        # Файл — fallback
        try:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(st.session_state.progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"❌ Ошибка сохранения прогресса локально: {str(e)}")

    # ---------- Курс ----------
    def set_course(self, subject, grade):
        st.session_state.selected_subject = subject
        st.session_state.selected_grade = grade

    def get_subject(self):
        return st.session_state.selected_subject

    def get_grade(self):
        return st.session_state.selected_grade

    def start_course(self, videos):
        st.session_state.videos = videos
        subject = self.get_subject()
        grade = self.get_grade()
        completed_titles = [
            t.split("_", 2)[-1]
            for t in st.session_state.progress["completed_topics"]
            if t.startswith(f"{subject}_{grade}_")
        ]
        start_index = 0
        for i, video in enumerate(videos):
            if video['title'] not in completed_titles:
                start_index = i
                break
        st.session_state.current_video_index = start_index
        st.session_state.current_stage = 'video'

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

    # ---------- Скора и ключи ----------
    def _topic_key(self, video_title):
        return f"{self.get_subject()}_{self.get_grade()}_{video_title}"

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

    def get_theory_score(self, video_title_or_topic_key):
        # принимает и чистый title, и уже собранный topic_key
        if "_" in video_title_or_topic_key and video_title_or_topic_key.count("_") >= 2:
            topic_key = video_title_or_topic_key
        else:
            topic_key = self._topic_key(video_title_or_topic_key)
        return st.session_state.progress["scores"].get(topic_key, {}).get("theory_score", None)

    def get_adaptive_difficulty(self):
        current_video = self.get_videos()[self.get_current_video_index()]
        theory_score = self.get_theory_score(current_video['title'])
        if theory_score is None:
            return "medium"
        elif theory_score < 60:
            return "easy"
        elif theory_score > 85:
            return "hard"
        return "medium"

    def clear_theory_data(self):
        for k in ['theory_questions', 'theory_answers']:
            if k in st.session_state:
                del st.session_state[k]

    def clear_practice_data(self):
        for k in ['practice_tasks', 'task_attempts', 'completed_tasks', 'current_task_type', 'current_task_index']:
            if k in st.session_state:
                del st.session_state[k]

def create_progress_chart_data(progress_data):
    scores = progress_data.get("scores", {})
    if not scores:
        return None
    data = []
    for topic_key, score_info in scores.items():
        subject, grade, topic = topic_key.split("_", 2)
        theory_score = score_info.get("theory_score", 0)
        practice_score = calculate_score(
            score_info.get("practice_completed", 0),
            score_info.get("practice_total", 1)
        )
        data.append({
            "Тема": f"{subject} {grade} - {topic[:20]}...",
            "Теория (%)": theory_score,
            "Практика (%)": practice_score,
            "Дата": score_info.get("date", "N/A")
        })
    df = pd.DataFrame(data)
    fig = px.bar(
        df, x="Тема", y=["Теория (%)", "Практика (%)"],
        barmode="group", title="Прогресс по темам", height=300
    )
    fig.update_layout(yaxis_title="Результат (%)", legend_title="Тип", margin=dict(t=50, b=50))
    return fig

def log_user_action(action, details):
    log_entry = {"timestamp": datetime.now().isoformat(), "action": action, "details": details}
    try:
        with open("user_actions.log", "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")
    except Exception:
        pass

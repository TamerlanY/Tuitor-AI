import os
import json
import re
from datetime import datetime
from fractions import Fraction

import streamlit as st
import pandas as pd
import plotly.express as px

from config import APP_CONFIG, UI_CONFIG


def _safe_to_float(s: str):
    """Пробует преобразовать строку к числу:
    - сначала как Fraction('1/2') -> 0.5,
    - потом как float('0.5'),
    иначе возвращает None.
    """
    try:
        return float(Fraction(s))
    except Exception:
        pass
    try:
        return float(s)
    except Exception:
        return None


def compare_answers(user_answer, correct_answer):
    """Сравнивает ответ пользователя с правильным.
    Поддержка: A/B/C/D, числа, дроби, множества, интервалы, неравенства.
    """
    user_answer = str(user_answer or "").strip().lower()
    correct_answer = str(correct_answer or "").strip().lower()

    # Нормализация текстовых операторов на символы
    def replace_textual_operators(text):
        text = text.replace("больше или равно", ">=")
        text = text.replace("меньше или равно", "<=")
        text = text.replace("больше", ">")
        text = text.replace("меньше", "<")
        return text

    user_answer = replace_textual_operators(user_answer)
    correct_answer = replace_textual_operators(correct_answer)

    # Быстрый кейс для множественного выбора
    if len(user_answer) == 1 and user_answer in "abcd":
        if len(correct_answer) == 1 and correct_answer in "abcd":
            return user_answer == correct_answer
        # иногда correct может быть "A"
        if correct_answer and correct_answer[0] in "abcd":
            return user_answer == correct_answer[0]

    # Единая нормализация
    def norm(s: str) -> str:
        s = s.replace("infinity", "inf")
        s = s.replace("–", "-").replace("—", "-")
        s = re.sub(r"\s+", "", s)
        return s

    u = norm(user_answer)
    c = norm(correct_answer)

    # Неравенства: разделяем по and/or/','/';'
    if any(op in u for op in ['>=', '<=', '>', '<']):
        user_parts = re.split(r'(?:and|or|,|;)', u)
        correct_parts = re.split(r'(?:and|or|,|;)', c)
        user_parts = sorted([p for p in map(norm, user_parts) if p])
        correct_parts = sorted([p for p in map(norm, correct_parts) if p])
        return user_parts == correct_parts

    # Интервалы вида [a,b) / (a,inf)
    if any(ch in u for ch in "[]()") or any(ch in c for ch in "[]()"):
        return u == c

    # Множества значений "2,-2,0"
    if ',' in u or ',' in c:
        u_set = set([x for x in u.split(',') if x != ""])
        c_set = set([x for x in c.split(',') if x != ""])
        # Пытаемся численно сравнить
        def to_num_set(ss):
            arr = []
            for x in ss:
                val = _safe_to_float(x)
                arr.append(val if val is not None else x)
            return set(arr)
        return to_num_set(u_set) == to_num_set(c_set)

    # Дроби и числа
    u_val = _safe_to_float(u)
    c_val = _safe_to_float(c)
    if u_val is not None and c_val is not None:
        return abs(u_val - c_val) < 1e-9

    # Фоллбек — просто строковое сравнение
    return u == c


def calculate_score(correct, total):
    return (correct / total * 100) if total > 0 else 0


def generate_progress_report(progress_data, topic_key):
    report = "<h3>📈 Отчет о прогрессе</h3><ul>"
    topic_scores = progress_data.get("scores", {}).get(topic_key, {})

    if "theory_score" in topic_scores:
        report += f"<li>Теория: {topic_scores['theory_score']:.0f}%</li>"
    if "practice_completed" in topic_scores:
        done = topic_scores['practice_completed']
        total = topic_scores.get('practice_total', 0)
        report += f"<li>Практика: {done}/{total} ({calculate_score(done, total):.0f}%)</li>"
    report += f"<li>Дата: {topic_scores.get('date', 'N/A')}</li>"
    report += "</ul>"
    return report


def get_subject_emoji(subject):
    emojis = {
        "Алгебра": "🔢",
        "Геометрия": "📐",
        "Физика": "⚛️",
        "Химия": "🧪",
        "Английский язык": "🇬🇧",
    }
    return emojis.get(subject, "📚")


class SessionManager:
    """Управление состоянием сессии и прогрессом."""
    def __init__(self):
        self.progress_file = APP_CONFIG["progress_file"]
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

    # ---- topic_key helpers ----
    def topic_key_for_title(self, video_title: str) -> str:
        return f"{self.get_subject()}_{self.get_grade()}_{video_title}"

    # ---- storage ----
    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"completed_topics": [], "scores": {}}

    def save_progress(self):
        try:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(st.session_state.progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"❌ Ошибка сохранения прогресса: {str(e)}")

    # ---- course/stage ----
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
            if v['title'] not in completed_titles:
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

    # ---- scores ----
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

    def get_theory_score(self, topic_key=None, video_title=None):
        """Можно передать готовый topic_key ИЛИ только title (сформируем ключ сами)."""
        if topic_key is None:
            if video_title is None:
                return None
            topic_key = self.topic_key_for_title(video_title)
        return st.session_state.progress["scores"].get(topic_key, {}).get("theory_score", None)

    def get_adaptive_difficulty(self):
        vids = self.get_videos()
        if not vids:
            return "medium"
        current_video = vids[self.get_current_video_index()]
        theory_score = self.get_theory_score(video_title=current_video['title'])
        if theory_score is None:
            return "medium"
        if theory_score < 60:
            return "easy"
        if theory_score > 85:
            return "hard"
        return "medium"

    # ---- cleanup ----
    def clear_theory_data(self):
        for key in ['theory_questions', 'theory_answers']:
            if key in st.session_state:
                del st.session_state[key]

    def clear_practice_data(self):
        for key in ['practice_tasks', 'task_attempts', 'completed_tasks', 'current_task_type', 'current_task_index']:
            if key in st.session_state:
                del st.session_state[key]


def create_progress_chart_data(progress_data):
    scores = progress_data.get("scores", {})
    if not scores:
        return None

    data = []
    for topic_key, info in scores.items():
        try:
            subject, grade, topic = topic_key.split("_", 2)
        except ValueError:
            subject, grade, topic = "?", "?", topic_key
        theory = info.get("theory_score", 0)
        practice = calculate_score(info.get("practice_completed", 0), max(1, info.get("practice_total", 1)))
        label = f"{subject} {grade} — {topic[:30]}{'...' if len(topic) > 30 else ''}"
        data.append({"Тема": label, "Теория (%)": theory, "Практика (%)": practice, "Дата": info.get("date", "N/A")})

    df = pd.DataFrame(data)
    fig = px.bar(df, x="Тема", y=["Теория (%)", "Практика (%)"], barmode="group",
                 title="Прогресс по темам", height=300)
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

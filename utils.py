import os
import json
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from config import APP_CONFIG, UI_CONFIG

def compare_answers(user_answer, correct_answer):
    """Сравнивает ответ пользователя с правильным, учитывая числа, множества, неравенства и текстовые ошибки."""
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
        answer = re.sub(r"\s+", "", answer)
        answer = answer.replace("infinity", "inf")
        answer = re.sub(r"[()]+", "", answer)
        return answer

    user_answer_norm = normalize_answer(user_answer)
    correct_answer_norm = normalize_answer(correct_answer)

    # неравенства
    if any(op in user_answer_norm for op in [">=", "<=", ">", "<"]):
        user_parts = re.split(r"(?:and|or|,|;)", user_answer_norm)
        correct_parts = re.split(r"(?:and|or|,|;)", correct_answer_norm)
        user_parts = sorted([p for p in user_parts if p])
        correct_parts = sorted([p for p in correct_parts if p])
        return user_parts == correct_parts

    # интервалы
    if any(c in user_answer for c in ["[", "]", "(", ")"]):
        return user_answer.replace(" ", "") == correct_answer.replace(" ", "")

    # множества через запятую
    if "," in user_answer or "," in correct_answer:
        user_set = set(user_answer_norm.split(","))
        correct_set = set(correct_answer_norm.split(","))
        return user_set == correct_set

    # дроби как 1/2
    if "/" in user_answer:
        try:
            user_val = eval(user_answer)
            correct_val = eval(correct_answer)
            return abs(user_val - correct_val) < 1e-6
        except Exception:
            pass

    # простой множественный выбор
    if correct_answer_norm in ["a", "b", "c", "d"]:
        return user_answer_norm == correct_answer_norm or user_answer_norm == correct_answer_norm[0]

    return user_answer_norm == correct_answer_norm


def calculate_score(correct, total):
    return (correct / total * 100) if total > 0 else 0


def generate_progress_report(progress_data, topic_key):
    report = "<h3>📈 Отчет о прогрессе</h3><ul>"
    topic_scores = progress_data.get("scores", {}).get(topic_key, {})
    if "theory_score" in topic_scores:
        report += f"<li>Теория: {topic_scores['theory_score']:.0f}%</li>"
    if "practice_completed" in topic_scores:
        prc = calculate_score(topic_scores.get("practice_completed", 0), topic_scores.get("practice_total", 1))
        report += f"<li>Практика: {topic_scores['practice_completed']}/{topic_scores['practice_total']} ({prc:.0f}%)</li>"
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


# ---------- САНИТАЙЗИНГ ВОПРОСОВ ТЕОРИИ ----------

def _normalize_options(opts):
    """Приводим к 4 опциям A/B/C/D и гарантируем формат 'A) ...'."""
    opts = list(opts or [])
    # срезаем всё что длиннее 4
    opts = opts[:4]
    # добиваем пустыми, если меньше 4
    while len(opts) < 4:
        opts.append("—")

    letters = ["A", "B", "C", "D"]
    fixed = []
    for i, raw in enumerate(opts[:4]):
        text = str(raw or "").strip()
        # Уберём случайные префиксы и проставим "A) ...":
        text = re.sub(r"^[A-Da-d][\)\.\:]\s*", "", text)
        fixed.append(f"{letters[i]}) {text if text else '—'}")
    return fixed


def sanitize_theory_questions(items):
    """Проверяем и чистим вопросы теории. Возвращаем список годных вопросов."""
    safe = []
    for q in items or []:
        question = str(q.get("question", "")).strip()
        options = _normalize_options(q.get("options"))
        correct = str(q.get("correct_answer", "")).strip().upper()
        if correct not in ["A", "B", "C", "D"]:
            # если модель прислала '1','2','3','4' — конвертим
            if correct in ["1", "2", "3", "4"]:
                mapping = {"1": "A", "2": "B", "3": "C", "4": "D"}
                correct = mapping[correct]
            else:
                # попытка угадать по тексту правильного варианта
                # если модель прислала полный текст, попробуем сопоставить
                correct = "A"  # дефолт, чтобы не падать

        explanation = str(q.get("explanation", "")).strip()
        if not question or not options or len(options) != 4:
            continue

        safe.append(
            {
                "question": question,
                "options": options,
                "correct_answer": correct,
                "explanation": explanation,
            }
        )
    return safe


# -------------- Сессия / прогресс --------------

class SessionManager:
    """Управление состоянием сессии и прогрессом."""
    def __init__(self, user_id=None):
        self.user_id = user_id
        self.progress_file = APP_CONFIG["progress_file"]
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
        for i, video in enumerate(videos):
            if video["title"] not in completed_titles:
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
        practice_score = calculate_score(
            score_info.get("practice_completed", 0),
            score_info.get("practice_total", 1),
        )
        data.append(
            {
                "Тема": f"{subject} {grade} - {topic[:20]}...",
                "Теория (%)": theory_score,
                "Практика (%)": practice_score,
                "Дата": score_info.get("date", "N/A"),
            }
        )
    df = pd.DataFrame(data)
    fig = px.bar(df, x="Тема", y=["Теория (%)", "Практика (%)"], barmode="group", title="Прогресс по темам", height=300)
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

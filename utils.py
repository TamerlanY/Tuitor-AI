import os
import json
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from config import APP_CONFIG, UI_CONFIG


def compare_answers(user_answer, correct_answer):
    """Сравнивает ответ пользователя с правильным, учитывая числа, множества, неравенства, интервалы и текстовые опечатки."""
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
        a = re.sub(r"\s+", "", answer)
        a = a.replace("infinity", "inf")
        return a

    ua = normalize_answer(user_answer)
    ca = normalize_answer(correct_answer)

    # Неравенства, возможны составные условия
    if any(op in ua for op in ['>=', '<=', '>', '<']) or any(op in ca for op in ['>=', '<=', '>', '<']):
        user_parts = re.split(r'(?:and|or|,|;)', ua)
        correct_parts = re.split(r'(?:and|or|,|;)', ca)
        user_parts = sorted([p for p in user_parts if p])
        correct_parts = sorted([p for p in correct_parts if p])
        return user_parts == correct_parts

    # Интервалы: сравниваем строково после нормализации пробелов
    if any(c in ua for c in ['[', ']', '(', ')']) or any(c in ca for c in ['[', ']', '(', ')']):
        return ua == ca

    # Множества значений через запятую
    if ',' in ua or ',' in ca:
        u_set = set([x for x in ua.split(',') if x])
        c_set = set([x for x in ca.split(',') if x])
        return u_set == c_set

    # Дроби (например, "1/2") vs числа
    if '/' in ua or '/' in ca:
        try:
            uval = eval(ua.replace("^", "**"))
            cval = eval(ca.replace("^", "**"))
            return abs(float(uval) - float(cval)) < 1e-6
        except Exception:
            pass

    # Прямое сравнение
    return ua == ca


def calculate_score(correct, total):
    """Вычисляет % правильных ответов."""
    return (correct / total * 100) if total > 0 else 0


def generate_progress_report(progress_data, topic_key):
    """HTML-отчёт о прогрессе по теме."""
    report = "<h3>📈 Отчет о прогрессе</h3><ul>"
    topic_scores = progress_data.get("scores", {}).get(topic_key, {})

    if "theory_score" in topic_scores:
        report += f"<li>Теория: {topic_scores['theory_score']:.0f}%</li>"
    if "practice_completed" in topic_scores:
        p = calculate_score(topic_scores.get('practice_completed', 0), topic_scores.get('practice_total', 1))
        report += f"<li>Практика: {topic_scores.get('practice_completed', 0)}/{topic_scores.get('practice_total', 0)} ({p:.0f}%)</li>"
    report += f"<li>Дата: {topic_scores.get('date', 'N/A')}</li>"
    report += "</ul>"
    return report


def get_subject_emoji(subject):
    """Эмодзи по предмету."""
    emojis = {
        "Алгебра": "🔢",
        "Геометрия": "📐",
        "Физика": "⚛️",
        "Химия": "🧪",
        "Английский язык": "🇬🇧"
    }
    return emojis.get(subject, "📚")


class SessionManager:
    """Управление состоянием и локальным прогрессом.

    user_id: сохраняем на будущее (если подключишь БД), сейчас не обязателен.
    """
    def __init__(self, user_id=None):
        self.user_id = user_id
        self.progress_file = APP_CONFIG.get("progress_file", "progress.json")

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

    # ---------- Хранилище прогресса (локально) ----------
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
        completed_titles = [t.split("_", 2)[-1] for t in st.session_state.progress["completed_topics"]
                            if t.startswith(f"{self.get_subject()}_{self.get_grade()}_")]
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

    # ---------- Сохранение результатов ----------
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
        # Оставляем для совместимости — сейчас теория от него не зависит
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
        keys = ['theory_questions', 'theory_answers']
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]

    def clear_practice_data(self):
        keys = ['practice_tasks', 'task_attempts', 'completed_tasks', 'current_task_type', 'current_task_index']
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]


def create_progress_chart_data(progress_data):
    """Строит Plotly-график прогресса."""
    scores = progress_data.get("scores", {})
    if not scores:
        return None

    data = []
    for topic_key, score_info in scores.items():
        try:
            subject, grade, topic = topic_key.split("_", 2)
        except ValueError:
            subject, grade, topic = "?", "?", topic_key
        theory_score = score_info.get("theory_score", 0)
        practice_score = calculate_score(
            score_info.get("practice_completed", 0),
            score_info.get("practice_total", 1)
        )
        data.append({
            "Тема": f"{subject} {grade} — {topic[:32]}{'…' if len(topic) > 32 else ''}",
            "Теория (%)": theory_score,
            "Практика (%)": practice_score,
            "Дата": score_info.get("date", "N/A")
        })

    df = pd.DataFrame(data)
    fig = px.bar(
        df,
        x="Тема",
        y=["Теория (%)", "Практика (%)"],
        barmode="group",
        title="Прогресс по темам",
        height=320
    )
    fig.update_layout(
        yaxis_title="Результат (%)",
        legend_title="Тип",
        margin=dict(t=50, b=50)
    )
    return fig


def log_user_action(action, details):
    """Логирование действий пользователя в файл."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "details": details
    }
    log_file = "user_actions.log"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")
    except Exception:
        pass


# -------- Локальная диагностика ошибок для практики --------
def diagnose_mistake(user_answer: str, correct_answer: str) -> str:
    """
    Пытается подсказать, где именно ошибка:
    - текстовые операторы vs символы (>=, <=, <, >)
    - формат неравенств/интервалов
    - порядок корней (множество значений)
    - десятичная vs дробь
    - пустой ответ
    """
    ua_raw = (str(user_answer or "")).strip()
    ca_raw = (str(correct_answer or "")).strip()
    if not ua_raw:
        return "Ответ пустой. Введите решение."

    def norm_ops(s: str) -> str:
        s = s.lower()
        s = s.replace("больше или равно", ">=").replace("меньше или равно", "<=")
        s = s.replace("больше", ">").replace("меньше", "<")
        return s

    ua = norm_ops(ua_raw)
    ca = norm_ops(ca_raw)

    # Текст вместо символов
    if any(x in ua_raw.lower() for x in ["больше", "меньше"]) and not any(op in ua for op in [">=", "<=", ">", "<"]):
        return "Используйте математические символы неравенств: >=, <=, >, < (не пишите их словами)."

    # Формат интервала
    if any(c in ua for c in "[]()") and not any(c in ca for c in "[]()"):
        return "Формат интервала не требуется. Введите числовой ответ/условие, как в задании."
    if any(c in ca for c in "[]()") and not any(c in ua for c in "[]()"):
        return "Ожидается ответ в виде интервала. Пример: [2, inf) или (-inf, 3]."

    # Множества значений
    if "," in ua or "," in ca:
        us = sorted([x.strip() for x in ua.split(",") if x.strip()])
        cs = sorted([x.strip() for x in ca.split(",") if x.strip()])
        if set(us) == set(cs) and us != cs:
            return "Значения совпадают, но формат/порядок отличается. Проверьте разделители и пробелы."
        if len(us) != len(cs):
            return "Число значений не совпадает. Возможно, пропущено или лишнее значение."

    # Десятичная vs дробь
    if "/" in ua or "/" in ca:
        try:
            uval = eval(ua.replace("^", "**"))
            cval = eval(ca.replace("^", "**"))
            if abs(float(uval) - float(cval)) > 1e-6:
                return "Не совпадает численное значение. Перепроверьте вычисления."
            else:
                return "Численное значение совпадает, но формат отличается. Напишите как десятичное число."
        except Exception:
            pass

    # Неравенства
    if any(op in ua for op in [">=", "<=", ">", "<"]) and not any(op in ca for op in [">=", "<=", ">", "<"]):
        return "Ожидается точное числовое значение, а не неравенство."
    if any(op in ca for op in [">=", "<=", ">", "<"]) and not any(op in ua for op in [">=", "<=", ">", "<"]):
        return "Ожидается неравенство (например, x >= 2). Укажите знак и границу."

    return "Проверьте формат ответа и вычисления: знаки неравенств, интервалы, разделители и порядок значений."

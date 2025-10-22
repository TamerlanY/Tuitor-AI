# utils.py
import os
import json
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
import requests

from config import APP_CONFIG, UI_CONFIG, SUPABASE_URL, SUPABASE_ANON_KEY


def compare_answers(user_answer, correct_answer):
    """
    Сравнивает ответ пользователя с правильным,
    учитывая числа, множества, неравенства и простые текстовые варианты (A/B/C/D).
    """
    user_answer = str(user_answer or "").strip().lower()
    correct_answer = str(correct_answer or "").strip().lower()

    # Нормализация словесных операторов
    def replace_textual_operators(text: str) -> str:
        text = text.replace("больше или равно", ">=")
        text = text.replace("меньше или равно", "<=")
        text = text.replace("больше", ">")
        text = text.replace("меньше", "<")
        return text

    user_answer = replace_textual_operators(user_answer)
    correct_answer = replace_textual_operators(correct_answer)

    # Удаляем лишние пробелы/скобки
    def normalize_answer(ans: str) -> str:
        ans = re.sub(r"\s+", "", ans)
        ans = ans.replace("infinity", "inf")
        ans = re.sub(r"[()]+", "", ans)
        return ans

    user_norm = normalize_answer(user_answer)
    correct_norm = normalize_answer(correct_answer)

    # Если ответ — одна буква (A/B/C/D) — сравниваем как вариант теста
    if len(correct_norm) == 1 and correct_norm in {"a", "b", "c", "d"}:
        return user_norm[:1] == correct_norm

    # Неравенства (возможны несколько условий, через and/or/, ;)
    if any(op in user_norm for op in [">=", "<=", ">", "<"]):
        user_parts = re.split(r"(?:and|or|,|;)", user_norm)
        correct_parts = re.split(r"(?:and|or|,|;)", correct_norm)
        user_parts = sorted([normalize_answer(p) for p in user_parts if p])
        correct_parts = sorted([normalize_answer(p) for p in correct_parts if p])
        return user_parts == correct_parts

    # Интервалы вроде [2, inf) — если вдруг встречаются (мы уже удалили скобки в normalize, так что сравниваем исходные)
    if any(c in (user_answer or "") for c in ["[", "]", "(", ")"]) or any(c in (correct_answer or "") for c in ["[", "]", "(", ")"]):
        return (user_answer or "").replace(" ", "") == (correct_answer or "").replace(" ", "")

    # Наборы (например, "2,-2")
    if "," in user_norm or "," in correct_norm:
        return set(user_norm.split(",")) == set(correct_norm.split(","))

    # Дроби "1/2" vs "0.5"
    if "/" in user_norm or "/" in correct_norm:
        try:
            user_val = eval(user_norm)
            correct_val = eval(correct_norm)
            return abs(user_val - correct_val) < 1e-6
        except Exception:
            pass

    # Обычное строковое совпадение
    return user_norm == correct_norm


def calculate_score(correct, total):
    return (correct / total * 100) if total > 0 else 0


def generate_progress_report(progress_data: dict, topic_key: str) -> str:
    """Генерирует простой HTML-отчёт по одной теме."""
    report = "<h3>📈 Отчет о прогрессе</h3><ul>"
    topic_scores = progress_data.get("scores", {}).get(topic_key, {})

    if "theory_score" in topic_scores:
        report += f"<li>Теория: {topic_scores['theory_score']:.0f}%</li>"

    if "practice_completed" in topic_scores:
        pc = topic_scores.get("practice_completed", 0)
        pt = topic_scores.get("practice_total", 0)
        report += f"<li>Практика: {pc}/{pt} ({calculate_score(pc, pt):.0f}%)</li>"

    report += f"<li>Дата: {topic_scores.get('date', 'N/A')}</li>"
    report += "</ul>"
    return report


def get_subject_emoji(subject: str) -> str:
    emojis = {
        "Алгебра": "🔢",
        "Геометрия": "📐",
        "Физика": "⚛️",
        "Химия": "🧪",
        "Английский язык": "🇬🇧",
    }
    return emojis.get(subject, "📚")


def create_progress_chart_data(progress_data: dict):
    """Возвращает фигуру Plotly для графика прогресса (или None)."""
    scores = progress_data.get("scores", {})
    if not scores:
        return None

    data = []
    for topic_key, score_info in scores.items():
        try:
            subject, grade, topic = topic_key.split("_", 2)
        except ValueError:
            # неожиданный ключ — пропустим
            continue

        theory_score = score_info.get("theory_score", 0)
        practice_score = calculate_score(
            score_info.get("practice_completed", 0),
            score_info.get("practice_total", 0) or 1,
        )
        data.append(
            {
                "Тема": f"{subject} {grade} — {topic[:24]}{'...' if len(topic) > 24 else ''}",
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
        height=320,
    )
    fig.update_layout(
        yaxis_title="Результат (%)",
        legend_title="Тип",
        margin=dict(t=40, b=40),
    )
    return fig


def log_user_action(action: str, details: dict):
    """Лёгкий локальный лог (не блокирует UI)."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "details": details,
    }
    try:
        with open("user_actions.log", "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")
    except Exception:
        pass


# =========================
# SessionManager (прогресс)
# =========================
class SessionManager:
    """
    Работает локально (progress.json) или, если доступны SUPABASE_URL/ANON_KEY и задан user_id,
    пробует синхронизировать прогресс в облаке (таблица `progress` со столбцами:
    - user_id (text, PK)
    - payload (jsonb)
    Политики RLS нужно настроить самостоятельно (разрешить запись anon пользователю по своему user_id).
    """

    def __init__(self, user_id: str | None = None):
        self.progress_file = APP_CONFIG["progress_file"]
        self.user_id = user_id

        # флажок облака
        self._cloud_on = bool(SUPABASE_URL and SUPABASE_ANON_KEY and self.user_id)

        # Инициализация session_state
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

    # ---------- Cloud (Supabase) helpers ----------
    def _cloud_headers(self):
        return {
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Prefer": "resolution=merge-duplicates",
        }

    def _cloud_fetch(self) -> dict | None:
        """
        Читает прогресс из Supabase (таблица progress, поиск по user_id).
        """
        try:
            url = f"{SUPABASE_URL}/rest/v1/progress"
            params = {"select": "user_id,payload", "user_id": f"eq.{self.user_id}"}
            r = requests.get(url, headers=self._cloud_headers(), params=params, timeout=6)
            r.raise_for_status()
            rows = r.json()
            if rows:
                return rows[0].get("payload") or {}
        except Exception:
            return None

    def _cloud_upsert(self, payload: dict) -> bool:
        """
        Пишет/апдейтит прогресс в Supabase.
        """
        try:
            url = f"{SUPABASE_URL}/rest/v1/progress"
            body = {"user_id": self.user_id, "payload": payload}
            r = requests.post(url, headers=self._cloud_headers(), json=body, timeout=6)
            # 201/204 — ок
            return r.status_code in (200, 201, 204)
        except Exception:
            return False

    # ---------- Local ----------
    def _local_load(self) -> dict:
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"completed_topics": [], "scores": {}}

    def _local_save(self, payload: dict):
        try:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"❌ Ошибка сохранения прогресса: {str(e)}")

    # ---------- Public API ----------
    def load_progress(self) -> dict:
        """
        Если доступно облако — пытаемся взять оттуда; иначе — локально.
        Если облако недоступно/ошибка — используем локальные данные.
        """
        if self._cloud_on:
            remote = self._cloud_fetch()
            if isinstance(remote, dict) and remote:
                return remote
        # fallback
        return self._local_load()

    def save_progress(self):
        """
        Сохраняем в локальный файл; если доступно облако — пытаемся синхронизировать.
        """
        payload = st.session_state.progress
        # локально
        self._local_save(payload)
        # облако (лучше не падать на UI, если ошибка — просто молча пропустить)
        if self._cloud_on:
            self._cloud_upsert(payload)

    # ----- Курс/видео навигация -----
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
            for t in st.session_state.progress.get("completed_topics", [])
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

    # ----- Сохранения -----
    def save_theory_score(self, topic_key, score: float):
        if "scores" not in st.session_state.progress:
            st.session_state.progress["scores"] = {}
        if topic_key not in st.session_state.progress["scores"]:
            st.session_state.progress["scores"][topic_key] = {}
        st.session_state.progress["scores"][topic_key]["theory_score"] = score
        st.session_state.progress["scores"][topic_key]["date"] = datetime.now().isoformat()
        self.save_progress()

    def save_practice_score(self, topic_key, completed: int, total: int):
        if "completed_topics" not in st.session_state.progress:
            st.session_state.progress["completed_topics"] = []
        if topic_key not in st.session_state.progress["completed_topics"]:
            st.session_state.progress["completed_topics"].append(topic_key)

        if "scores" not in st.session_state.progress:
            st.session_state.progress["scores"] = {}
        if topic_key not in st.session_state.progress["scores"]:
            st.session_state.progress["scores"][topic_key] = {}

        st.session_state.progress["scores"][topic_key]["practice_completed"] = completed
        st.session_state.progress["scores"][topic_key]["practice_total"] = total
        st.session_state.progress["scores"][topic_key]["date"] = datetime.now().isoformat()
        self.save_progress()

    def get_theory_score(self, video_title: str):
        topic_key = f"{self.get_subject()}_{self.get_grade()}_{video_title}"
        return st.session_state.progress.get("scores", {}).get(topic_key, {}).get("theory_score", None)

    def get_adaptive_difficulty(self):
        """
        Возвращает easy/medium/hard исходя из теоретического результата по текущему видео.
        (используется только для практики)
        """
        if not self.get_videos():
            return "medium"
        current_video = self.get_videos()[self.get_current_video_index()]
        theory_score = self.get_theory_score(current_video["title"])
        if theory_score is None:
            return "medium"
        if theory_score < 60:
            return "easy"
        if theory_score > 85:
            return "hard"
        return "medium"

    # ---- Сбросы состояния на экранах ----
    def clear_theory_data(self):
        for key in ["theory_questions", "theory_answers"]:
            if key in st.session_state:
                del st.session_state[key]

    def clear_practice_data(self):
        for key in ["practice_tasks", "task_attempts", "completed_tasks", "current_task_type", "current_task_index"]:
            if key in st.session_state:
                del st.session_state[key]

# config.py
from dotenv import load_dotenv
import os

# Для локалки: подтянуть .env (на проде можно просто env переменные)
load_dotenv()

# API-ключи здесь не валидируем, т.к. app.py сам читает их и стопает при отсутствии
# YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

PLAYLISTS = {
    "Алгебра": {
        "7":  "PLCRqj4jDCIYmUtgQCGy3l5GGYbiDBR3p-",
        "8":  "PLCRqj4jDCIYkk9CMV6wBQR16eHz_SRU1j",
        "9":  "PLCRqj4jDCIYl7ZP0JefXdLcXcEIh8LY5m",
        "10": "PLCRqj4jDCIYlaZBTUCrK2xq65quwfPEXE",
        "11": "PLCRqj4jDCIYkL1lREiEg-APcjYbvfqCdn",
    },
    "Геометрия": {
        "7":  "PLeRoaPcjXF1ffHLNvtX66hXxtHLNvf9aw",
        "8":  "PLeRoaPcjXF1ecF21g27Q7ZDfF8Gktvwf0",
        "9":  "PLeRoaPcjXF1dNqOv9ghksZF4GG1cVkg74",
        "10": "PLeRoaPcjXF1eelu2Ou-iy5A0ntPOwjIjq",
        "11": "PLeRoaPcjXF1eg1bQxQX_KkM4EpEdBedIu",
    },
    "Физика": {
        "7":  "PLdjp7wVqN3WtJGEEvLOcgTG3J8cv5sdew",
        "8":  "PLdjp7wVqN3Wssi0MhFBZTuiz6Ev5YHVBa",
        "9":  "PLdjp7wVqN3Wv_OjT7TdWbY91v0rk4VA53",
        "10": "PLdjp7wVqN3Wu8hFD-nzI6vQe3tqLXzIA-",
        "11": "PLdjp7wVqN3WtM6h-DpIRXBe5iYdySPzF9",
    },
    "Химия": {
        "7":  "PLoe4L7cYJo_WFiJs6BqpJ6zuYMjnB_vLe",
        "8":  "PLoe4L7cYJo_U-02hkvjDHAa0e8hQAb_Dx",
        "9":  "PLoe4L7cYJo_VgSiVkh-I1bNdFMc1U5jqO",
        "10": "PLoe4L7cYJo_Udsk8PI85OJggGpz7nj5J7",
        "11": "PLoe4L7cYJo_V_QrkuRbYaJuVyR3dYKlDa",
    },
    "Английский язык": {
        "7":  "PLD6SPjEPomatk5Pp2z7j-9kOxmgTUiRSr",
        "8":  "PL7j3OJlBURb7jc_Romw7Sw0bKRLK2X9GY",
        "9":  "PLD6SPjEPomasUQxxBEBNyZGbzZY6pEfPQ",
        "10": "PLYB0SmefqEskabgi9CfLoYtXTA3U8VNKS",
    },
}

APP_CONFIG = {
    "youtube_max_results": 50,
    "theory_questions_count": 5,
    "tasks_per_difficulty": {"easy": 3, "medium": 3, "hard": 2},
    "max_attempts_per_task": 3,
    "theory_pass_threshold": 60,
    "progress_file": "progress.json",
}

DEEPSEEK_CONFIG = {
    "model": "deepseek-chat",
    "temperature": 0.7,
    "max_tokens": 4000,
    "retry_attempts": 3,
    "timeout": 30,
}

UI_CONFIG = {
    "page_title": "AI Тьютор",
    "page_icon": "📚",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "task_type_names": {
        "easy": "Легкий уровень",
        "medium": "Средний уровень",
        "hard": "Сложный уровень",
    },
}

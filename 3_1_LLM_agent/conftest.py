# conftest.py

import sys
from pathlib import Path

# Тесты лежат в подпапке tests, а пакет llm_agent - рядом с ней. Без этой строки
# "from llm_agent..." работает только при запуске pytest из этой директории.
sys.path.insert(0, str(Path(__file__).resolve().parent))

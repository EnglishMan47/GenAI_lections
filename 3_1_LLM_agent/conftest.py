# conftest.py

import sys
from pathlib import Path

# Тесты лежат в подпапке tests, а рядом пакет llm_agent. Без этой строки работает только при запуске pytest из этой папки
sys.path.insert(0, str(Path(__file__).resolve().parent))

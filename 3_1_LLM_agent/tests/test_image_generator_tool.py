# tests/test_image_generator_tool.py

"""
Юнит-тесты инструмента ImageGeneratorTool.

Все обращения к сети подменяются заглушками: тесты должны проходить в GitHub
Actions, где нет ни токенов, ни доступа к сервисам генерации, и не тратить
бесплатные лимиты аккаунта.
"""

import base64
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# При запуске через pytest путь добавляет conftest.py, но файл можно запустить и
# напрямую - тогда пакет llm_agent нужно найти самим.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from llm_agent.tool_image_generator import (
    MAX_PROMPT_LENGTH,
    REPLICATE_ASPECT_RATIOS,
    ImageGeneratorTool,
)

# Минимальный валидный ответ сервиса: несколько байт, изображающих картинку.
FAKE_IMAGE = b"\xff\xd8\xff\xe0fake jpeg content"
FAKE_HOST_URL = "https://example-host.test/abc123.png"


def make_response(status=200, content=FAKE_IMAGE, json_data=None,
                  content_type="image/jpeg", text=""):
    """Собирает заглушку ответа requests с нужными полями."""
    response = Mock()
    response.status_code = status
    response.content = content
    response.text = text
    response.headers = {"Content-Type": content_type}
    response.json = Mock(return_value=json_data or {})
    response.raise_for_status = Mock()
    response.close = Mock()
    return response


def make_http_error(status):
    """Собирает исключение HTTPError с заполненным полем response."""
    error = requests.exceptions.HTTPError(f"{status} Client Error")
    error.response = Mock()
    error.response.status_code = status
    return error


class TestImageGeneratorTool(unittest.TestCase):
    """Проверка разбора запроса, выбора провайдера, генерации и обработки ошибок."""

    def setUp(self):
        """Готовит инструмент с временной папкой и фиктивным токеном."""
        self.temp_dir = tempfile.mkdtemp()
        self.tool = ImageGeneratorTool(
            provider="huggingface",
            api_key="hf_test_token",
            output_dir=self.temp_dir,
            upload_to_host=False,
        )

    # --- Тест 1: разбор размера из текста запроса ------------------------

    def test_parse_prompt_extracts_size(self):
        """Размер после вертикальной черты отделяется от описания."""
        text, width, height = self.tool._parse_prompt("рыжий кот | 512x640")

        self.assertEqual(text, "рыжий кот")
        self.assertEqual((width, height), (512, 640))

        # Кириллическая "х" вместо латинской "x" - модель пишет и так, и так.
        _, width, height = self.tool._parse_prompt("кот | 800х600")
        self.assertEqual((width, height), (800, 600))

        # Без указания размера берутся значения по умолчанию.
        text, width, height = self.tool._parse_prompt("кот в шляпе")
        self.assertEqual(text, "кот в шляпе")
        self.assertEqual((width, height), (self.tool.width, self.tool.height))

    # --- Тест 2: проверка описания и размеров ----------------------------

    def test_validation_rejects_bad_input(self):
        """Некорректные описание и размер отвергаются с понятной ошибкой."""
        # Описание: пустое, слишком длинное, не строка.
        for bad in ("   ", "а" * (MAX_PROMPT_LENGTH + 1), None):
            with self.assertRaises(ValueError, msg=repr(bad)[:40]):
                self.tool._parse_prompt(bad)

        # Размер: за пределами диапазона сверху и снизу, нечисловой.
        for bad in ((10000, 10000), (10, 10), ("широкая", 512)):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.tool._validate_size(*bad)

        # Допустимые значения проходят, лишние пробелы схлопываются.
        self.assertEqual(self.tool._validate_prompt("  рыжий   кот  "), "рыжий кот")
        self.assertEqual(self.tool._validate_size("512", "640"), (512, 640))

    # --- Тест 3: подбор соотношения сторон -------------------------------

    def test_to_aspect_ratio_picks_closest(self):
        """Размер в пикселях приводится к ближайшему допустимому соотношению."""
        cases = {(1024, 1024): "1:1", (1920, 1080): "16:9", (1024, 768): "4:3"}

        for (width, height), expected in cases.items():
            result = self.tool._to_aspect_ratio(width, height, REPLICATE_ASPECT_RATIOS)
            self.assertEqual(result, expected, f"{width}x{height}")

    # --- Тест 4: очередь провайдеров -------------------------------------

    def test_provider_chain_skips_providers_without_key(self):
        """Провайдеры без ключа исключаются, бесплатный остаётся последним."""
        self.tool.api_keys = {"replicate": None, "huggingface": "hf_test_token"}
        self.assertEqual(self.tool._provider_chain(), ["huggingface", "pollinations"])

        # Без ключа основной провайдер выпадает, работа не прекращается.
        self.tool.api_keys = {"replicate": None, "huggingface": None}
        self.assertEqual(self.tool._provider_chain(), ["pollinations"])

        # С выключенным fallback запасные не подставляются вообще.
        self.tool.fallback = False
        with self.assertRaises(ValueError):
            self.tool._provider_chain()

    # --- Тест 5: успешная генерация --------------------------------------

    def test_use_returns_link_to_generated_image(self):
        """При успешном ответе сервиса возвращается ссылка на сохранённый файл."""
        with patch("requests.post", return_value=make_response()) as post:
            result = self.tool.use("ginger cat in a hat | 512x512")

        self.assertIn("512x512", result)
        self.assertIn("huggingface", result)
        self.assertIn("Ссылка: file:///", result)

        # Размер и описание действительно ушли в запрос.
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["inputs"], "ginger cat in a hat")
        self.assertEqual(payload["parameters"]["width"], 512)

        # Файл появился на диске и содержит присланные байты.
        files = list(Path(self.temp_dir).glob("*.jpg"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].read_bytes(), FAKE_IMAGE)

    # --- Тест 6: разбор ответа Hugging Face ------------------------------

    def test_decode_huggingface_handles_broken_answer(self):
        """Картинка достаётся из base64, а повреждённый ответ даёт понятную ошибку."""
        encoded = base64.b64encode(FAKE_IMAGE).decode()
        payload = {"data": [{"b64_json": encoded}]}
        self.assertEqual(self.tool._decode_huggingface(payload), FAKE_IMAGE)

        for broken in ({}, {"data": []}, {"data": [{}]}, "не словарь"):
            with self.assertRaises(ValueError):
                self.tool._decode_huggingface(broken)

    # --- Тест 7: переключение на запасного провайдера --------------------

    def test_use_switches_provider_on_http_error(self):
        """Если основной провайдер ответил ошибкой, работает следующий в очереди."""
        with patch("requests.post", side_effect=make_http_error(402)), \
                patch("requests.get", return_value=make_response()):
            result = self.tool.use("ginger cat | 512x512")

        # Сработал бесплатный pollinations, ссылка ведёт на его адрес.
        self.assertIn("pollinations", result)
        self.assertIn("image.pollinations.ai", result)

    # --- Тест 8: ошибки не превращаются в исключения ---------------------

    def test_use_returns_message_instead_of_raising(self):
        """Любая ошибка возвращается строкой: агент не должен падать из-за картинки."""
        with patch("requests.post", side_effect=requests.exceptions.Timeout):
            self.assertIn("Ошибка", self.tool.use("cat"))

        # Некорректный ввод тоже не приводит к исключению.
        self.assertIn("Ошибка", self.tool.use(""))
        self.assertIn("Ошибка", self.tool.use("cat | 9999x9999"))

        # Сервис ответил страницей вместо картинки.
        html = make_response(content=b"<html>error</html>", content_type="text/html")
        with patch("requests.post", return_value=html):
            self.assertIn("Ошибка", self.tool.use("cat"))

    # --- Тест 9: сборка ссылки для pollinations --------------------------

    def test_build_url_encodes_prompt(self):
        """Пробелы и кириллица кодируются, размер уходит в параметры запроса."""
        url = self.tool._build_url("рыжий кот", 512, 640)

        self.assertTrue(url.startswith("https://image.pollinations.ai/prompt/"))
        self.assertNotIn(" ", url)
        self.assertIn("%D1%80%D1%8B%D0%B6%D0%B8%D0%B9", url)
        self.assertIn("width=512", url)
        self.assertIn("height=640", url)

    # --- Тест 10: загрузка на хостинг ------------------------------------

    def test_upload_falls_through_to_next_host(self):
        """Отказ хостинга не теряет картинку: пробуется следующий, затем диск."""
        self.tool.upload_to_host = True

        # Первый хостинг отказывает, второй отвечает в своём формате (uguu).
        answers = [
            make_http_error(412),
            make_response(json_data={"files": [{"url": FAKE_HOST_URL}]}),
        ]

        with patch("requests.post", side_effect=answers):
            url = self.tool._save_image(FAKE_IMAGE, "cat", "jpg")
        self.assertEqual(url, FAKE_HOST_URL)

        # Если отказали все хостинги, возвращается локальная ссылка на файл.
        with patch("requests.post", side_effect=make_http_error(500)):
            url = self.tool._save_image(FAKE_IMAGE, "cat", "jpg")
        self.assertTrue(url.startswith("file:///"))

    # --- Тест 11: проверка параметров при создании -----------------------

    def test_init_rejects_invalid_arguments(self):
        """Неизвестный провайдер и бессмысленные параметры отвергаются сразу."""
        for kwargs in ({"provider": "midjourney"}, {"timeout": 0},
                       {"timeout": "быстро"}, {"seed": -5}, {"width": 99999}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                ImageGeneratorTool(api_key="hf_test_token", **kwargs)


    # --- Тест 12: генерация через Replicate ------------------------------

    def test_replicate_generation(self):
        """Разбор ответа Replicate: сразу готов, готов после ожидания или отказ."""
        tool = ImageGeneratorTool(provider="replicate", api_key="r8_test",
                                  output_dir=self.temp_dir, fallback=False)

        # Готовый ответ: ссылка приходит списком.
        answer = {"status": "succeeded", "output": ["https://replicate.test/out.jpg"]}
        with patch("requests.post", return_value=make_response(json_data=answer)) as post:
            result = tool.use("ginger cat | 1920x1080")

        self.assertIn("https://replicate.test/out.jpg", result)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["input"]["aspect_ratio"], "16:9")
        self.assertIn("r8_test", post.call_args.kwargs["headers"]["Authorization"])

        # Долгая генерация: статус опрашивается до готовности.
        started = {"status": "processing",
                   "urls": {"get": "https://api.replicate.test/predictions/1"}}
        finished = {"status": "succeeded", "output": "https://replicate.test/late.jpg"}
        with patch("requests.post", return_value=make_response(json_data=started)), \
                patch("requests.get", return_value=make_response(json_data=finished)), \
                patch("time.sleep"):
            self.assertIn("late.jpg", tool.use("cat"))

        # Отказ, пустой результат и потерянная ссылка на статус.
        for answer in ({"status": "failed", "error": "NSFW content detected"},
                       {"status": "succeeded", "output": []},
                       {"status": "processing", "urls": {}}):
            with patch("requests.post", return_value=make_response(json_data=answer)):
                self.assertIn("Ошибка", tool.use("cat"))

    # --- Тест 13: второй формат API у Hugging Face -----------------------

    def test_huggingface_openai_route(self):
        """Сторонний исполнитель отвечает картинкой в base64, а не байтами."""
        tool = ImageGeneratorTool(provider="huggingface", api_key="hf_test_token",
                                  hf_route="nscale", output_dir=self.temp_dir,
                                  upload_to_host=False)
        encoded = base64.b64encode(FAKE_IMAGE).decode()
        answer = {"data": [{"b64_json": encoded}]}

        with patch("requests.post", return_value=make_response(json_data=answer)) as post:
            result = tool.use("ginger cat | 512x512")

        self.assertIn("Ссылка: file:///", result)

        # У этого маршрута размер передаётся строкой, а не парой чисел.
        self.assertEqual(post.call_args.kwargs["json"]["size"], "512x512")


def run_all_tests():
    """
    Отдельная тестовая функция, запускающая все юнит-тесты класса.

    Требование лабораторной работы: тесты должны вызываться внутри одной
    тестовой функции. Запуск через pytest при этом тоже работает.
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TestImageGeneratorTool)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


def test_run_all_tests():
    """Точка входа для pytest: прогоняет весь набор одной функцией."""
    assert run_all_tests()


if __name__ == "__main__":
    raise SystemExit(0 if run_all_tests() else 1)

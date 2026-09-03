# llm_agent/tool_image_generator.py

import base64
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlencode

import requests
from decouple import config

# Сервисы отдают готовую картинку байтами и не хранят её у себя, поэтому файл
# сначала сохраняется на диск, а публичная ссылка получается загрузкой на хостинг.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "generated_images"

# Запасной провайдер: Replicate. Требует положительный баланс на счету.
REPLICATE_API_URL = "https://api.replicate.com/v1/models/{model}/predictions"
REPLICATE_DEFAULT_MODEL = "black-forest-labs/flux-schnell"

# Основной провайдер: Hugging Face с моделью Stability AI. Бесплатный токен,
# точный размер в пикселях (платные сервисы принимают только пропорции).
HUGGINGFACE_API_URL = "https://router.huggingface.co/{route}/v1/images/generations"
HUGGINGFACE_DEFAULT_ROUTE = "hf-inference"
HUGGINGFACE_DEFAULT_MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"

# У Hugging Face два разных API. Собственный исполнитель hf-inference работает по
# старой схеме: модель в адресе, ответ - сразу байты картинки. Сторонние
# исполнители (nscale, fal-ai) отвечают по OpenAI-совместимой схеме с base64.
HUGGINGFACE_LEGACY_ROUTE = "hf-inference"
HUGGINGFACE_LEGACY_URL = "https://router.huggingface.co/hf-inference/models/{model}"

# Replicate не принимает произвольные width/height - только соотношение сторон
# из своего списка, поэтому размер приходится приводить к ближайшему значению.
REPLICATE_ASPECT_RATIOS = ("1:1", "16:9", "21:9", "3:2", "2:3", "4:5",
                           "5:4", "3:4", "4:3", "9:16", "9:21")

# Запасной провайдер: работает без ключа, нужен чтобы код запускался у любого,
# кто склонировал репозиторий и не заводил платные аккаунты.
POLLINATIONS_BASE_URL = "https://image.pollinations.ai/prompt"

# Чтобы выполнить требование "возвращать URL", сохранённый файл выкладывается на
# бесплатный хостинг. Внимание: загруженная картинка становится доступна любому в
# интернете; отключается параметром upload_to_host=False.
#
# Хостинги пробуются по очереди: любой из них может отказать, полагаться на один
# нельзя. Тонкость: litterbox отвечает "412 Precondition Failed" на стандартный
# User-Agent библиотеки requests, поэтому его приходится подменять.
UPLOAD_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ImageGeneratorTool/1.0"

IMAGE_HOSTS = (
    {
        "name": "litterbox",
        "url": "https://litterbox.catbox.moe/resources/internals/api.php",
        "field": "fileToUpload",
        "data": {"reqtype": "fileupload", "time": "72h"},
        "format": "text",
    },
    {
        "name": "uguu",
        "url": "https://uguu.se/upload",
        "field": "files[]",
        "data": {},
        "format": "uguu",
    },
    {
        "name": "tmpfiles",
        "url": "https://tmpfiles.org/api/v1/upload",
        "field": "file",
        "data": {},
        "format": "tmpfiles",
    },
)

# Из какой переменной окружения брать ключ для каждого провайдера с авторизацией.
PROVIDER_ENV_VARS = {
    "replicate": "REPLICATE_API_TOKEN",
    "huggingface": "HUGGINGFACE_API_TOKEN",
}

SUPPORTED_PROVIDERS = ("huggingface", "replicate", "pollinations")

# Порядок, в котором пробуются запасные провайдеры, если основной недоступен:
# сначала бесплатный по токену, затем совсем без ключа.
FALLBACK_ORDER = ("huggingface", "pollinations")

# Ограничения на размер картинки и длину описания.
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024

# Что модель рисовать не должна. Диффузионные модели плохо справляются с кистями
# рук - лишние и сросшиеся пальцы встречаются постоянно, - и явный запрет заметно
# улучшает результат. Работает только на маршруте hf-inference: OpenAI-совместимая
# схема сторонних исполнителей параметр negative_prompt не предусматривает.
DEFAULT_NEGATIVE_PROMPT = (
    "deformed hands, extra fingers, missing fingers, fused fingers, "
    "mutated hands, bad anatomy, extra limbs, malformed, blurry"
)

# Больше шагов - аккуратнее мелкие детали, в первую очередь пальцы.
DEFAULT_STEPS = 40
MIN_SIZE = 64
MAX_SIZE = 2048
MAX_PROMPT_LENGTH = 1000

# Пользователь может дописать размер в конце запроса: "рыжий кот в шляпе | 512x512".
# Допускаем и латинскую 'x', и кириллическую 'х', и звёздочку - модель пишет по-разному.
SIZE_PATTERN = re.compile(r"\s*\|\s*(\d+)\s*[xх*]\s*(\d+)\s*$", re.IGNORECASE)

# Для имени файла оставляем только безопасные символы.
SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9а-яА-ЯёЁ]+")


class ImageGeneratorTool:
    """Инструмент для генерации изображений по текстовому описанию. Возвращает URL картинки."""

    name = "image_generator"
    description = (
        "Генерирует изображение по текстовому описанию и возвращает ссылку (URL) на него. "
        "Используй, когда просят нарисовать, сгенерировать или создать картинку. "
        "Описание передавай на английском языке: модели генерации не понимают русский. "
        "Размер можно указать в конце через вертикальную черту, например: "
        "'watercolor painting of a ginger cat wearing a hat | 512x512'."
    )

    def __init__(self, provider: str = "huggingface", api_key: str = None,
                 model: str = REPLICATE_DEFAULT_MODEL,
                 hf_route: str = HUGGINGFACE_DEFAULT_ROUTE,
                 hf_model: str = HUGGINGFACE_DEFAULT_MODEL, width: int = DEFAULT_WIDTH,
                 height: int = DEFAULT_HEIGHT, timeout: int = 60, seed: int = None,
                 negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
                 steps: int = DEFAULT_STEPS,
                 output_dir=DEFAULT_OUTPUT_DIR, fallback: bool = True,
                 upload_to_host: bool = True):
        """
        Инициализирует инструмент генерации изображений.

        Args:
            provider (str): "huggingface" (по умолчанию), "replicate"
                            или "pollinations" (работает без ключа).
            api_key (str, optional): Ключ провайдера. Если не передан, берётся из
                                     переменной окружения (см. PROVIDER_ENV_VARS).
            model (str): Имя модели для Replicate.
            hf_route (str): Исполнитель, к которому роутер Hugging Face шлёт запрос.
            hf_model (str): Имя модели для Hugging Face.
            width (int): Ширина изображения по умолчанию.
            height (int): Высота изображения по умолчанию.
            timeout (int): Таймаут HTTP-запросов в секундах.
            seed (int, optional): Зерно генерации, чтобы результат был воспроизводимым.
            negative_prompt (str): Перечень того, чего на картинке быть не должно.
            steps (int): Число шагов генерации: больше - аккуратнее детали, но дольше.
            output_dir: Папка, куда сохраняются сгенерированные картинки.
            fallback (bool): Пробовать ли запасных провайдеров, если основной
                             ответил ошибкой (нет средств, неверный ключ, лимит).
            upload_to_host (bool): Выкладывать ли сохранённую картинку на
                             бесплатный хостинг, чтобы вернуть публичную
                             http-ссылку вместо локальной file://. Картинка при
                             этом становится доступна всем в интернете.

        Raises:
            ValueError: Если указан неизвестный провайдер или некорректный размер.
        """
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Неизвестный провайдер '{provider}'. Доступны: {', '.join(SUPPORTED_PROVIDERS)}"
            )
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"Таймаут должен быть положительным числом, получено: {timeout!r}")
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError(f"Зерно генерации должно быть целым неотрицательным числом: {seed!r}")

        # Ключи всех провайдеров читаем сразу: они нужны, чтобы решить, кого
        # вообще есть смысл пробовать в качестве запасного.
        self.api_keys = {name: config(env_var, default=None)
                         for name, env_var in PROVIDER_ENV_VARS.items()}

        if api_key:
            self.api_keys[provider] = api_key
        self.provider = provider
        self.model = model
        self.hf_route = hf_route
        self.hf_model = hf_model
        self.width, self.height = self._validate_size(width, height)
        self.timeout = timeout
        self.seed = seed
        self.negative_prompt = negative_prompt
        self.steps = steps
        self.output_dir = Path(output_dir)
        self.fallback = fallback
        self.upload_to_host = upload_to_host

    def use(self, prompt: str) -> str:
        """
        Генерирует изображение по описанию и возвращает строку со ссылкой на него.

        Args:
            prompt (str): Описание картинки, например "рыжий кот в шляпе | 512x512".

        Returns:
            str: Строка с URL изображения или понятное сообщение об ошибке.
        """
        try:
            text, width, height = self._parse_prompt(prompt)
            print(f"> Генерирую изображение ({width}x{height}) по описанию: '{text}'")

            chain = self._provider_chain()

            for position, provider in enumerate(chain):
                try:
                    url = self._generate(provider, text, width, height)
                except requests.exceptions.HTTPError as e:
                    # Кончились кредиты, протух ключ, превышен лимит - сам запрос
                    # при этом корректный, поэтому пробуем следующего провайдера.
                    if position == len(chain) - 1:
                        raise
                    print(f"> {provider} вернул ошибку ({e.response.status_code}), "
                          f"пробую следующего провайдера: {chain[position + 1]}.")
                    continue

                print(f"> Изображение готово: {url}")
                return self._format_result(text, url, width, height, provider)

        except ValueError as e:
            return f"Ошибка: {e}"
        except requests.exceptions.Timeout:
            return (
                f"Ошибка: сервис генерации изображений не ответил за {self.timeout} секунд. "
                "Попробуйте повторить запрос или уменьшить размер картинки."
            )
        except requests.exceptions.RequestException as e:
            return f"Ошибка при обращении к сервису генерации изображений: {e}"
        except OSError as e:
            # Нет прав на папку, кончилось место на диске и т.п.
            return f"Ошибка: не удалось сохранить изображение на диск ({e})."

    def _provider_chain(self) -> list:
        """
        Составляет список провайдеров, которых имеет смысл пробовать по очереди.

        Провайдеры без ключа отбрасываются - обращаться к ним бессмысленно.
        Если fallback выключен, остаётся только выбранный пользователем.

        Raises:
            ValueError: Если пробовать некого (нужен ключ, а его нет).
        """
        candidates = [self.provider]

        if self.fallback:
            candidates += [name for name in FALLBACK_ORDER if name != self.provider]

        chain = [name for name in candidates
                 if name not in PROVIDER_ENV_VARS or self.api_keys.get(name)]

        if not chain:
            raise ValueError(
                f"нет ключа для провайдера '{self.provider}' "
                f"(ожидается {PROVIDER_ENV_VARS[self.provider]}) и запасные недоступны."
            )

        return chain

    def _generate(self, provider: str, text: str, width: int, height: int) -> str:
        """Вызывает нужного провайдера и возвращает ссылку на картинку."""
        if provider == "replicate":
            return self._generate_via_replicate(text, width, height)
        if provider == "huggingface":
            return self._generate_via_huggingface(text, width, height)

        url = self._build_url(text, width, height)
        self._check_url_available(url)
        return url

    def _parse_prompt(self, prompt: str) -> tuple:
        """
        Разбирает пользовательский запрос на описание и размер изображения.

        Args:
            prompt (str): Исходная строка запроса.

        Returns:
            tuple: (описание, ширина, высота).
        """
        if not isinstance(prompt, str):
            raise ValueError("описание изображения должно быть строкой.")

        width, height = self.width, self.height
        match = SIZE_PATTERN.search(prompt)

        if match:
            width, height = self._validate_size(int(match.group(1)), int(match.group(2)))
            prompt = prompt[:match.start()]

        return self._validate_prompt(prompt), width, height

    def _validate_prompt(self, text: str) -> str:
        """
        Проверяет описание изображения и возвращает его очищенную версию.

        Raises:
            ValueError: Если описание пустое или слишком длинное.
        """
        text = " ".join(text.split())

        if not text:
            raise ValueError("описание изображения не может быть пустым.")
        if len(text) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"описание слишком длинное ({len(text)} символов), "
                f"максимум - {MAX_PROMPT_LENGTH}."
            )

        return text

    def _validate_size(self, width, height) -> tuple:
        """
        Проверяет, что размеры картинки - целые числа в допустимом диапазоне.

        Raises:
            ValueError: Если размер не является числом или выходит за границы.
        """
        try:
            width, height = int(width), int(height)
        except (TypeError, ValueError):
            raise ValueError("размеры изображения должны быть целыми числами.")

        for value in (width, height):
            if not MIN_SIZE <= value <= MAX_SIZE:
                raise ValueError(
                    f"размер {value} вне допустимого диапазона "
                    f"{MIN_SIZE}-{MAX_SIZE} пикселей."
                )

        return width, height

    def _to_aspect_ratio(self, width: int, height: int, ratios: tuple) -> str:
        """
        Переводит размер в пикселях в ближайшее допустимое соотношение сторон.

        Провайдеры принимают только значения из своего списка, поэтому выбираем
        то, чья пропорция ближе всего к запрошенной.
        """
        target = width / height

        def distance(ratio: str) -> float:
            w, h = ratio.split(":")
            return abs(int(w) / int(h) - target)

        return min(ratios, key=distance)

    def _ensure_image_response(self, response, service: str) -> None:
        """
        Проверяет, что сервис прислал именно картинку, а не HTML-страницу с ошибкой.

        Код 200 сам по себе этого не гарантирует: некоторые сервисы отдают
        страницу "попробуйте позже" с успешным статусом.

        Raises:
            ValueError: Если тип содержимого не image/*.
        """
        content_type = response.headers.get("Content-Type", "")

        if not content_type.startswith("image/"):
            raise ValueError(
                f"{service} вернул не изображение, а '{content_type or 'неизвестный тип'}'."
            )

    def _save_image(self, content: bytes, text: str, extension: str) -> str:
        """
        Сохраняет картинку в папку output_dir и возвращает file://-ссылку на неё.

        Raises:
            ValueError: Если сервис вернул пустой ответ.
        """
        if not content:
            raise ValueError("сервис вернул пустое изображение.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        slug = SLUG_PATTERN.sub("_", text).strip("_")[:40] or "image"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"{slug}_{timestamp}.{extension}"
        path.write_bytes(content)

        if self.upload_to_host:
            return self._upload_to_host(path, content)

        return path.as_uri()

    def _upload_to_host(self, path: Path, content: bytes) -> str:
        """
        Выкладывает картинку на бесплатный хостинг и возвращает публичную ссылку.

        Хостинги пробуются по очереди, пока какой-нибудь не примет файл. Если не
        принял никто, возвращаем локальную ссылку: терять уже сгенерированную
        картинку из-за недоступности стороннего сервиса не стоит.
        """
        for host in IMAGE_HOSTS:
            try:
                response = requests.post(
                    host["url"],
                    data=host["data"],
                    files={host["field"]: (path.name, content)},
                    headers={"User-Agent": UPLOAD_USER_AGENT},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return self._parse_host_response(host, response)

            except (requests.exceptions.RequestException, ValueError, KeyError):
                # Хостинг мог отказать по своим причинам - молча пробуем следующий.
                continue

        print("> Ни один хостинг не доступен, возвращаю локальную ссылку.")
        return path.as_uri()

    def _parse_host_response(self, host: dict, response) -> str:
        """
        Достаёт ссылку из ответа хостинга: у каждого свой формат.

        Raises:
            ValueError: Если ссылки в ответе нет или она выглядит некорректно.
        """
        if host["format"] == "text":
            url = response.text.strip()
        elif host["format"] == "uguu":
            files = (response.json() or {}).get("files") or []
            url = files[0].get("url", "") if files else ""
        else:
            url = ((response.json() or {}).get("data") or {}).get("url", "")
            # tmpfiles отдаёт ссылку на страницу просмотра, прямая - через /dl/.
            url = url.replace("tmpfiles.org/", "tmpfiles.org/dl/", 1)

        if not url.startswith("http"):
            raise ValueError(f"неожиданный ответ: {str(url or response.text)[:80]}")

        return url

    def _generate_via_huggingface(self, text: str, width: int, height: int) -> str:
        """
        Генерирует изображение через роутер Hugging Face и возвращает ссылку на файл.

        Единственный из провайдеров, который принимает точный размер, а не
        соотношение сторон. Картинка приходит в base64 внутри JSON.
        """
        headers = {
            "Authorization": f"Bearer {self.api_keys['huggingface']}",
            "Content-Type": "application/json",
        }

        if self.hf_route == HUGGINGFACE_LEGACY_ROUTE:
            return self._generate_via_huggingface_legacy(headers, text, width, height)

        payload = {
            "prompt": text,
            "model": self.hf_model,
            "response_format": "b64_json",
            "size": f"{width}x{height}",
        }

        if self.seed is not None:
            payload["seed"] = self.seed

        response = requests.post(
            HUGGINGFACE_API_URL.format(route=self.hf_route),
            json=payload, headers=headers, timeout=self.timeout,
        )
        response.raise_for_status()

        return self._save_image(self._decode_huggingface(response.json()), text, "png")

    def _generate_via_huggingface_legacy(self, headers: dict, text: str,
                                        width: int, height: int) -> str:
        """
        Обращается к собственному исполнителю Hugging Face (hf-inference).

        Здесь модель указывается в адресе, размер уходит в parameters, а ответ
        приходит готовыми байтами картинки, без JSON и base64.
        """
        payload = {
            "inputs": text,
            "parameters": {"width": width, "height": height},
        }

        if self.negative_prompt:
            payload["parameters"]["negative_prompt"] = self.negative_prompt
        if self.steps:
            payload["parameters"]["num_inference_steps"] = self.steps
        if self.seed is not None:
            payload["parameters"]["seed"] = self.seed

        response = requests.post(
            HUGGINGFACE_LEGACY_URL.format(model=self.hf_model),
            json=payload, headers=headers, timeout=self.timeout,
        )
        response.raise_for_status()
        self._ensure_image_response(response, "Hugging Face")

        return self._save_image(response.content, text, "jpg")

    def _decode_huggingface(self, payload: dict) -> bytes:
        """
        Достаёт байты картинки из ответа Hugging Face.

        Raises:
            ValueError: Если в ответе нет изображения или base64 повреждён.
        """
        items = payload.get("data") if isinstance(payload, dict) else None

        if not items or not isinstance(items[0], dict) or not items[0].get("b64_json"):
            raise ValueError("Hugging Face вернул ответ без изображения.")

        try:
            return base64.b64decode(items[0]["b64_json"], validate=True)
        except (ValueError, TypeError) as e:
            raise ValueError(f"не удалось декодировать картинку от Hugging Face: {e}")

    def _generate_via_replicate(self, text: str, width: int, height: int) -> str:
        """
        Генерирует изображение через Replicate API и возвращает URL результата.

        Заголовок 'Prefer: wait' просит сервер дождаться готовности картинки,
        но при долгой генерации ответ всё равно может вернуться незавершённым -
        тогда дожидаемся результата опросом.
        """
        headers = {
            "Authorization": f"Bearer {self.api_keys['replicate']}",
            "Content-Type": "application/json",
            "Prefer": "wait",
        }
        payload = {
            "input": {
                "prompt": text,
                "aspect_ratio": self._to_aspect_ratio(width, height, REPLICATE_ASPECT_RATIOS),
                "num_outputs": 1,
                "output_format": "jpg",
            }
        }

        if self.seed is not None:
            payload["input"]["seed"] = self.seed

        response = requests.post(
            REPLICATE_API_URL.format(model=self.model),
            json=payload, headers=headers, timeout=self.timeout,
        )
        response.raise_for_status()
        prediction = response.json()

        if prediction.get("status") not in ("succeeded", "failed", "canceled"):
            prediction = self._poll_replicate(prediction, headers)

        return self._extract_replicate_url(prediction)

    def _poll_replicate(self, prediction: dict, headers: dict) -> dict:
        """Периодически опрашивает Replicate, пока генерация не завершится."""
        poll_url = (prediction.get("urls") or {}).get("get")

        if not poll_url:
            raise ValueError("Replicate не вернул ссылку для проверки статуса генерации.")

        deadline = time.time() + self.timeout

        while time.time() < deadline:
            time.sleep(2)
            response = requests.get(poll_url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            prediction = response.json()

            if prediction.get("status") in ("succeeded", "failed", "canceled"):
                return prediction

        raise ValueError(f"Replicate не завершил генерацию за {self.timeout} секунд.")

    def _extract_replicate_url(self, prediction: dict) -> str:
        """Достаёт ссылку на картинку из ответа Replicate."""
        if prediction.get("status") != "succeeded":
            detail = prediction.get("error") or prediction.get("status")
            raise ValueError(f"Replicate не смог сгенерировать изображение ({detail}).")

        output = prediction.get("output")

        # Разные модели возвращают либо одну ссылку, либо список ссылок.
        if isinstance(output, list):
            output = output[0] if output else None
        if not output:
            raise ValueError("Replicate вернул пустой результат генерации.")

        return output

    def _build_url(self, text: str, width: int, height: int) -> str:
        """
        Собирает URL картинки для сервиса Pollinations.

        Описание уходит в путь запроса, поэтому его нужно закодировать:
        иначе пробелы и кириллица сломают ссылку.
        """
        params = {"width": width, "height": height, "nologo": "true"}

        if self.seed is not None:
            params["seed"] = self.seed

        return f"{POLLINATIONS_BASE_URL}/{quote(text, safe='')}?{urlencode(params)}"

    def _check_url_available(self, url: str) -> bool:
        """
        Убеждается, что по ссылке действительно отдаётся изображение.

        Сервис генерирует картинку в момент первого обращения, поэтому запрос
        одновременно и запускает генерацию, и проверяет её результат. Тело ответа
        не выкачиваем - нам нужен только URL.
        """
        response = requests.get(url, timeout=self.timeout, stream=True)
        response.close()
        response.raise_for_status()
        self._ensure_image_response(response, "Pollinations")
        return True

    def _format_result(self, text: str, url: str, width: int, height: int,
                       provider: str) -> str:
        """Готовит итоговый ответ для агента."""
        return (
            f"Изображение по запросу '{text}' сгенерировано "
            f"({width}x{height}, провайдер {provider}).\n"
            f"Ссылка: {url}"
        )

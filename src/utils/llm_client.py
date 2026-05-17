"""
Unified LLM client — supports DeepSeek (text-only) + VLM backends
for multimodal reflection (GPT-4o / Gemini / Ollama Qwen-VL).

Key design:
  LLMClient  → text reasoning (DeepSeek-V4-Pro, no vision)
  VLMClient  → multimodal reflection (GPT-4o etc., real image understanding)
"""
import os
import time
import base64
import logging
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from openai import OpenAI, APIError, RateLimitError

logger = logging.getLogger(__name__)


# =============================================================================
# Helper: resolve API key from config value or environment variable
# =============================================================================

def _resolve_api_key(value: str) -> str:
    """
    If `value` looks like an actual API key (e.g. starts with 'sk-'), use it directly.
    Otherwise treat it as an environment variable name and look it up via os.getenv.
    Allows both patterns in config.yaml:
        api_key_env: "DEEPSEEK_API_KEY"    # env var name (old pattern)
        api_key_env: "sk-abc123..."        # literal key (new, more convenient)
    """
    if not value:
        return ""
    # Looks like a literal key: common prefixes + reasonable length, no spaces
    if value.startswith(("sk-", "sk-proj-", "AKIA", "d-")) or (
        len(value) > 20 and " " not in value
    ):
        return value
    # Otherwise, treat as env var name
    return os.getenv(value, "")


# =============================================================================
# Config
# =============================================================================

@dataclass
class LLMConfig:
    provider: str = "deepseek"
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    max_tokens: int = 2048
    temperature: float = 0.0
    max_retries: int = 3
    retry_delay: float = 2.0


@dataclass
class VLMConfig:
    """Configuration for a vision-capable model (used in reflection)."""
    provider: str = "openai"           # openai | gemini | ollama
    api_key_env: str = "OPENAI_API_KEY"  # env var name, or actual key if starts with sk-
    api_key: str = ""                    # direct key (takes priority over env var)
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    max_tokens: int = 600
    temperature: float = 0.0
    max_retries: int = 3
    retry_delay: float = 2.0


# =============================================================================
# LLMClient — text-only reasoning (DeepSeek)
# =============================================================================

class LLMClient:
    """
    Unified LLM client. DeepSeek-V4-Pro is text-only:
    images are described in text, NOT sent as pixels.

    For vision tasks (reflection over actual CXR images), use VLMClient.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = _resolve_api_key(config.api_key_env)

        self.client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            max_retries=0,
        )
        logger.info(f"LLM ready: {config.provider} @ {config.model}")

    def _chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        model = model or self.config.model
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                return content.strip() if content else None

            except RateLimitError:
                wait = self.config.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limited. Retrying in {wait:.1f}s (attempt {attempt+1})...")
                time.sleep(wait)

            except APIError as e:
                logger.error(f"API error (attempt {attempt+1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    return None

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return None

        return None

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self._chat(messages, model=model, max_tokens=max_tokens)

    def chat_with_images(
        self,
        system_prompt: str,
        user_text: str,
        image_paths: Optional[List[str]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """
        DeepSeek is text-only, so images are referenced by filename in the prompt.
        The case_descriptor in the benchmark already encodes key radiological findings.
        For real image analysis, use VLMClient instead.
        """
        if image_paths:
            img_names = [os.path.basename(p) for p in image_paths]
            user_text += (
                f"\n\n[Referenced CXR image(s): {', '.join(img_names)}. "
                f"Use the case descriptor and clinical question above for diagnosis.]"
            )
        return self.chat(system_prompt, user_text, model=model, max_tokens=max_tokens)


# =============================================================================
# VLMClient — vision-capable, used for multimodal reflection
# =============================================================================

class VLMClient:
    """
    Vision-Language Model client for multimodal reflection.

    Unlike LLMClient (text-only DeepSeek), this actually sends images
    to a vision-capable model (GPT-4o / Gemini / Ollama Qwen-VL).

    This is the KEY FIX for the "VLM absence" problem:
    after receiving ground-truth feedback, the agent must reflect
    on the *actual CXR images* + question + prediction + trace + truth.
    """

    # Supported image MIME types per provider
    _MIME_MAP = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }

    def __init__(self, config: VLMConfig):
        self.config = config
        self.provider = config.provider

        if config.provider == "gemini":
            # Gemini uses its own SDK, not OpenAI-compatible
            self._init_gemini(config)
        else:
            # OpenAI / Ollama both use OpenAI-compatible API
            api_key = _resolve_api_key(config.api_key_env) or "ollama"  # Ollama ignores key
            self.client = OpenAI(
                api_key=api_key,
                base_url=config.base_url if config.base_url else None,
                max_retries=0,
            )
            self._use_sdk = False

        logger.info(f"VLM ready: {config.provider} @ {config.model}")

    def _init_gemini(self, config: VLMConfig):
        """Initialize Google Gemini SDK client."""
        try:
            import google.generativeai as genai
            api_key = _resolve_api_key(config.api_key_env)
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(config.model)
            self._use_sdk = True
        except ImportError:
            logger.error(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )
            raise

    def chat_with_images(
        self,
        system_prompt: str,
        user_text: str,
        image_paths: Optional[List[str]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """
        Send text + images to VLM and get response.
        This is the REAL multimodal call that LLMClient cannot do.
        """
        model = model or self.config.model
        max_tokens = max_tokens or self.config.max_tokens

        if self._use_sdk and self.provider == "gemini":
            return self._chat_gemini(system_prompt, user_text, image_paths)

        # OpenAI-compatible path (GPT-4o / Ollama Qwen-VL)
        return self._chat_openai_compatible(
            system_prompt, user_text, image_paths, model, max_tokens
        )

    def _chat_openai_compatible(
        self,
        system_prompt: str,
        user_text: str,
        image_paths: Optional[List[str]],
        model: str,
        max_tokens: int,
    ) -> Optional[str]:
        """OpenAI-compatible vision API call."""
        content: list = [{"type": "text", "text": user_text}]

        # Encode images as base64 data URIs
        if image_paths:
            for img_path in image_paths:
                if not os.path.exists(img_path):
                    logger.warning(f"Image not found: {img_path}")
                    continue
                try:
                    b64 = self._encode_image(img_path)
                    ext = os.path.splitext(img_path)[1].lower()
                    mime = self._MIME_MAP.get(ext, "image/png")
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    })
                except Exception as e:
                    logger.warning(f"Failed to encode {img_path}: {e}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self.config.temperature,
                )
                content = response.choices[0].message.content
                return content.strip() if content else None

            except RateLimitError:
                wait = self.config.retry_delay * (2 ** attempt)
                logger.warning(f"VLM rate limited. Retrying in {wait:.1f}s...")
                time.sleep(wait)

            except APIError as e:
                logger.error(f"VLM API error (attempt {attempt+1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    return None

            except Exception as e:
                logger.error(f"VLM unexpected error: {e}")
                return None

        return None

    def _chat_gemini(
        self,
        system_prompt: str,
        user_text: str,
        image_paths: Optional[List[str]],
    ) -> Optional[str]:
        """Gemini-specific vision API call."""
        try:
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
        except ImportError:
            logger.error("google-generativeai not available")
            return None

        parts = [system_prompt + "\n\n" + user_text]

        if image_paths:
            from PIL import Image
            for img_path in image_paths:
                if not os.path.exists(img_path):
                    continue
                try:
                    parts.append(Image.open(img_path))
                except Exception as e:
                    logger.warning(f"Failed to load {img_path}: {e}")

        safety = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        try:
            response = self._gemini_model.generate_content(
                parts,
                safety_settings=safety,
            )
            return response.text.strip() if response.text else None
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None

    def _encode_image(self, path: str) -> str:
        """Read image file and return base64-encoded string."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


# =============================================================================
# Factory
# =============================================================================

def create_vlm_client(vlm_config: Optional[dict]) -> Optional[VLMClient]:
    """Create VLMClient from config dict, or None if not configured."""
    if not vlm_config or not vlm_config.get("model"):
        logger.warning("No VLM configured. Reflection will be text-only (no image analysis).")
        return None

    cfg = VLMConfig(
        provider=vlm_config.get("provider", "openai"),
        api_key_env=vlm_config.get("api_key_env", "OPENAI_API_KEY"),
        base_url=vlm_config.get("base_url", "https://api.openai.com/v1"),
        model=vlm_config["model"],
        max_tokens=vlm_config.get("max_tokens", 600),
        temperature=vlm_config.get("temperature", 0.0),
    )
    return VLMClient(cfg)

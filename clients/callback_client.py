# clients/callback_client.py
# ─────────────────────────────────────────────────────────────
# HTTP client untuk callback hasil analisis ke Node.js.
# Callback URL dan header diambil dari payload request
# (dinamis — bukan hardcode di .env).
#
# Dilengkapi retry otomatis jika callback gagal.
# ─────────────────────────────────────────────────────────────

import asyncio
import logging
import httpx

from config import get_settings

logger   = logging.getLogger("fradara.callback")
settings = get_settings()


async def send_callback(
    callback_url    : str,
    callback_headers: dict,
    payload         : dict,
) -> bool:
    """
    Kirim hasil analisis ke callback URL dari Node.js.

    Parameter:
        callback_url     : dari payload.callbackUrl
        callback_headers : dari payload.callbackHeaders
        payload          : BatchCallbackPayload dalam bentuk dict

    Return True jika berhasil, False jika semua retry gagal.
    """
    # Gabung header default Python + header dinamis dari Node.js
    headers = {
        "Content-Type": "application/json",
        "x-service"   : "fradara-python",
        **callback_headers,
    }

    for attempt in range(1, settings.HTTP_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
                resp = await client.post(
                    callback_url,
                    json    = payload,
                    headers = headers,
                )
                resp.raise_for_status()
                logger.info(
                    "Callback OK → %s | status=%d | attempt=%d/%d",
                    callback_url, resp.status_code,
                    attempt, settings.HTTP_MAX_RETRIES,
                )
                return True

        except httpx.HTTPStatusError as e:
            logger.warning(
                "Callback HTTP error → %s | status=%d | attempt=%d/%d",
                callback_url, e.response.status_code,
                attempt, settings.HTTP_MAX_RETRIES,
            )
        except httpx.RequestError as e:
            logger.warning(
                "Callback request error → %s | %s | attempt=%d/%d",
                callback_url, str(e),
                attempt, settings.HTTP_MAX_RETRIES,
            )

        if attempt < settings.HTTP_MAX_RETRIES:
            logger.info("Retry dalam %.1f detik...", settings.HTTP_RETRY_DELAY)
            await asyncio.sleep(settings.HTTP_RETRY_DELAY)

    logger.error(
        "Callback GAGAL setelah %d attempt → %s",
        settings.HTTP_MAX_RETRIES, callback_url,
    )
    return False
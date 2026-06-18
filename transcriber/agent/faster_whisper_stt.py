"""
FasterWhisper STT Plugin for LiveKit Agents
===========================================

Local speech-to-text using faster-whisper with GPU acceleration.
No cloud APIs required - runs entirely on your hardware.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

import numpy as np
import sys
import os
import asyncio

# Ensure NVIDIA CUDA DLLs are added to the DLL search path on Windows
if sys.platform == "win32":
    try:
        import nvidia.cublas
        import nvidia.cudnn
        import nvidia.cuda_runtime
        
        paths = [
            os.path.join(list(nvidia.cublas.__path__)[0], "bin"),
            os.path.join(list(nvidia.cudnn.__path__)[0], "bin"),
            os.path.join(list(nvidia.cuda_runtime.__path__)[0], "bin"),
        ]
        
        for p in paths:
            if os.path.isdir(p):
                os.add_dll_directory(p)
                
        # Also add to PATH so C++ libraries loaded by extensions can find them
        os.environ["PATH"] = ";".join(paths) + ";" + os.environ["PATH"]
    except (ImportError, IndexError, AttributeError):
        pass

from faster_whisper import WhisperModel

from livekit import rtc
from livekit.agents import stt, APIConnectOptions, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr

__all__ = ["FasterWhisperSTT"]

logger = logging.getLogger(__name__)

# Type aliases for better IDE support
ModelSize = Literal["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v2", "large-v3"]
Device = Literal["cuda", "cpu", "auto"]
ComputeType = Literal["float16", "float32", "int8", "int8_float16", "int8_float32"]


class FasterWhisperSTT(stt.STT):
    """
    LiveKit STT plugin using FasterWhisper for local speech recognition.

    This plugin integrates the faster-whisper library with LiveKit's Agents
    framework, enabling fully local speech-to-text without cloud dependencies.
    """

    # Process-wide lock shared across ALL instances (and any per-session wrappers
    # the framework creates). Guarantees only one CTranslate2/CUDA inference runs
    # at a time, preventing cuBLAS conflicts when multiple participants speak
    # simultaneously.
    #
    # Initialised eagerly at class-definition time. asyncio.Lock in Python 3.10+
    # does NOT bind to the event loop at construction — it binds on first await,
    # so this is safe to create here without a running event loop.
    _process_inference_lock: asyncio.Lock = asyncio.Lock()

    def __init__(
        self,
        model_size: ModelSize = "medium",
        device: Device = "cuda",
        compute_type: ComputeType = "float16",
        language: str = "en",
        beam_size: int = 5,
        vad_filter: bool = False,
        local_files_only: bool = True,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False
            )
        )

        self._language = language
        self._beam_size = beam_size
        self._vad_filter = vad_filter

        logger.info(f"Loading FasterWhisper model: {model_size} on {device} ({compute_type})")

        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            local_files_only=local_files_only,
        )

        logger.info(f"FasterWhisper ready - language={language}, beam_size={beam_size}")


    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions
    ) -> stt.SpeechEvent:
        """
        Process audio buffer and return transcription.

        Handles both single AudioFrame and lists of AudioFrames from LiveKit.
        Audio is normalized to float32 [-1, 1] range for Whisper processing.
        """
        # Resample to 16000Hz mono if needed (Whisper model requirements)
        target_sample_rate = 16000
        target_channels = 1

        frames = buffer if isinstance(buffer, list) else [buffer]
        resampled_frames = []
        resampler = None
        for frame in frames:
            if frame.sample_rate != target_sample_rate or frame.num_channels != target_channels:
                if resampler is None:
                    resampler = rtc.AudioResampler(
                        input_rate=frame.sample_rate,
                        output_rate=target_sample_rate,
                        num_channels=target_channels,
                        quality=rtc.AudioResamplerQuality.HIGH,
                    )
                resampled_frames.extend(resampler.push(frame))
            else:
                resampled_frames.append(frame)
        if resampler is not None:
            resampled_frames.extend(resampler.flush())

        # Convert resampled frames to numpy array
        all_data = []
        for frame in resampled_frames:
            frame_data = np.frombuffer(frame.data, dtype=np.int16)
            all_data.append(frame_data)
        
        if all_data:
            audio_data = np.concatenate(all_data).astype(np.float32) / 32768.0
        else:
            audio_data = np.empty(0, dtype=np.float32)

        # Use provided language or fall back to configured default
        lang = language if language is not NOT_GIVEN else self._language

        # Run transcription with optimized settings offloaded to a worker thread
        # to avoid blocking the single-threaded asyncio event loop.
        start_time = time.perf_counter()
        
        def run_inference():
            segments_generator, info = self._model.transcribe(
                audio_data,
                beam_size=self._beam_size,
                best_of=self._beam_size,
                temperature=0.0,  # Greedy decoding for consistency
                vad_filter=self._vad_filter,
                language=lang,
            )
            # Evaluate the generator inside the thread to do the heavy computation
            return list(segments_generator), info

        async with FasterWhisperSTT._process_inference_lock:
            segments, info = await asyncio.to_thread(run_inference)
        
        # Combine all segments into final text
        text = "".join(segment.text for segment in segments).strip()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if text:
            logger.debug(f"Transcribed ({info.language}, {info.duration:.1f}s): {text}")

        logger.debug(f"STT latency: {elapsed_ms:.0f}ms for {info.duration:.1f}s audio")

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(
                text=text,
                start_time=0,
                end_time=0,
                language=lang or ""
            )],
        )

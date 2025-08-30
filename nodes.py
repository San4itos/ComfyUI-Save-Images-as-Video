# nodes.py
import subprocess
import os
import shlex
import tempfile
import numpy as np
from PIL import Image
import folder_paths
import torch
import torchaudio
from .ffmpeg_path_resolver import get_ffmpeg_path
from .node_logger import log_node_info, log_node_success, log_node_error, log_node_warning, log_node_debug 

class FFmpegConverterBase:
    """Базовий клас, що містить спільну логіку для роботи з FFmpeg."""
    
    def _build_ffmpeg_params(self, base_params, override_str, log_prefix):
        """
        Формує фінальний список параметрів для FFmpeg, об'єднуючи базові 
        параметри з GUI та параметри, введені користувачем.
        Параметри користувача мають вищий пріоритет.
        """
        # Використовуємо shlex для безпечного парсингу рядка
        try:
            override_args = shlex.split(override_str)
        except ValueError as e:
            log_node_error(log_prefix, f"Error parsing override parameters: {e}. Ignoring overrides.")
            override_args = []

        final_params = base_params.copy()
        
        # Ітеруємо по спарсених аргументах. Очікуємо пари: флаг, значення.
        it = iter(override_args)
        for flag in it:
            if flag.startswith('-'):
                try:
                    # Наступний елемент - це значення для нашого флага
                    value = next(it)
                    if flag in final_params:
                        log_node_warning(log_prefix, f"Overriding GUI parameter '{flag}' with value '{value}' (was '{final_params[flag]}').")
                    final_params[flag] = value
                except StopIteration:
                    log_node_warning(log_prefix, f"Flag '{flag}' provided without a value. Ignoring.")
            else:
                log_node_warning(log_prefix, f"Unexpected token '{flag}' in override parameters. Should be a flag starting with '-'. Ignoring.")

        # Перетворюємо словник у список для subprocess
        result_list = []
        for key, val in final_params.items():
            result_list.extend([key, str(val)])
            
        return result_list

    def _execute_ffmpeg_command(self, ffmpeg_cmd, log_prefix):
        log_node_info(log_prefix, f"Executing ffmpeg: {' '.join(ffmpeg_cmd)}")
        try:
            process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
            stdout, stderr = process.communicate(timeout=300)
            if process.returncode != 0:
                err_msg = f"ffmpeg error (code {process.returncode}):\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                log_node_error(log_prefix, err_msg)
                return {"ui": {"text": [f"ffmpeg error (code {process.returncode}): Check console for details."]}}
            else:
                log_node_success(log_prefix, "FFmpeg command executed successfully.")
                if stderr.strip():
                    log_node_warning(log_prefix, f"ffmpeg stderr (warnings):\n{stderr}", msg_color_override="GREY")
                return None # Успіх
        except Exception as e:
            log_node_error(log_prefix, f"Python error during ffmpeg execution: {e}")
            return {"ui": {"text": [f"Python error: {e}"]}}


class SaveFramesToVideoFFmpeg(FFmpegConverterBase):
    NODE_LOG_PREFIX = "SaveVideoFFMPEG"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.ffmpeg_executable_path = get_ffmpeg_path()

    @classmethod
    def INPUT_TYPES(cls):
           return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "VID"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "codec": (["libx264", "libx265", "libvpx-vp9", "libsvtav1"], {"default": "libx264"}),
                "pixel_format": (["yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le", "yuv444p10le", "rgb24"], {"default": "yuv420p"}),
                "crf": ("INT", {"default": 23, "min": 0, "max": 63, "step": 1}),
                "output_format": (["mp4", "webm", "mov", "avi", "mkv"], {"default": "mp4"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "audio_codec": (["aac", "mp3", "libopus", "copy"], {"default": "aac"}),
                "audio_bitrate": (["96k", "128k", "160k", "192k", "256k", "320k"], {"default": "192k"}),
                "output_file_opt": ("STRING", {"multiline": True, "default": "-preset medium", "tooltip": "Custom FFmpeg output options. One option per line, e.g., -preset slow"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "San4itos"

    def save_video(self, images, filename_prefix, fps, codec, pixel_format, crf, output_format, 
                   audio=None, audio_codec="aac", audio_bitrate="192k", output_file_opt=""):
        
        h, w = images[0].shape[0], images[0].shape[1]
        full_output_folder, filename_part, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, self.output_dir, w, h)
        video_filename = f"{filename_part}_{counter:05}_.{output_format}"
        video_full_path = os.path.join(full_output_folder, video_filename)

        with tempfile.TemporaryDirectory() as temp_dir:
            for i, image_tensor in enumerate(images):
                img_pil = Image.fromarray((image_tensor.cpu().numpy() * 255).astype(np.uint8))
                img_pil.save(os.path.join(temp_dir, f"frame_{i:06d}.png"), "PNG")

            ffmpeg_cmd = [self.ffmpeg_executable_path, '-y', '-framerate', str(fps), '-i', os.path.join(temp_dir, 'frame_%06d.png')]
            has_audio = audio and "waveform" in audio and audio["waveform"].numel() > 0

            if has_audio:
                temp_audio_file = os.path.join(temp_dir, "temp_audio.wav")
                waveform_tensor = audio["waveform"]
                if waveform_tensor.shape[0] > 1:
                    log_node_warning(self.NODE_LOG_PREFIX, f"Audio batch size is {waveform_tensor.shape[0]}. Using the first audio track.")
                torchaudio.save(temp_audio_file, waveform_tensor[0].cpu(), audio["sample_rate"])
                ffmpeg_cmd.extend(['-i', temp_audio_file])

            base_params = {
                '-c:v': codec,
                '-pix_fmt': pixel_format,
                '-crf': crf
            }
            
            final_params = self._build_ffmpeg_params(base_params, output_file_opt, self.NODE_LOG_PREFIX)
            ffmpeg_cmd.extend(final_params)

            if has_audio:
                ffmpeg_cmd.extend(['-c:a', audio_codec, '-b:a', audio_bitrate, '-shortest'])
            else:
                ffmpeg_cmd.extend(['-an'])
            
            ffmpeg_cmd.append(video_full_path)

            error = self._execute_ffmpeg_command(ffmpeg_cmd, self.NODE_LOG_PREFIX)
            if error: return error

            preview = [{"filename": video_filename, "subfolder": subfolder, "type": self.type}]
            return {"ui": {"videos": preview}}


class VideoPathWrapper:
    def __init__(self, filepath):
        self.filepath = filepath
        self._is_direct_path = True

    def get_components(self):
        from comfy.comfy_types import InputImpl
        return InputImpl.VideoFromFile(self.filepath).get_components()


class LoadVideoByPath_san4itos:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))], ["video"])
        return {"required": {"video_file": (sorted(files), {"video_upload": True})}}

    CATEGORY = "San4itos"
    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "load_video"

    def load_video(self, video_file):
        video_path = folder_paths.get_annotated_filepath(video_file)
        return (VideoPathWrapper(video_path),)


class ConvertVideoFFmpeg(FFmpegConverterBase):
    NODE_LOG_PREFIX = "ConvertVideoFFMPEG"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.ffmpeg_executable_path = get_ffmpeg_path()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "filename_prefix": ("STRING", {"default": "VID_conv"}),
                "codec": (["libx264", "libx265", "libvpx-vp9", "libsvtav1", "copy"], {"default": "libx264"}),
                "pixel_format": (["yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le", "rgb24", "copy"], {"default": "yuv420p"}),
                "crf": ("INT", {"default": 23, "min": 0, "max": 63, "step": 1}),
                "output_format": (["mp4", "webm", "mov", "avi", "mkv"], {"default": "mp4"}),
                "audio_handling": (["copy original", "replace with new", "remove audio"], {"default": "copy original"}),
            },
            "optional": {
                "audio": ("AUDIO", {}),
                "audio_codec": (["aac", "mp3", "libopus"], {"default": "aac"}),
                "audio_bitrate": (["96k", "128k", "160k", "192k", "256k", "320k"], {"default": "192k"}),
                "output_file_opt": ("STRING", {"multiline": True, "default": "-preset medium", "tooltip": "Custom FFmpeg output options. One option per line, e.g., -preset slow"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "convert_video"
    OUTPUT_NODE = True
    CATEGORY = "San4itos"

    def convert_video(self, video, filename_prefix, codec, pixel_format, crf, output_format, audio_handling,
                      audio=None, audio_codec="aac", audio_bitrate="192k", output_file_opt=""):

        full_output_folder, filename_part, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, self.output_dir, 0, 0)
        video_filename = f"{filename_part}_{counter:05}_.{output_format}"
        video_full_path = os.path.join(full_output_folder, video_filename)

        is_direct_path = hasattr(video, '_is_direct_path')

        error = None
        if is_direct_path:
            log_node_info(self.NODE_LOG_PREFIX, "Direct path detected. Using fast conversion.")
            ffmpeg_cmd = [self.ffmpeg_executable_path, '-y', '-i', video.filepath]
            
            base_params = {}
            if codec != 'copy': base_params['-c:v'] = codec
            if pixel_format != 'copy': base_params['-pix_fmt'] = pixel_format
            if codec not in ['copy']: base_params['-crf'] = crf

            final_params = self._build_ffmpeg_params(base_params, output_file_opt, self.NODE_LOG_PREFIX)
            ffmpeg_cmd.extend(final_params)

            if audio_handling == "copy original": ffmpeg_cmd.extend(['-c:a', 'copy'])
            elif audio_handling == "remove audio": ffmpeg_cmd.extend(['-an'])
            elif audio_handling == "replace with new": log_node_warning(self.NODE_LOG_PREFIX, "Audio replacement is not supported in direct path mode. Audio will be copied.")
            
            ffmpeg_cmd.append(video_full_path)
            error = self._execute_ffmpeg_command(ffmpeg_cmd, self.NODE_LOG_PREFIX)
        else:
            log_node_info(self.NODE_LOG_PREFIX, "Standard video object detected. Using compatibility mode.")
            components = video.get_components()
            images, source_audio, source_fps = components.images, components.audio, float(components.frame_rate)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                for i, image_tensor in enumerate(images):
                    Image.fromarray((image_tensor.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(temp_dir, f"frame_{i:06d}.png"))

                ffmpeg_cmd = [self.ffmpeg_executable_path, '-y', '-framerate', str(source_fps), '-i', os.path.join(temp_dir, 'frame_%06d.png')]
                
                final_audio = source_audio if audio_handling == "copy original" else audio if audio_handling == "replace with new" else None
                has_audio = final_audio and "waveform" in final_audio and final_audio["waveform"].numel() > 0

                if has_audio:
                    temp_audio_file = os.path.join(temp_dir, "temp_audio.wav")
                    torchaudio.save(temp_audio_file, final_audio["waveform"][0].cpu(), final_audio["sample_rate"])
                    ffmpeg_cmd.extend(['-i', temp_audio_file])

                base_params = {'-c:v': codec if codec != "copy" else "libx264", '-pix_fmt': pixel_format, '-crf': crf}
                final_params = self._build_ffmpeg_params(base_params, output_file_opt, self.NODE_LOG_PREFIX)
                ffmpeg_cmd.extend(final_params)

                if has_audio: ffmpeg_cmd.extend(['-c:a', audio_codec, '-b:a', audio_bitrate, '-shortest'])
                else: ffmpeg_cmd.extend(['-an'])
                
                ffmpeg_cmd.append(video_full_path)
                error = self._execute_ffmpeg_command(ffmpeg_cmd, self.NODE_LOG_PREFIX)

        if error: return error

        preview = [{"filename": video_filename, "subfolder": subfolder, "type": self.type}]
        return {"ui": {"videos": preview}}


NODE_CLASS_MAPPINGS = {
    "SaveFramesToVideoFFmpeg_san4itos": SaveFramesToVideoFFmpeg,
    "ConvertVideoFFmpeg_san4itos": ConvertVideoFFmpeg,
    "LoadVideoByPath_san4itos": LoadVideoByPath_san4itos,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveFramesToVideoFFmpeg_san4itos": "Save Images to Video (FFmpeg)",
    "ConvertVideoFFmpeg_san4itos": "Convert Video (FFmpeg)",
    "LoadVideoByPath_san4itos": "Load Video by Path",
}
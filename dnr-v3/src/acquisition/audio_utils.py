import os

import ffmpeg
import resampy
import soundfile as sf


def resample_and_save_python(
    audio_file, output_file, output_fs=48000, output_bitrate=24
):
    data, original_fs = sf.read(audio_file, always_2d=True)  # (n_samples, n_channels)

    if original_fs != output_fs:
        data = resampy.resample(
            data, original_fs, output_fs, axis=0, filter="kaiser_best"
        )

    if output_bitrate == 24:
        subtype = "PCM_24"
    elif output_bitrate == 16:
        subtype = "PCM_16"
    else:
        raise ValueError(f"Unsupported bitrate: {output_bitrate}")

    sf.write(output_file, data, output_fs, subtype=subtype)


def resample_and_save(
    audio_file, output_file, output_fs=48000, output_bitrate=24, backend="ffmpeg"
):
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    info = sf.info(audio_file)

    original_format = info.format
    original_subtype = info.subtype
    original_fs = info.samplerate

    if backend == "ffmpeg":
        ffmpeg.input(audio_file).output(
            output_file,
            ar=output_fs,
            acodec="pcm_s24le" if output_bitrate == 24 else "pcm_s16le",
        ).run(overwrite_output=True, quiet=True)

    elif backend == "python":
        resample_and_save_python(audio_file, output_file, output_fs, output_bitrate)

    return {
        "audio_file": audio_file,
        "output_file": output_file,
        "original_fs": original_fs,
        "original_format": original_format,
        "original_subtype": original_subtype,
        "output_fs": output_fs,
        "output_bitrate": output_bitrate,
    }


def safe_resample_and_save(audio_file, output_file, output_fs=48000, output_bitrate=24):
    try:
        return resample_and_save(audio_file, output_file, output_fs, output_bitrate)
    except Exception as e:
        return {
            "audio_file": audio_file,
            "output_file": output_file,
        }

        print(f"Error processing {audio_file}: {e}")


def wav_to_flac(wav_file, flac_file):
    os.makedirs(os.path.dirname(flac_file), exist_ok=True)

    err, out = (
        ffmpeg.input(wav_file)
        .output(flac_file, acodec="flac")
        .run(overwrite_output=True, capture_stdout=False, capture_stderr=True)
    )

    if err:
        print(err)

    return {"wav_file": wav_file, "flac_file": flac_file}

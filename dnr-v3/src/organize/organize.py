import hydra
from omegaconf import DictConfig

from .datasets.moisesdb import organize_moisesdb

# TODO: defer imports to the function level
# from .datasets.aishell3 import organize_aishell3
# from .datasets.arabic_speech_corpus import organize_arabic_speech_corpus
# from .datasets.audiocite import organize_audiocite
# from .datasets.faroese import organize_faroese
# from .datasets.fma import organize_fma
# from .datasets.fsd50k import organize_fsd50k
# from .datasets.google_accented_eng import organize_accented_english
# from .datasets.google_generic import organize_generic
# from .datasets.google_indic import organize_indic
# from .datasets.google_spanish import organize_spanish
# from .datasets.google_yoruba import organize_yoruba
# from .datasets.google_za import organize_za
# from .datasets.hui import organize_hui
# from .datasets.jvnv_pjs import organize_jvnv_pjs
# from .datasets.kazakh import organize_kazakh
# from .datasets.lamit import organize_lamit
# from .datasets.librispeech_hq import organize_librispeech_hq
# from .datasets.malaya import organize_malaya
# from .datasets.ukrainian import organize_ukrainian
from .datasets.musdb18hq import organize_musdb18hq

_ORGANIZER = {
    # "speech-english-slr12-librispeech-hq": organize_librispeech_hq,
    # "effects-fsd50k": organize_fsd50k,
    # "music-fma": organize_fma,
    # "speech-german-hui": organize_hui,
    # "speech-chinese-slr93-aishell3": organize_aishell3,
    # "speech-spanish-slr-google": organize_spanish,
    # "speech-french-slr139-audiocite": organize_audiocite,
    # "speech-indic-slr-google": organize_indic,
    # "speech-inc-slr-google": organize_indic,
    # "speech-dra-slr-google": organize_indic,
    # "speech-za-slr32-google": organize_za,
    # "speech-bnt-slr32-google": organize_za,
    # "speech-yoruba-slr86-google": organize_yoruba,
    # "speech-english-slr70-google-nigerian": organize_accented_english,
    # "speech-english-slr71-google-chilean": organize_accented_english,
    # "speech-english-slr83-google-british-isles": organize_accented_english,
    # "speech-javanese-slr41-google": organize_generic,
    # "speech-sundanese-slr44-google": organize_generic,
    # "speech-burmese-slr80-google": organize_generic,
    # "speech-khmer-slr42-google": organize_generic,
    # "speech-basque-slr76-google": organize_generic,
    # "speech-galician-slr77-google": organize_generic,
    # "speech-catalan-slr69-google": organize_generic,
    # "speech-faroese-slr125-google": organize_generic,
    # "speech-japanese-jvnv-pjs": organize_jvnv_pjs,
    # "speech-italian-lamit": organize_lamit,
    # "speech-malay-malaya-speech": organize_malaya,
    # "speech-ukrainian-smoliakov": organize_ukrainian,
    # "speech-kazakh-slr140": organize_kazakh,
    # "speech-faroese-slr125-blark": organize_faroese,
    # "speech-arabic-asc": organize_arabic_speech_corpus,
    "music-musdb18hq": organize_musdb18hq,
    "music-moisesdb": organize_moisesdb,
}


@hydra.main(config_path=os.path.expandvars("$ORG_CONFIG_ROOT"))
def organize(cfg: DictConfig) -> None:
    """
    Main entry point for organizing a raw dataset into the project's format.

    This function uses Hydra to load a dataset-specific configuration and then
    calls the appropriate organizer function from the _ORGANIZER dictionary.

    Args:
        cfg: The Hydra configuration object for a specific dataset.
    """

    organizer = _ORGANIZER.get(cfg.dataset.name, None)

    if organizer is None:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    organizer(cfg)

    print("Organized dataset.")


if __name__ == "__main__":
    organize()

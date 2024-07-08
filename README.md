# Divide and Remaster v3

Divide and Remaster v3 is a multilingual rework of the Divide and Remaster v2 dataset by Pétermann et al. 

The major changes from DnR v2 are as follows:
- the dialogue stem now contains content from more than 30 languages across various language families;
- speech, vocals, and/or vocalizations have been removed from the music and effects stems;
- loudness and timing parametrization have been adjusted to approximate the distributions of real cinematic content;
- the mastering process now preserves relative loudness between stems and approximates standard industry practices.

## Getting the dataset

DnR v3 variants are available via Zenodo. See below for links.

| Variant | Variant Code | Size | Link |
| -- | -- | -- | -- |
| Multilingual | multi | TBA |
| English | eng | TBA |
| German | deu | TBA |
| Faroese | fao | TBA |
| French | fra | TBA |
| Spanish | spa |  TBA|
| Chinese | cmn |  TBA|
| Basque | eus |  TBA|
| Japanese | jpn |  TBA|
| Indo-Aryan | inc |  TBA|
| Dravidian | dra |  TBA|
| Bantu | bnt | TBA |
| Yoruba | yor |  TBA|

### Extracting

```bash

export DNR_V3_ROOT="{path to your desired Dnr v3 folder}"
export VARIANT="{variant-code}"

mkdir -p $DNR_V3_ROOT
cd $DNR_V3_ROOT

tar -xvf "dnr-v3-$VARIANT-metadata.tar.gz"
tar -xvf "dnr-v3-$VARIANT-audio.train.tar.gz"
tar -xvf "dnr-v3-$VARIANT-audio.val.tar.gz"
tar -xvf "dnr-v3-$VARIANT-audio.test.tar.gz"
```


## Dataset Structure

```
.
└── multi/
    ├── audio/
    │   ├── train/
    │   │   └── {clip-id}/
    │   │       ├── speech.flac
    │   │       ├── music.flac
    │   │       ├── sfx.flac
    │   │       ├── sfx_fg.flac
    │   │       ├── sfx_bg.flac
    │   │       └── mixture.flac
    │   ├── val/
    │   │   └── {clip-id}/
    │   │       └── ...
    │   └── test/
    │       └── {clip-id}/
    │           └── ...
    ├── manifest/
    │   ├── train/
    │   │   └── {clip-id}/
    │   │       ├── speech.csv
    │   │       ├── music.csv
    │   │       ├── sfx_fg.csv
    │   │       └── sfx_bg.csv
    │   ├── val/
    │   │   └── {clip-id}/
    │   │       └── ...
    │   └── test/
    │       └── {clip-id}/
    │           └── ...
    └── audio_metadata/
        ├── train/
        │   └── {clip-id}.csv
        ├── val/
        │   └── {clip-id}.csv
        └── test/
            └── {clip-id}.csv
```


## License

Divide and Remaster v3 is released under the CC BY-SA 4.0 license. See wiki for full license information.

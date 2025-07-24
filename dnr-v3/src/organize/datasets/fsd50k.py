import os
from enum import StrEnum
from typing import List

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from omegaconf import DictConfig
from tqdm.contrib.concurrent import process_map

from ...const import AUDIO, MANIFEST
from ..license_utils import (
    license_family_to_usage,
    url_to_license,
    url_to_license_family,
)
from ..utils import (
    ffmpeg_reformat_to_buffer,
    get_acodec_soundfile,
    preface,
)


class FreeSoundClassMapping(StrEnum):
    V2_FOREGROUND = "foreground"
    V2_BACKGROUND = "background"
    V2_UNUSABLE = "unusable"

    V3_EFFECTS_NO_VOX = "effects-no-vox"
    V3_MUSIC = "music"
    V3_VOX = "vox"
    V3_UNUSABLE = "unusable"


FSD50K_CLASS_MAPPING_DNR_V2 = {
    "Accelerating_and_revving_and_vroom": FreeSoundClassMapping.V2_FOREGROUND,
    "Accordion": FreeSoundClassMapping.V2_UNUSABLE,
    "Acoustic_guitar": FreeSoundClassMapping.V2_UNUSABLE,
    "Aircraft": FreeSoundClassMapping.V2_BACKGROUND,
    "Alarm": FreeSoundClassMapping.V2_BACKGROUND,
    "Animal": FreeSoundClassMapping.V2_FOREGROUND,
    "Applause": FreeSoundClassMapping.V2_FOREGROUND,
    "Bark": FreeSoundClassMapping.V2_FOREGROUND,
    "Bass_drum": FreeSoundClassMapping.V2_UNUSABLE,
    "Bass_guitar": FreeSoundClassMapping.V2_UNUSABLE,
    "Bathtub_(filling_or_washing)": FreeSoundClassMapping.V2_FOREGROUND,
    "Bell": FreeSoundClassMapping.V2_FOREGROUND,
    "Bicycle": FreeSoundClassMapping.V2_FOREGROUND,
    "Bicycle_bell": FreeSoundClassMapping.V2_FOREGROUND,
    "Bird": FreeSoundClassMapping.V2_BACKGROUND,
    "Bird_vocalization_and_bird_call_and_bird_song": FreeSoundClassMapping.V2_FOREGROUND,
    "Boat_and_Water_vehicle": FreeSoundClassMapping.V2_BACKGROUND,
    "Boiling": FreeSoundClassMapping.V2_FOREGROUND,
    "Boom": FreeSoundClassMapping.V2_FOREGROUND,
    "Bowed_string_instrument": FreeSoundClassMapping.V2_UNUSABLE,
    "Brass_instrument": FreeSoundClassMapping.V2_UNUSABLE,
    "Breathing": FreeSoundClassMapping.V2_UNUSABLE,
    "Burping_and_eructation": FreeSoundClassMapping.V2_FOREGROUND,
    "Bus": FreeSoundClassMapping.V2_BACKGROUND,
    "Buzz": FreeSoundClassMapping.V2_FOREGROUND,
    "Camera": FreeSoundClassMapping.V2_FOREGROUND,
    "Car": FreeSoundClassMapping.V2_BACKGROUND,
    "Car_passing_by": FreeSoundClassMapping.V2_BACKGROUND,
    "Cat": FreeSoundClassMapping.V2_FOREGROUND,
    "Chatter": FreeSoundClassMapping.V2_BACKGROUND,
    "Cheering": FreeSoundClassMapping.V2_BACKGROUND,
    "Chewing_and_mastication": FreeSoundClassMapping.V2_FOREGROUND,
    "Chicken_and_rooster": FreeSoundClassMapping.V2_FOREGROUND,
    "Child_speech_and_kid_speaking": FreeSoundClassMapping.V2_UNUSABLE,
    "Chime": FreeSoundClassMapping.V2_UNUSABLE,
    "Chink_and_clink": FreeSoundClassMapping.V2_FOREGROUND,
    "Chirp_and_tweet": FreeSoundClassMapping.V2_FOREGROUND,
    "Chuckle_and_chortle": FreeSoundClassMapping.V2_UNUSABLE,
    "Church_bell": FreeSoundClassMapping.V2_UNUSABLE,
    "Clapping": FreeSoundClassMapping.V2_FOREGROUND,
    "Clock": FreeSoundClassMapping.V2_FOREGROUND,
    "Coin_(dropping)": FreeSoundClassMapping.V2_FOREGROUND,
    "Computer_keyboard": FreeSoundClassMapping.V2_FOREGROUND,
    "Conversation": FreeSoundClassMapping.V2_UNUSABLE,
    "Cough": FreeSoundClassMapping.V2_UNUSABLE,
    "Cowbell": FreeSoundClassMapping.V2_UNUSABLE,
    "Crack": FreeSoundClassMapping.V2_FOREGROUND,
    "Crackle": FreeSoundClassMapping.V2_FOREGROUND,
    "Crash_cymbal": FreeSoundClassMapping.V2_UNUSABLE,
    "Cricket": FreeSoundClassMapping.V2_FOREGROUND,
    "Crow": FreeSoundClassMapping.V2_FOREGROUND,
    "Crowd": FreeSoundClassMapping.V2_BACKGROUND,
    "Crumpling_and_crinkling": FreeSoundClassMapping.V2_FOREGROUND,
    "Crushing": FreeSoundClassMapping.V2_FOREGROUND,
    "Crying_and_sobbing": FreeSoundClassMapping.V2_UNUSABLE,
    "Cupboard_open_or_close": FreeSoundClassMapping.V2_FOREGROUND,
    "Cutlery_and_silverware": FreeSoundClassMapping.V2_FOREGROUND,
    "Cymbal": FreeSoundClassMapping.V2_UNUSABLE,
    "Dishes_and_pots_and_pans": FreeSoundClassMapping.V2_FOREGROUND,
    "Dog": FreeSoundClassMapping.V2_FOREGROUND,
    "Domestic_animals_and_pets": FreeSoundClassMapping.V2_BACKGROUND,
    "Domestic_sounds_and_home_sounds": FreeSoundClassMapping.V2_BACKGROUND,
    "Door": FreeSoundClassMapping.V2_FOREGROUND,
    "Doorbell": FreeSoundClassMapping.V2_FOREGROUND,
    "Drawer_open_or_close": FreeSoundClassMapping.V2_FOREGROUND,
    "Drill": FreeSoundClassMapping.V2_FOREGROUND,
    "Drip": FreeSoundClassMapping.V2_FOREGROUND,
    "Drum": FreeSoundClassMapping.V2_UNUSABLE,
    "Drum_kit": FreeSoundClassMapping.V2_UNUSABLE,
    "Electric_guitar": FreeSoundClassMapping.V2_UNUSABLE,
    "Engine": FreeSoundClassMapping.V2_BACKGROUND,
    "Engine_starting": FreeSoundClassMapping.V2_FOREGROUND,
    "Explosion": FreeSoundClassMapping.V2_FOREGROUND,
    "Fart": FreeSoundClassMapping.V2_FOREGROUND,
    "Female_singing": FreeSoundClassMapping.V2_UNUSABLE,
    "Female_speech_and_woman_speaking": FreeSoundClassMapping.V2_UNUSABLE,
    "Fill_(with_liquid)": FreeSoundClassMapping.V2_FOREGROUND,
    "Finger_snapping": FreeSoundClassMapping.V2_FOREGROUND,
    "Fire": FreeSoundClassMapping.V2_BACKGROUND,
    "Fireworks": FreeSoundClassMapping.V2_FOREGROUND,
    "Fixed-wing_aircraft_and_airplane": FreeSoundClassMapping.V2_BACKGROUND,
    "Fowl": FreeSoundClassMapping.V2_BACKGROUND,
    "Frog": FreeSoundClassMapping.V2_FOREGROUND,
    "Frying_(food)": FreeSoundClassMapping.V2_BACKGROUND,
    "Gasp": FreeSoundClassMapping.V2_UNUSABLE,
    "Giggle": FreeSoundClassMapping.V2_UNUSABLE,
    "Glass": FreeSoundClassMapping.V2_FOREGROUND,
    "Glockenspiel": FreeSoundClassMapping.V2_UNUSABLE,
    "Gong": FreeSoundClassMapping.V2_UNUSABLE,
    "Growling": FreeSoundClassMapping.V2_FOREGROUND,
    "Guitar": FreeSoundClassMapping.V2_UNUSABLE,
    "Gull_and_seagull": FreeSoundClassMapping.V2_BACKGROUND,
    "Gunshot_and_gunfire": FreeSoundClassMapping.V2_FOREGROUND,
    "Gurgling": FreeSoundClassMapping.V2_FOREGROUND,
    "Hammer": FreeSoundClassMapping.V2_FOREGROUND,
    "Hands": FreeSoundClassMapping.V2_FOREGROUND,
    "Harmonica": FreeSoundClassMapping.V2_UNUSABLE,
    "Harp": FreeSoundClassMapping.V2_UNUSABLE,
    "Hi-hat": FreeSoundClassMapping.V2_UNUSABLE,
    "Hiss": FreeSoundClassMapping.V2_BACKGROUND,
    "Human_group_actions": FreeSoundClassMapping.V2_UNUSABLE,
    "Human_voice": FreeSoundClassMapping.V2_UNUSABLE,
    "Idling": FreeSoundClassMapping.V2_BACKGROUND,
    "Insect": FreeSoundClassMapping.V2_BACKGROUND,
    "Keyboard_(musical)": FreeSoundClassMapping.V2_UNUSABLE,
    "Keys_jangling": FreeSoundClassMapping.V2_FOREGROUND,
    "Knock": FreeSoundClassMapping.V2_FOREGROUND,
    "Laughter": FreeSoundClassMapping.V2_UNUSABLE,
    "Liquid": FreeSoundClassMapping.V2_FOREGROUND,
    "Livestock_and_farm_animals_and_working_animals": FreeSoundClassMapping.V2_BACKGROUND,
    "Male_singing": FreeSoundClassMapping.V2_UNUSABLE,
    "Male_speech_and_man_speaking": FreeSoundClassMapping.V2_UNUSABLE,
    "Mallet_percussion": FreeSoundClassMapping.V2_UNUSABLE,
    "Marimba_and_xylophone": FreeSoundClassMapping.V2_UNUSABLE,
    "Mechanical_fan": FreeSoundClassMapping.V2_BACKGROUND,
    "Mechanisms": FreeSoundClassMapping.V2_BACKGROUND,
    "Meow": FreeSoundClassMapping.V2_FOREGROUND,
    "Microwave_oven": FreeSoundClassMapping.V2_BACKGROUND,
    "Motor_vehicle_(road)": FreeSoundClassMapping.V2_BACKGROUND,
    "Motorcycle": FreeSoundClassMapping.V2_BACKGROUND,
    "Music": FreeSoundClassMapping.V2_UNUSABLE,
    "Musical_instrument": FreeSoundClassMapping.V2_UNUSABLE,
    "Ocean": FreeSoundClassMapping.V2_BACKGROUND,
    "Organ": FreeSoundClassMapping.V2_UNUSABLE,
    "Packing_tape_and_duct_tape": FreeSoundClassMapping.V2_FOREGROUND,
    "Percussion": FreeSoundClassMapping.V2_UNUSABLE,
    "Piano": FreeSoundClassMapping.V2_UNUSABLE,
    "Plucked_string_instrument": FreeSoundClassMapping.V2_UNUSABLE,
    "Pour": FreeSoundClassMapping.V2_FOREGROUND,
    "Power_tool": FreeSoundClassMapping.V2_FOREGROUND,
    "Printer": FreeSoundClassMapping.V2_FOREGROUND,
    "Purr": FreeSoundClassMapping.V2_FOREGROUND,
    "Race_car_and_auto_racing": FreeSoundClassMapping.V2_BACKGROUND,
    "Rail_transport": FreeSoundClassMapping.V2_BACKGROUND,
    "Rain": FreeSoundClassMapping.V2_BACKGROUND,
    "Raindrop": FreeSoundClassMapping.V2_BACKGROUND,
    "Ratchet_and_pawl": FreeSoundClassMapping.V2_FOREGROUND,
    "Rattle": FreeSoundClassMapping.V2_FOREGROUND,
    "Rattle_(instrument)": FreeSoundClassMapping.V2_UNUSABLE,
    "Respiratory_sounds": FreeSoundClassMapping.V2_UNUSABLE,
    "Ringtone": FreeSoundClassMapping.V2_UNUSABLE,
    "Run": FreeSoundClassMapping.V2_FOREGROUND,
    "Sawing": FreeSoundClassMapping.V2_FOREGROUND,
    "Scissors": FreeSoundClassMapping.V2_FOREGROUND,
    "Scratching_(performance_technique)": FreeSoundClassMapping.V2_FOREGROUND,
    "Screaming": FreeSoundClassMapping.V2_UNUSABLE,
    "Screech": FreeSoundClassMapping.V2_UNUSABLE,
    "Shatter": FreeSoundClassMapping.V2_FOREGROUND,
    "Shout": FreeSoundClassMapping.V2_UNUSABLE,
    "Sigh": FreeSoundClassMapping.V2_UNUSABLE,
    "Singing": FreeSoundClassMapping.V2_UNUSABLE,
    "Sink_(filling_or_washing)": FreeSoundClassMapping.V2_BACKGROUND,
    "Siren": FreeSoundClassMapping.V2_BACKGROUND,
    "Skateboard": FreeSoundClassMapping.V2_FOREGROUND,
    "Slam": FreeSoundClassMapping.V2_FOREGROUND,
    "Sliding_door": FreeSoundClassMapping.V2_FOREGROUND,
    "Snare_drum": FreeSoundClassMapping.V2_UNUSABLE,
    "Sneeze": FreeSoundClassMapping.V2_UNUSABLE,
    "Speech": FreeSoundClassMapping.V2_UNUSABLE,
    "Speech_synthesizer": FreeSoundClassMapping.V2_UNUSABLE,
    "Splash_and_splatter": FreeSoundClassMapping.V2_FOREGROUND,
    "Squeak": FreeSoundClassMapping.V2_FOREGROUND,
    "Stream": FreeSoundClassMapping.V2_BACKGROUND,
    "Strum": FreeSoundClassMapping.V2_UNUSABLE,
    "Subway_and_metro_and_underground": FreeSoundClassMapping.V2_BACKGROUND,
    "Tabla": FreeSoundClassMapping.V2_UNUSABLE,
    "Tambourine": FreeSoundClassMapping.V2_UNUSABLE,
    "Tap": FreeSoundClassMapping.V2_FOREGROUND,
    "Tearing": FreeSoundClassMapping.V2_FOREGROUND,
    "Telephone": FreeSoundClassMapping.V2_FOREGROUND,
    "Thump_and_thud": FreeSoundClassMapping.V2_FOREGROUND,
    "Thunder": FreeSoundClassMapping.V2_BACKGROUND,
    "Thunderstorm": FreeSoundClassMapping.V2_BACKGROUND,
    "Tick": FreeSoundClassMapping.V2_FOREGROUND,
    "Tick-tock": FreeSoundClassMapping.V2_FOREGROUND,
    "Toilet_flush": FreeSoundClassMapping.V2_FOREGROUND,
    "Tools": FreeSoundClassMapping.V2_FOREGROUND,
    "Traffic_noise_and_roadway_noise": FreeSoundClassMapping.V2_BACKGROUND,
    "Train": FreeSoundClassMapping.V2_BACKGROUND,
    "Trickle_and_dribble": FreeSoundClassMapping.V2_BACKGROUND,
    "Truck": FreeSoundClassMapping.V2_BACKGROUND,
    "Trumpet": FreeSoundClassMapping.V2_UNUSABLE,
    "Typewriter": FreeSoundClassMapping.V2_FOREGROUND,
    "Typing": FreeSoundClassMapping.V2_FOREGROUND,
    "Vehicle": FreeSoundClassMapping.V2_BACKGROUND,
    "Vehicle_horn_and_car_horn_and_honking": FreeSoundClassMapping.V2_FOREGROUND,
    "Walk_and_footsteps": FreeSoundClassMapping.V2_BACKGROUND,
    "Water": FreeSoundClassMapping.V2_FOREGROUND,
    "Water_tap_and_faucet": FreeSoundClassMapping.V2_BACKGROUND,
    "Waves_and_surf": FreeSoundClassMapping.V2_BACKGROUND,
    "Whispering": FreeSoundClassMapping.V2_UNUSABLE,
    "Whoosh_and_swoosh_and_swish": FreeSoundClassMapping.V2_FOREGROUND,
    "Wild_animals": FreeSoundClassMapping.V2_BACKGROUND,
    "Wind": FreeSoundClassMapping.V2_BACKGROUND,
    "Wind_chime": FreeSoundClassMapping.V2_UNUSABLE,
    "Wind_instrument_and_woodwind_instrument": FreeSoundClassMapping.V2_UNUSABLE,
    "Wood": FreeSoundClassMapping.V2_FOREGROUND,
    "Writing": FreeSoundClassMapping.V2_FOREGROUND,
    "Yell": FreeSoundClassMapping.V2_UNUSABLE,
    "Zipper_(clothing)": FreeSoundClassMapping.V2_FOREGROUND,
}

FSD50K_CLASS_MAPPING_DNR_V3 = {
    "Accelerating_and_revving_and_vroom": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Accordion": FreeSoundClassMapping.V3_MUSIC,
    "Acoustic_guitar": FreeSoundClassMapping.V3_MUSIC,
    "Aircraft": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Alarm": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Animal": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Applause": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Bark": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Bass_drum": FreeSoundClassMapping.V3_MUSIC,
    "Bass_guitar": FreeSoundClassMapping.V3_MUSIC,
    "Bathtub_(filling_or_washing)": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Bell": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Bicycle": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Bicycle_bell": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Bird": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Bird_vocalization_and_bird_call_and_bird_song": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Boat_and_Water_vehicle": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Boiling": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Boom": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Bowed_string_instrument": FreeSoundClassMapping.V3_MUSIC,
    "Brass_instrument": FreeSoundClassMapping.V3_MUSIC,
    "Breathing": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Burping_and_eructation": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Bus": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Buzz": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Camera": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Car": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Car_passing_by": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Cat": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Chatter": FreeSoundClassMapping.V3_VOX,
    "Cheering": FreeSoundClassMapping.V3_VOX,
    "Chewing_and_mastication": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Chicken_and_rooster": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Child_speech_and_kid_speaking": FreeSoundClassMapping.V3_VOX,
    "Chime": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Chink_and_clink": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Chirp_and_tweet": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Chuckle_and_chortle": FreeSoundClassMapping.V3_VOX,
    "Church_bell": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Clapping": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Clock": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Coin_(dropping)": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Computer_keyboard": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Conversation": FreeSoundClassMapping.V3_VOX,
    "Cough": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Cowbell": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Crack": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Crackle": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Crash_cymbal": FreeSoundClassMapping.V3_MUSIC,
    "Cricket": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Crow": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Crowd": FreeSoundClassMapping.V3_VOX,
    "Crumpling_and_crinkling": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Crushing": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Crying_and_sobbing": FreeSoundClassMapping.V3_VOX,
    "Cupboard_open_or_close": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Cutlery_and_silverware": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Cymbal": FreeSoundClassMapping.V3_MUSIC,
    "Dishes_and_pots_and_pans": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Dog": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Domestic_animals_and_pets": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Domestic_sounds_and_home_sounds": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Door": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Doorbell": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Drawer_open_or_close": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Drill": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Drip": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Drum": FreeSoundClassMapping.V3_MUSIC,
    "Drum_kit": FreeSoundClassMapping.V3_MUSIC,
    "Electric_guitar": FreeSoundClassMapping.V3_MUSIC,
    "Engine": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Engine_starting": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Explosion": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Fart": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Female_singing": FreeSoundClassMapping.V3_VOX,
    "Female_speech_and_woman_speaking": FreeSoundClassMapping.V3_VOX,
    "Fill_(with_liquid)": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Finger_snapping": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Fire": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Fireworks": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Fixed-wing_aircraft_and_airplane": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Fowl": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Frog": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Frying_(food)": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Gasp": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Giggle": FreeSoundClassMapping.V3_VOX,
    "Glass": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Glockenspiel": FreeSoundClassMapping.V3_MUSIC,
    "Gong": FreeSoundClassMapping.V3_MUSIC,
    "Growling": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Guitar": FreeSoundClassMapping.V3_MUSIC,
    "Gull_and_seagull": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Gunshot_and_gunfire": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Gurgling": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Hammer": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Hands": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Harmonica": FreeSoundClassMapping.V3_MUSIC,
    "Harp": FreeSoundClassMapping.V3_MUSIC,
    "Hi-hat": FreeSoundClassMapping.V3_MUSIC,
    "Hiss": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Human_group_actions": FreeSoundClassMapping.V3_UNUSABLE,
    "Human_voice": FreeSoundClassMapping.V3_VOX,
    "Idling": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Insect": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Keyboard_(musical)": FreeSoundClassMapping.V3_MUSIC,
    "Keys_jangling": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Knock": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Laughter": FreeSoundClassMapping.V3_VOX,
    "Liquid": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Livestock_and_farm_animals_and_working_animals": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Male_singing": FreeSoundClassMapping.V3_VOX,
    "Male_speech_and_man_speaking": FreeSoundClassMapping.V3_VOX,
    "Mallet_percussion": FreeSoundClassMapping.V3_MUSIC,
    "Marimba_and_xylophone": FreeSoundClassMapping.V3_MUSIC,
    "Mechanical_fan": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Mechanisms": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Meow": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Microwave_oven": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Motor_vehicle_(road)": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Motorcycle": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Music": FreeSoundClassMapping.V3_MUSIC,
    "Musical_instrument": FreeSoundClassMapping.V3_MUSIC,
    "Ocean": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Organ": FreeSoundClassMapping.V3_MUSIC,
    "Packing_tape_and_duct_tape": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Percussion": FreeSoundClassMapping.V3_MUSIC,
    "Piano": FreeSoundClassMapping.V3_MUSIC,
    "Plucked_string_instrument": FreeSoundClassMapping.V3_MUSIC,
    "Pour": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Power_tool": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Printer": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Purr": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Race_car_and_auto_racing": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Rail_transport": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Rain": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Raindrop": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Ratchet_and_pawl": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Rattle": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Rattle_(instrument)": FreeSoundClassMapping.V3_MUSIC,
    "Respiratory_sounds": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Ringtone": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Run": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Sawing": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Scissors": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Scratching_(performance_technique)": FreeSoundClassMapping.V3_MUSIC,
    "Screaming": FreeSoundClassMapping.V3_VOX,
    "Screech": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Shatter": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Shout": FreeSoundClassMapping.V3_VOX,
    "Sigh": FreeSoundClassMapping.V3_VOX,
    "Singing": FreeSoundClassMapping.V3_VOX,
    "Sink_(filling_or_washing)": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Siren": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Skateboard": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Slam": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Sliding_door": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Snare_drum": FreeSoundClassMapping.V3_MUSIC,
    "Sneeze": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Speech": FreeSoundClassMapping.V3_VOX,
    "Speech_synthesizer": FreeSoundClassMapping.V3_VOX,
    "Splash_and_splatter": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Squeak": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Stream": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Strum": FreeSoundClassMapping.V3_MUSIC,
    "Subway_and_metro_and_underground": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Tabla": FreeSoundClassMapping.V3_MUSIC,
    "Tambourine": FreeSoundClassMapping.V3_MUSIC,
    "Tap": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Tearing": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Telephone": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Thump_and_thud": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Thunder": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Thunderstorm": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Tick": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Tick-tock": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Toilet_flush": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Tools": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Traffic_noise_and_roadway_noise": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Train": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Trickle_and_dribble": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Truck": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Trumpet": FreeSoundClassMapping.V3_MUSIC,
    "Typewriter": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Typing": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Vehicle": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Vehicle_horn_and_car_horn_and_honking": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Walk_and_footsteps": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Water": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Water_tap_and_faucet": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Waves_and_surf": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Whispering": FreeSoundClassMapping.V3_VOX,
    "Whoosh_and_swoosh_and_swish": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Wild_animals": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Wind": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Wind_chime": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Wind_instrument_and_woodwind_instrument": FreeSoundClassMapping.V3_MUSIC,
    "Wood": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Writing": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
    "Yell": FreeSoundClassMapping.V3_VOX,
    "Zipper_(clothing)": FreeSoundClassMapping.V3_EFFECTS_NO_VOX,
}


def _clean_class_mapping(
    class_map: List[FreeSoundClassMapping], version
) -> FreeSoundClassMapping:
    """Resolves a list of class mappings for a file into a single, final class."""
    if len(class_map) == 0:
        raise ValueError("Empty class mapping")

    if len(class_map) == 1:
        return class_map[0]

    if version == "v2":
        if FreeSoundClassMapping.V2_UNUSABLE in class_map:
            return FreeSoundClassMapping.V2_UNUSABLE
        elif FreeSoundClassMapping.V2_FOREGROUND in class_map:
            return FreeSoundClassMapping.V2_FOREGROUND
        elif FreeSoundClassMapping.V2_BACKGROUND in class_map:
            return FreeSoundClassMapping.V2_BACKGROUND
        else:
            raise ValueError(f"Invalid class mapping: {class_map}")

    if version == "v3":
        return FreeSoundClassMapping.V3_UNUSABLE


def _label_not_found(label):
    """Helper function to handle labels that are not found in the mapping."""
    # print(f"{label} not found")
    return None


def trim_relative_silence(
    audio,
    sampling_rate,
    frame_duration_ms=40.0,
    hop_duration_ms=10.0,
    db_threshold_rel_peak=20,
):
    """Trims leading and trailing silence from an audio signal based on a dB threshold."""
    audio, (start_index, end_index) = librosa.effects.trim(
        audio,
        top_db=db_threshold_rel_peak,
        ref=np.max,
        frame_length=int(sampling_rate * frame_duration_ms / 1000),
        hop_length=int(sampling_rate * hop_duration_ms / 1000),
    )

    start_time = start_index / sampling_rate
    end_time = end_index / sampling_rate

    return audio, start_time, end_time


def organize_file(
    file: str,
    class_func: str,
    license_: str,
    license_usage: str,
    data_path: str,
    split: str,
    cfg: DictConfig,
) -> None:
    """
    Organizes a single FSD50K file.

    This involves reformatting, trimming silence, and saving the file to the
    appropriate 'cleaned' directory based on its class and license.
    """
    subset = cfg.dataset.subset.format(license=license_usage, class_func=class_func)

    file_path = os.path.join(data_path, f"{file}.wav")

    output_path = os.path.join(
        cfg.data.cleaned_data_root,
        cfg.dataset.name,
        AUDIO,
        subset,
        split,
        f"{file}.wav",
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    audio = ffmpeg_reformat_to_buffer(
        file_path, cfg.audio.sampling_rate, cfg.audio.channels
    )

    audio, start_time, end_time = trim_relative_silence(
        audio,
        cfg.audio.sampling_rate,
        frame_duration_ms=cfg.dsp.trim.frame_duration_ms,
        hop_duration_ms=cfg.dsp.trim.hop_duration_ms,
        db_threshold_rel_peak=cfg.dsp.trim.db_threshold_rel_peak,
    )

    acodec = get_acodec_soundfile(cfg.audio.bit_depth)

    sf.write(output_path, audio, samplerate=cfg.audio.sampling_rate, subtype=acodec)

    manifest_entry = {
        "file": file_path.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
        "cleaned_path": output_path.replace(
            cfg.data.cleaned_data_root, "$CLEANED_DATA_ROOT"
        ),
        "license": license_,
        "license_usage": license_usage,
        "class": class_func,
        "dsid_fsd50k": file,
        "subset": subset,
    }

    return manifest_entry


def _print_labels_with_no_mappings(labels, mapping):
    """Helper function to print labels that could not be mapped to a class."""
    if len(mapping) == 0:
        print(labels)


def organize_split(split: str, cfg: DictConfig) -> None:
    """Organizes all files for a single FSD50K dataset split."""
    raw_split = cfg.splits[split]

    manifest = []

    for raw_src in raw_split:
        if split in ["train", "val"]:
            cols = ["fname", "split", "labels"]
        elif split == "test":
            cols = ["fname", "labels"]
        else:
            raise ValueError(f"Invalid split: {split}")

        metadata_df = pd.read_csv(
            os.path.join(cfg.data.raw_data_root, raw_src.split_metadata),
        )[cols]

        df = metadata_df.copy()

        if split in ["train", "val"]:
            df = df[df["split"] == split]

        df["labels"] = df["labels"].apply(lambda x: str(x).split(","))
        if cfg.class_mapping_version == "v2":
            df["class_mapping"] = df["labels"].apply(
                lambda x: list(
                    set(
                        FSD50K_CLASS_MAPPING_DNR_V2.get(label, _label_not_found(label))
                        for label in x
                    )
                )
            )
        elif cfg.class_mapping_version == "v3":
            df["class_mapping"] = df["labels"].apply(
                lambda x: list(
                    set(
                        FSD50K_CLASS_MAPPING_DNR_V3.get(label, _label_not_found(label))
                        for label in x
                    )
                )
            )
        else:
            raise ValueError(
                f"Invalid class mapping version: {cfg.class_mapping_version}"
            )

        df["class_mapping"] = df["class_mapping"].apply(
            lambda x: [y for y in x if y is not None]
        )

        df.apply(
            lambda x: _print_labels_with_no_mappings(x["labels"], x["class_mapping"]),
            axis=1,
        )

        df["class"] = df["class_mapping"].apply(
            lambda x: _clean_class_mapping(x, cfg.class_mapping_version)
        )

        licenses = pd.read_json(
            os.path.join(cfg.data.raw_data_root, raw_src.license_metadata),
            orient="index",
        ).reset_index(names="fname")
        df = df.merge(licenses, on="fname")

        df["license_family"] = df["license"].apply(url_to_license_family)
        df["license"] = df["license"].apply(url_to_license)

        df["license_usage"] = df["license_family"].apply(license_family_to_usage)

        data_path = os.path.join(cfg.data.raw_data_root, raw_src.path)

        files = df["fname"].tolist()
        classes = df["class"].tolist()
        licenses = df["license"].tolist()
        license_usage = df["license_usage"].tolist()

        manifest += process_map(
            organize_file,
            files,
            classes,
            licenses,
            license_usage,
            [data_path] * len(files),
            [split] * len(files),
            [cfg] * len(files),
            chunksize=8,
        )

    manifest_df = pd.DataFrame(manifest)

    for subset, dfg in manifest_df.groupby("subset"):
        manifest_path = os.path.join(
            cfg.data.cleaned_data_root,
            cfg.dataset.name,
            MANIFEST,
            subset,
            f"{split}.csv",
        )

        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

        dfg.to_csv(manifest_path, index=False)


def organize_fsd50k(cfg: DictConfig) -> None:
    """Main entry point for organizing the FSD50K dataset."""
    preface(cfg)

    # class_df = pd.read_csv(os.path.join(cfg.data.raw_data_root, cfg.class_specs),
    #                        names=["id", "name", "code"])

    for split in cfg.splits:
        print(f"Organizing split: {split}")
        organize_split(split=split, cfg=cfg)

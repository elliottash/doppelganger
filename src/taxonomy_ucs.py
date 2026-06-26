"""UCS taxonomy backbone for the diverse movie/game SFX corpus.

We label real audio with the Universal Category System (UCS) CatID — the public, industry
standard used by Soundly/Soundminer/BOOM/Sonniss — rather than a bespoke scheme, so labels are
professional, recognizable, and a superset of both DCASE-7 and the game's sound design.

This module provides:
  * UCS_CATEGORIES : the CatIDs we target (movie/game-relevant subset), with a human name and a
    coarse MORPHOLOGY (transient/texture/tonal/whoosh/vocal/mechanical/designed/ambience).
  * AUDIOSET_TO_UCS : map FSD50K/AudioSet leaf-class names (lowercased) -> CatID. Many-to-one.
    Refined against FSD50K's vocabulary.csv at ingest time (unmapped -> dropped, logged).
  * helpers to map a clip's AudioSet label set to a single CatID (most specific wins).

The CLAP zero-shot text used to VERIFY a mapped label is `verify_prompt(catid)`.
"""
from __future__ import annotations

# CatID -> (human name, morphology, CLAP-verification phrase)
UCS_CATEGORIES = {
    # --- impact / transient ---
    "FOOT": ("footsteps", "transient", "footsteps walking"),
    "IMPT": ("impacts / hits", "transient", "a hard bang, thud or impact"),
    "GUN":  ("guns / firearms", "transient", "a gunshot"),
    "EXPL": ("explosions", "transient", "an explosion"),
    "GLAS": ("glass", "transient", "glass breaking"),
    "WOOD": ("wood", "transient", "wood creaking or breaking"),
    "METL": ("metal", "transient", "a metallic clang"),
    "DOOR": ("doors", "transient", "a door opening, closing or knocking"),
    "CRSH": ("crashes / destruction", "transient", "a crash, smash and destruction"),
    "KEYS": ("keys / coins / small metal", "transient", "jingling keys or coins"),
    "DSHS": ("dishes / cutlery", "transient", "dishes and cutlery clinking"),
    # --- texture / ambience ---
    "WATR": ("water / liquid", "texture", "water pouring or splashing"),
    "FIRE": ("fire", "texture", "a crackling fire"),
    "WIND": ("wind", "texture", "wind blowing"),
    "RAIN": ("rain", "texture", "steady rain"),
    "WTHR": ("weather / thunder", "texture", "thunder and storm"),
    "AMB":  ("ambiences", "ambience", "city traffic and road noise"),
    "CRWD": ("crowds", "ambience", "a crowd of people clapping and cheering"),
    "NATR": ("nature / vegetation", "texture", "rustling leaves and nature"),
    # --- tonal / designed ---
    "BELL": ("bells", "tonal", "a bell ringing"),
    "ALRM": ("alarms / sirens", "tonal", "an alarm or siren"),
    "UI":   ("user interface / beeps", "tonal", "an electronic UI beep"),
    "MUSC": ("musical hits / instruments", "tonal", "a musical instrument playing"),
    "MAGI": ("magic", "designed", "a magic spell sound effect"),
    "SCIF": ("sci-fi", "designed", "a sci-fi sound effect"),
    "WHSH": ("whoosh / swish", "whoosh", "a fast whoosh"),
    "ELEC": ("electricity", "texture", "an electric buzz or hum"),
    # --- mechanical / vehicles ---
    "MECH": ("mechanical", "mechanical", "a machine, motor or mechanism running"),
    "VEH":  ("vehicles / engines", "mechanical", "a vehicle engine"),
    "TOOL": ("tools", "mechanical", "a power tool, drill or saw"),
    "CLCK": ("clocks / ticks", "mechanical", "a clock ticking"),
    "COMP": ("computer / keyboard", "mechanical", "typing on a keyboard"),
    "PHON": ("telephones", "tonal", "a telephone ringing"),
    # --- creatures / humans ---
    "ANML": ("animals", "vocal", "an animal, dog, cat or insect"),
    "BIRD": ("birds", "vocal", "a bird call"),
    "CREA": ("creatures (designed)", "vocal", "a growling monster creature"),
    "VOX":  ("human voice / non-speech", "vocal", "a human voice, cough, laugh or shout"),
    "BODY": ("body / foley", "transient", "a person eating, snapping or clapping"),
    "CLTH": ("cloth", "texture", "cloth, fabric or zipper movement"),
}

MORPHOLOGY_OF = {cid: m for cid, (_, m, _) in UCS_CATEGORIES.items()}


def verify_prompt(catid: str) -> str:
    return UCS_CATEGORIES[catid][2]


# FSD50K leaf-class (exact name, lowercased) -> CatID. Keys match FSD50K vocabulary.csv.
# Broad parents (Animal, Music, Vehicle, Domestic_sounds, Mechanisms, Liquid, Human_voice) and
# speech/singing are intentionally omitted so labels stay precise.
AUDIOSET_TO_UCS = {
    # guns / explosions / impacts / breaking / materials
    "gunshot_and_gunfire": "GUN", "explosion": "EXPL", "fireworks": "EXPL", "boom": "EXPL",
    "glass": "GLAS", "shatter": "GLAS", "chink_and_clink": "GLAS",
    "wood": "WOOD", "crack": "WOOD", "crushing": "CRSH", "crumpling_and_crinkling": "CLTH",
    "tap": "IMPT", "knock": "DOOR", "thump_and_thud": "IMPT", "slam": "DOOR", "tearing": "CLTH",
    "door": "DOOR", "doorbell": "BELL", "sliding_door": "DOOR", "cupboard_open_or_close": "DOOR",
    "drawer_open_or_close": "DOOR", "dishes_and_pots_and_pans": "DSHS",
    "cutlery_and_silverware": "DSHS", "coin_(dropping)": "KEYS", "keys_jangling": "KEYS",
    "packing_tape_and_duct_tape": "CLTH", "zipper_(clothing)": "CLTH",
    # water / fire / weather
    "water": "WATR", "water_tap_and_faucet": "WATR", "pour": "WATR", "splash_and_splatter": "WATR",
    "drip": "WATR", "gurgling": "WATR", "stream": "WATR", "ocean": "WATR", "waves_and_surf": "WATR",
    "boiling": "WATR", "trickle_and_dribble": "WATR", "fill_(with_liquid)": "WATR",
    "bathtub_(filling_or_washing)": "WATR", "sink_(filling_or_washing)": "WATR",
    "toilet_flush": "WATR", "fire": "FIRE", "crackle": "FIRE", "frying_(food)": "FIRE",
    "wind": "WIND", "rain": "RAIN", "raindrop": "RAIN", "thunder": "WTHR", "thunderstorm": "WTHR",
    # tonal / ui / phone / alarm
    "bell": "BELL", "church_bell": "BELL", "bicycle_bell": "BELL", "chime": "BELL",
    "wind_chime": "BELL", "cowbell": "BELL", "gong": "MUSC",
    "alarm": "ALRM", "siren": "ALRM", "buzz": "ELEC",
    "telephone": "PHON", "ringtone": "PHON",
    # mechanical / vehicles / tools / clocks / computer
    "engine": "VEH", "engine_starting": "VEH", "idling": "VEH",
    "accelerating_and_revving_and_vroom": "VEH", "car": "VEH", "car_passing_by": "VEH",
    "motorcycle": "VEH", "truck": "VEH", "bus": "VEH", "train": "VEH", "aircraft": "VEH",
    "fixed-wing_aircraft_and_airplane": "VEH", "boat_and_water_vehicle": "VEH",
    "motor_vehicle_(road)": "VEH", "race_car_and_auto_racing": "VEH", "rail_transport": "VEH",
    "subway_and_metro_and_underground": "VEH", "bicycle": "VEH",
    "vehicle_horn_and_car_horn_and_honking": "VEH", "traffic_noise_and_roadway_noise": "AMB",
    "power_tool": "TOOL", "drill": "TOOL", "sawing": "TOOL", "hammer": "TOOL", "scissors": "TOOL",
    "mechanical_fan": "MECH", "microwave_oven": "MECH", "printer": "MECH", "ratchet_and_pawl": "MECH",
    "rattle": "MECH", "squeak": "MECH", "camera": "TOOL",
    "clock": "CLCK", "tick": "CLCK", "tick-tock": "CLCK",
    "computer_keyboard": "COMP", "typing": "COMP", "typewriter": "COMP",
    # musical instruments -> MUSC (tonal stingers/hits); capped per category downstream
    "accordion": "MUSC", "acoustic_guitar": "MUSC", "bass_drum": "MUSC", "bass_guitar": "MUSC",
    "bowed_string_instrument": "MUSC", "brass_instrument": "MUSC", "crash_cymbal": "MUSC",
    "cymbal": "MUSC", "drum": "MUSC", "drum_kit": "MUSC", "electric_guitar": "MUSC",
    "glockenspiel": "MUSC", "guitar": "MUSC", "harmonica": "MUSC", "harp": "MUSC",
    "hi-hat": "MUSC", "keyboard_(musical)": "MUSC", "mallet_percussion": "MUSC",
    "marimba_and_xylophone": "MUSC", "organ": "MUSC", "percussion": "MUSC", "piano": "MUSC",
    "plucked_string_instrument": "MUSC", "rattle_(instrument)": "MUSC", "snare_drum": "MUSC",
    "tabla": "MUSC", "tambourine": "MUSC", "trumpet": "MUSC",
    "wind_instrument_and_woodwind_instrument": "MUSC", "scratching_(performance_technique)": "MUSC",
    # creatures / animals / birds / humans (non-speech) / body / crowd
    "dog": "ANML", "bark": "ANML", "cat": "ANML", "meow": "ANML", "purr": "ANML",
    "chicken_and_rooster": "ANML", "fowl": "ANML", "frog": "ANML", "insect": "ANML",
    "cricket": "ANML", "horse": "ANML",
    "livestock_and_farm_animals_and_working_animals": "ANML", "wild_animals": "ANML",
    "bird": "BIRD", "bird_vocalization_and_bird_call_and_bird_song": "BIRD", "crow": "BIRD",
    "chirp_and_tweet": "BIRD", "gull_and_seagull": "BIRD",
    "growling": "CREA", "hiss": "CREA", "roar": "CREA",
    "cough": "VOX", "sneeze": "VOX", "laughter": "VOX", "shout": "VOX", "screaming": "VOX",
    "yell": "VOX", "gasp": "VOX", "sigh": "VOX", "giggle": "VOX", "burping_and_eructation": "VOX",
    "breathing": "VOX", "respiratory_sounds": "VOX", "chuckle_and_chortle": "VOX",
    "crying_and_sobbing": "VOX", "fart": "BODY", "chewing_and_mastication": "BODY",
    "finger_snapping": "BODY", "hands": "BODY", "clapping": "CRWD", "applause": "CRWD",
    "crowd": "CRWD", "cheering": "CRWD", "chatter": "CRWD",
    "walk_and_footsteps": "FOOT", "run": "FOOT", "whoosh_and_swoosh_and_swish": "WHSH",
}


def label_to_catid(labels: list[str]) -> str | None:
    """Map a clip's AudioSet label list to one CatID (first/most specific match)."""
    for lab in labels:
        cid = AUDIOSET_TO_UCS.get(lab.strip().lower())
        if cid:
            return cid
    return None

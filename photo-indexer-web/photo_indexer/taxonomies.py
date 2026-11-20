"""Built-in taxonomies for classifying image content: faces, animals, pets, places, locations, objects."""

from __future__ import annotations

from typing import Dict, List, Optional

from .models import Taxonomy, TaxonomyEntry


def _entry(
    label: str,
    synonyms: Optional[List[str]] = None,
    parent: Optional[str] = None,
    threshold: float = 0.7,
) -> TaxonomyEntry:
    return TaxonomyEntry(
        label=label,
        synonyms=synonyms or [],
        parent=parent,
        confidence_threshold=threshold,
    )


FACE_TAXONOMY = Taxonomy(
    name="faces",
    version="1.0.0",
    description="Human face categories detected in images.",
    entries=[
        _entry("person", ["human", "individual", "people"]),
        _entry("child", ["kid", "baby", "toddler", "infant"]),
        _entry("adult", ["grown-up"]),
        _entry("elderly", ["senior", "old person"]),
        _entry("group", ["crowd", "multiple people"]),
        _entry("portrait", ["headshot", "selfie"]),
        _entry("face_closeup", ["close-up face"]),
    ],
)

ANIMAL_TAXONOMY = Taxonomy(
    name="animals",
    version="1.0.0",
    description="Wild and domestic animals detected in images.",
    entries=[
        _entry("dog", ["canine", "puppy"]),
        _entry("cat", ["feline", "kitten"]),
        _entry("bird", ["avian"]),
        _entry("horse", ["equine", "pony"]),
        _entry("cow", ["cattle", "bull", "calf"]),
        _entry("sheep", ["lamb", "ram"]),
        _entry("pig", ["swine", "hog"]),
        _entry("chicken", ["hen", "rooster"]),
        _entry("duck", ["goose", "swan"]),
        _entry("fish", ["seafood"]),
        _entry("deer", ["stag", "buck", "doe"]),
        _entry("bear", ["grizzly", "polar bear"]),
        _entry("wolf", ["wild dog"]),
        _entry("lion", ["big cat"]),
        _entry("tiger", ["big cat"]),
        _entry("elephant", ["pachyderm"]),
        _entry("monkey", ["primate", "ape", "chimpanzee", "gorilla"]),
        _entry("rabbit", ["bunny", "hare"]),
        _entry("squirrel", ["rodent"]),
        _entry("fox", ["vixen"]),
        _entry("snake", ["reptile", "serpent"]),
        _entry("turtle", ["tortoise"]),
        _entry("frog", ["amphibian", "toad"]),
        _entry("butterfly", ["insect", "moth", "bee"]),
    ],
)

PET_TAXONOMY = Taxonomy(
    name="pets",
    version="1.0.0",
    description="Domestic pets and companion animals.",
    entries=[
        _entry("dog", ["canine", "puppy", "pet dog"]),
        _entry("cat", ["feline", "kitten", "pet cat"]),
        _entry("bird", ["pet bird", "parrot", "canary"]),
        _entry("fish", ["pet fish", "goldfish", "aquarium"]),
        _entry("rabbit", ["pet rabbit", "bunny"]),
        _entry("hamster", ["rodent pet"]),
        _entry("guinea pig", ["cavy"]),
        _entry("ferret", []),
        _entry("turtle", ["pet turtle", "tortoise"]),
        _entry("snake", ["pet snake", "reptile"]),
        _entry("horse", ["pet horse", "pony"]),
        _entry("other_pet", ["exotic pet"]),
    ],
)

PLACE_TAXONOMY = Taxonomy(
    name="places",
    version="1.0.0",
    description="Places and settings detected in images.",
    entries=[
        _entry("home", ["house", "residence", "indoor"]),
        _entry("kitchen", ["cooking area"]),
        _entry("bedroom", ["sleeping area"]),
        _entry("living room", ["lounge"]),
        _entry("bathroom", ["restroom"]),
        _entry("office", ["workspace", "study"]),
        _entry("outdoors", ["outside", "nature"]),
        _entry("park", ["garden", "playground"]),
        _entry("beach", ["shore", "coast", "seaside"]),
        _entry("mountain", ["hill", "peak"]),
        _entry("forest", ["woods", "jungle"]),
        _entry("desert", ["dunes"]),
        _entry("city", ["urban", "town", "downtown"]),
        _entry("street", ["road", "avenue"]),
        _entry("building", ["architecture", "structure"]),
        _entry("bridge", ["overpass"]),
        _entry("restaurant", ["cafe", "diner"]),
        _entry("store", ["shop", "market", "mall"]),
        _entry("school", ["classroom", "university"]),
        _entry("hospital", ["clinic", "medical"]),
        _entry("airport", ["terminal", "runway"]),
        _entry("station", ["train station", "subway"]),
        _entry("stadium", ["arena", "sports field"]),
        _entry("temple", ["church", "mosque", "synagogue"]),
        _entry("hotel", ["resort", "inn"]),
        _entry("farm", ["barn", "ranch"]),
        _entry("lake", ["pond", "river", "stream"]),
        _entry("waterfall", ["cascade"]),
    ],
)

LOCATION_TAXONOMY = Taxonomy(
    name="locations",
    version="1.0.0",
    description="Geographic locations and landmarks detected in images.",
    entries=[
        _entry("landmark", ["monument", "famous place"]),
        _entry("cityscape", ["skyline"]),
        _entry("landscape", ["scenery"]),
        _entry("seascape", ["ocean view"]),
        _entry("architectural_landmark", ["building", "tower", "bridge"]),
        _entry("natural_landmark", ["canyon", "cave", "waterfall"]),
        _entry("indoor_location", ["interior", "room"]),
        _entry("outdoor_location", ["exterior"]),
    ],
)

OBJECT_TAXONOMY = Taxonomy(
    name="objects",
    version="1.0.0",
    description="Common everyday objects detected in images.",
    entries=[
        _entry("car", ["vehicle", "automobile", "truck"]),
        _entry("bicycle", ["bike", "cycle"]),
        _entry("motorcycle", ["motorbike"]),
        _entry("airplane", ["plane", "jet"]),
        _entry("boat", ["ship", "yacht", "canoe"]),
        _entry("train", ["locomotive"]),
        _entry("traffic light", ["signal"]),
        _entry("fire hydrant", []),
        _entry("stop sign", ["sign"]),
        _entry("parking meter", []),
        _entry("bench", ["seat"]),
        _entry("bird", ["avian"]),
        _entry("cat", ["feline"]),
        _entry("dog", ["canine"]),
        _entry("horse", ["equine"]),
        _entry("sheep", ["lamb"]),
        _entry("cow", ["cattle"]),
        _entry("elephant", []),
        _entry("bear", []),
        _entry("zebra", []),
        _entry("giraffe", []),
        _entry("backpack", ["bag", "rucksack"]),
        _entry("umbrella", ["parasol"]),
        _entry("handbag", ["purse", "tote"]),
        _entry("suitcase", ["luggage"]),
        _entry("frisbee", ["flying disc"]),
        _entry("skis", ["snowboard"]),
        _entry("snowboard", []),
        _entry("sports ball", ["football", "basketball", "soccer ball"]),
        _entry("kite", []),
        _entry("baseball bat", ["bat"]),
        _entry("baseball glove", ["mitt"]),
        _entry("skateboard", []),
        _entry("surfboard", []),
        _entry("tennis racket", ["racket"]),
        _entry("bottle", ["water bottle"]),
        _entry("wine glass", ["glass"]),
        _entry("cup", ["mug", "coffee cup"]),
        _entry("fork", ["cutlery"]),
        _entry("knife", ["cutlery"]),
        _entry("spoon", ["cutlery"]),
        _entry("bowl", ["dish"]),
        _entry("banana", ["fruit"]),
        _entry("apple", ["fruit"]),
        _entry("sandwich", ["food"]),
        _entry("orange", ["fruit"]),
        _entry("broccoli", ["vegetable"]),
        _entry("carrot", ["vegetable"]),
        _entry("hot dog", ["food"]),
        _entry("pizza", ["food"]),
        _entry("donut", ["pastry"]),
        _entry("cake", ["pastry"]),
        _entry("chair", ["seat", "furniture"]),
        _entry("couch", ["sofa", "furniture"]),
        _entry("potted plant", ["plant"]),
        _entry("bed", ["furniture"]),
        _entry("dining table", ["table", "furniture"]),
        _entry("toilet", ["restroom"]),
        _entry("tv", ["television", "monitor", "screen"]),
        _entry("laptop", ["computer", "notebook"]),
        _entry("mouse", ["computer mouse"]),
        _entry("remote", ["remote control"]),
        _entry("keyboard", ["computer keyboard"]),
        _entry("cell phone", ["phone", "smartphone", "mobile"]),
        _entry("microwave", ["oven"]),
        _entry("oven", ["stove"]),
        _entry("toaster", []),
        _entry("sink", ["washbasin"]),
        _entry("refrigerator", ["fridge"]),
        _entry("book", ["publication"]),
        _entry("clock", ["watch", "timepiece"]),
        _entry("vase", []),
        _entry("scissors", []),
        _entry("teddy bear", ["stuffed animal", "plush"]),
        _entry("hair drier", ["hair dryer"]),
        _entry("toothbrush", []),
    ],
)

ALL_TAXONOMIES: Dict[str, Taxonomy] = {
    "faces": FACE_TAXONOMY,
    "animals": ANIMAL_TAXONOMY,
    "pets": PET_TAXONOMY,
    "places": PLACE_TAXONOMY,
    "locations": LOCATION_TAXONOMY,
    "objects": OBJECT_TAXONOMY,
}

TARGET_TAXONOMY_NAMES = ["faces", "animals", "pets", "places", "locations", "objects"]


def get_taxonomy(name: str) -> Optional[Taxonomy]:
    """Return a taxonomy by name, or None if not found."""
    return ALL_TAXONOMIES.get(name)


def list_taxonomies() -> List[str]:
    """Return all available taxonomy names."""
    return list(ALL_TAXONOMIES.keys())

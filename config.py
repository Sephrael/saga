# config.py
"""
Configuration settings for the Saga Novel Generation system.
Centralizes constants for API endpoints, model names, file paths,
generation parameters, validation thresholds, and logging settings.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright 2025 Dennis Lewis
"""

import os
import numpy as np
import logging # Import logging to use its constants like INFO
from typing import Optional, Type, List

# --- API and Model Configuration ---
# URL for the Ollama server providing embeddings
OLLAMA_EMBED_URL: str = "http://192.168.64.1:11434"
# Base URL for the OpenAI-compatible API endpoint
OPENAI_API_BASE: str = "http://192.168.64.1:8080/v1"
# API Key for the OpenAI-compatible endpoint (replace if necessary)
OPENAI_API_KEY: str = "nope"

# Name of the embedding model used via Ollama
EMBEDDING_MODEL: str = "nomic-embed-text:latest"

# --- Model Tiering ---
# Largest model for most critical/creative tasks
LARGE_MODEL: str = "Qwen3-30B-A3B" # For drafting, revision, detailed planning
# Medium model for analysis, structured generation, less critical tasks
MEDIUM_MODEL: str = "Qwen3-30B-A3B" # For KG extraction, initial setup, consistency, JSON updates
# Smallest model for very simple, fast tasks
SUMMARIZATION_MODEL: str = "Qwen3-30B-A3B" # For chapter summaries

# Define main model for backward compatibility or general use (points to large)
MAIN_GENERATION_MODEL: str = LARGE_MODEL 

# Specific model assignments for clarity (can point to tiered models)
JSON_CORRECTION_MODEL: str = MEDIUM_MODEL
CONSISTENCY_CHECK_MODEL: str = MEDIUM_MODEL
KNOWLEDGE_UPDATE_MODEL: str = MEDIUM_MODEL # For combined char/world JSON updates & KG extraction
INITIAL_SETUP_MODEL: str = MEDIUM_MODEL # For plot outline and world-building generation
PLANNING_MODEL: str = LARGE_MODEL # Detailed scene planning benefits from larger model
DRAFTING_MODEL: str = LARGE_MODEL
REVISION_MODEL: str = LARGE_MODEL


# --- Output and File Paths ---
# Directory to store all output files (database, JSON state, chapter texts)
OUTPUT_DIR: str = "novel_output"
# Path to the SQLite database file
DATABASE_FILE: str = os.path.join(OUTPUT_DIR, "novel_data.db")
# Path to the JSON file storing the plot outline
PLOT_OUTLINE_FILE: str = os.path.join(OUTPUT_DIR, "plot_outline.json")
# Path to the JSON file storing character profiles
CHARACTER_PROFILES_FILE: str = os.path.join(OUTPUT_DIR, "character_profiles.json")
# Path to the JSON file storing world-building information
WORLD_BUILDER_FILE: str = os.path.join(OUTPUT_DIR, "world_building.json")

# Ensure the main output directory exists upon script load
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "chapters"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "chapter_logs"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "debug_outputs"), exist_ok=True)


# --- Generation Parameters ---
# Maximum number of characters to include in the context prompt for the LLM
MAX_CONTEXT_LENGTH: int = 64000 # Be mindful of model context window limits
# Default maximum number of tokens the LLM should generate in a response (can be overridden)
MAX_GENERATION_TOKENS: int = 30000 # General fallback; specific tasks might need more or less
# Maximum number of characters from a chapter to use for knowledge update prompts (character/world analysis)
KNOWLEDGE_UPDATE_SNIPPET_SIZE: int = 8000
# Number of most relevant past chapters to retrieve for semantic context generation
CONTEXT_CHAPTER_COUNT: int = 5
# Number of chapters to attempt writing in a single execution run
CHAPTERS_PER_RUN: int = 3 # Reduced for testing efficiency


# --- Agentic Planning ---
ENABLE_AGENTIC_PLANNING: bool = True # Flag to enable the new planning step
MAX_PLANNING_TOKENS: int = 16000 # Max tokens for the planning LLM call (might need more for detailed scenes)
# Limits for context snippets in planning prompts
PLANNING_CONTEXT_MAX_CHARS_PER_PROFILE_DESC: int = 100
PLANNING_CONTEXT_MAX_RECENT_DEV_PER_PROFILE: int = 150
PLANNING_CONTEXT_MAX_CHARACTERS_IN_SNIPPET: int = 5
PLANNING_CONTEXT_MAX_LOCATIONS_IN_SNIPPET: int = 3
PLANNING_CONTEXT_MAX_FACTIONS_IN_SNIPPET: int = 2
PLANNING_CONTEXT_MAX_SYSTEMS_IN_SNIPPET: int = 2


# --- Revision and Validation ---
# Cosine similarity threshold below which chapter coherence triggers revision
REVISION_COHERENCE_THRESHOLD: float = 0.65
# Flag to enable triggering revisions based on consistency check failures
REVISION_CONSISTENCY_TRIGGER: bool = True
# Flag to enable triggering revisions based on plot arc validation failures
PLOT_ARC_VALIDATION_TRIGGER: bool = True
# Similarity threshold for revision check (revisions less similar than this are accepted)
REVISION_SIMILARITY_ACCEPTANCE: float = 0.99 # Revisions scoring >= this are rejected as too similar
# Max tokens for specific LLM calls during evaluation/update
MAX_SUMMARY_TOKENS: int = 1500
MAX_CONSISTENCY_TOKENS: int = 4000
MAX_PLOT_VALIDATION_TOKENS: int = 1500
MAX_KG_TRIPLE_TOKENS: int = 8000 # For KG extraction from chapter text
MAX_PREPOP_KG_TOKENS: int = 16000 # For pre-populating KG from plot/world JSON

# --- Draft Evaluation ---
MIN_ACCEPTABLE_DRAFT_LENGTH: int = 4000 # Minimum character length for a generated draft

# --- Dynamic State Adaptation ---
ENABLE_DYNAMIC_STATE_ADAPTATION: bool = True # Allow LLM to propose modifications to profiles/world

# --- Knowledge Graph ---
KG_PREPOPULATION_CHAPTER_NUM: int = 0 # Chapter number for foundational KG data

# --- Embedding Configuration ---
# Expected dimension of the embeddings generated by EMBEDDING_MODEL
EXPECTED_EMBEDDING_DIM: int = 768
# NumPy data type to use for storing and processing embeddings
EMBEDDING_DTYPE: np.dtype = np.dtype(np.float32) # Explicitly np.dtype object
# Maximum number of embedding results to cache in memory (using LRU cache)
EMBEDDING_CACHE_SIZE: int = 128

# --- Logging ---
# Logging level for the application (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOG_LEVEL: str = "INFO"
# Format string for log messages
LOG_FORMAT: str = '%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
# Path for optional log file (set to None to disable file logging)
LOG_FILE: Optional[str] = os.path.join(OUTPUT_DIR, "saga_run.log")

# --- Plot Outline Generation Settings ---
# If True, genre, theme, setting, protagonist, conflict are randomized from lists below.
# If False, uses CONFIGURED_GENRE, CONFIGURED_THEME, CONFIGURED_SETTING, and LLM generates protagonist/conflict based on them.
UNHINGED_PLOT_MODE: bool = True # Set to True for randomized paradoxical combinations

# Used if UNHINGED_PLOT_MODE is False
CONFIGURED_GENRE: str = "hard science fiction"
CONFIGURED_THEME: str = "the nature of consciousness and isolation"
CONFIGURED_SETTING_DESCRIPTION: str = "A derelict deep-space research vessel adrift in the Kuiper Belt, with remnants of a long-dead human crew and a decaying technological infrastructure."
# Protagonist description and name will be generated by LLM based on the above in normal mode.
# If UNHINGED_PLOT_MODE is True, the LLM will still generate a specific protagonist_description based on the random archetype.

# Default protagonist name if plot outline is generated and LLM fails to provide one (both modes)
DEFAULT_PROTAGONIST_NAME: str = "Sága"
# Default title if plot outline is generated and LLM fails to provide one (both modes)
DEFAULT_PLOT_OUTLINE_TITLE: str = "Untitled Saga"

# --- Lists for Unhinged Plot Mode Randomization ---

# (Expand these lists for more variety)

UNHINGED_GENRES: List[str] = [
    "hard science fiction", "soft science fiction", "space opera", "military science fiction", "cyberpunk",
    "post-cyberpunk", "biopunk", "nanopunk", "solarpunk", "dieselpunk", "atompunk", "cassette futurism",
    "science fantasy", "alternate history", "retrofuturism", "time travel fiction", "apocalyptic fiction", "post-apocalyptic fiction",
    "dying earth", "generation ship fiction", "planetary romance", "first contact", "alien invasion", "climate fiction", "cli-fi",
    "technothriller", "afrofuturism", "silkpunk", "quantum fiction", "mundane science fiction", "space western", "high fantasy",
    "epic fantasy", "low fantasy", "portal fantasy", "urban fantasy", "contemporary fantasy", "historical fantasy", "mythic fiction",
    "dark fantasy", "heroic fantasy", "sword and sorcery", "fairy tale fantasy", "magical realism", "new weird", "weird fiction",
    "grimdark fantasy", "noblebright fantasy", "flintlock fantasy", "gaslamp fantasy", "arcanepunk", "elfpunk", "mannerpunk",
    "wuxia", "xianxia", "progression fantasy", "litrpg", "gamelit", "isekai", "arthurian fantasy", "celtic fantasy", "norse fantasy",
    "oriental fantasy", "sandalpunk", "mythpunk", "fabulism", "secondary world fantasy", "contemporary fairy tales", "gothic horror",
    "cosmic horror", "lovecraftian horror", "folk horror", "body horror", "psychological horror", "supernatural horror", "splatterpunk",
    "slasher", "monster fiction", "ghost stories", "vampire fiction", "werewolf fiction", "zombie fiction", "haunted house stories",
    "occult detective fiction", "religious horror", "quiet horror", "survival horror", "eco-horror", "medical horror", "found footage",
    "bizarro fiction", "new weird horror", "torture porn", "gothic romance", "cozy mystery", "hardboiled detective fiction", "noir fiction",
    "police procedural", "amateur sleuth", "historical mystery", "forensic thriller", "legal thriller", "medical thriller", "political thriller",
    "spy fiction", "espionage", "action thriller", "conspiracy fiction", "crime fiction", "heist story", "locked room mystery", "whodunit",
    "howdunit", "historical whodunit", "domestic thriller", "psychological thriller", "technothriller", "courtroom drama", "nordic noir",
    "tartan noir", "true crime", "contemporary romance", "historical romance", "regency romance", "gothic romance",
    "paranormal romance", "urban fantasy romance", "science fiction romance", "western romance", "medical romance",
    "sports romance", "second chance romance", "friends-to-lovers", "enemies-to-lovers", "fake relationship", "arranged marriage",
    "age gap romance", "forbidden love", "slow burn romance", "romantic comedy", "romantic suspense", "sweet romance",
    "erotic romance", "clean romance", "lgbtq+ romance", "multicultural romance", "dual timeline romance", "seasoned romance",
    "holiday romance", "dark romance", "reverse harem", "alternative history", "biographical historical fiction", "historical adventure",
    "historical fantasy", "historical mystery", "historical romance", "medieval fiction", "renaissance fiction", "tudor fiction",
    "regency fiction", "victorian fiction", "western fiction", "colonial fiction", "plantation fiction", "civil war fiction", "world war fiction",
    "ancient world fiction", "biblical fiction", "historical saga", "nautical fiction", "maritime fiction", "military historical fiction",
    "historical epic", "historical domestic fiction", "prehistoric fiction", "wuxia", "ancient egyptian fiction", "celtic historical fiction",
    "classical historical fiction", "samurai fiction", "pirate fiction", "literary fiction", "contemporary fiction", "bildungsroman",
    "coming-of-age", "campus novel", "domestic fiction", "epistolary fiction", "family saga", "feminist fiction", "lgbtq+ fiction",
    "magical realism", "metafiction", "modernist fiction", "new adult fiction", "philosophical fiction", "political fiction",
    "postmodern fiction", "psychological fiction", "realism", "social novel", "southern gothic", "transgressive fiction", "urban fiction",
    "autofiction", "hyperfiction", "maximalism", "minimalism", "mundane fiction", "slice-of-life", "upmarket fiction", "women's fiction",
    "literary thriller", "black comedy", "farce", "humor", "parody", "romantic comedy", "satire", "social satire", "political satire",
    "absurdist fiction", "absurdist comedy", "comedy of manners", "screwball comedy", "slapstick", "tragicomedy", "picaresque novel",
    "tall tale", "comic fantasy", "comic science fiction", "humorous memoir", "campus comedy", "workplace comedy", "action-adventure",
    "epic", "swashbuckler", "historical adventure", "maritime adventure", "military adventure", "survival story", "expedition fiction",
    "lost world fiction", "treasure hunt", "jungle adventure", "arctic adventure", "antarctic adventure", "desert adventure",
    "mountain climbing fiction", "safari fiction", "exploration fiction", "quest narrative", "travel fiction", "wilderness fiction",
    "colonial adventure", "nautical fiction", "pirate fiction", "absurdism", "anti-novel", "beat fiction", "ergodic literature", "experimental fiction",
    "hypertext fiction", "nouveau roman", "stream of consciousness", "surrealism", "visual novel", "interactive fiction", "concrete prose",
    "conceptual fiction", "dadaist fiction", "cut-up technique", "oulipian constraints", "pataphysical fiction", "flarf", "combinatorial literature",
    "multimedia fiction", "dystopian romance", "sci-fi mystery", "historical fantasy", "paranormal mystery", "cyberpunk noir",
    "steampunk adventure", "urban fantasy romance", "horror comedy", "mystical realism", "post-apocalyptic western", "science fantasy",
    "romantic suspense", "historical horror", "gothic science fiction", "techno-thriller romance", "fantasy detective", "comedic fantasy",
    "supernatural thriller", "historical science fiction", "literary horror", "mythological retelling", "flash fiction", "short story", "novella", "novel",
    "series", "serial", "trilogy", "duology", "quartet", "epic", "saga", "shared world", "anthology", "collection", "episodic fiction", "verse novel",
    "microfiction", "mosaic novel", "linked short stories", "afrofuturism", "african fiction", "asian american fiction", "caribbean fiction",
    "chicano literature", "indigenous futurism", "latin american fiction", "native american fiction", "postcolonial fiction", "australian outback fiction",
    "nordic noir", "mediterranean noir", "southern gothic", "western americana", "asian fantasy", "wuxia", "xianxia", "silkpunk", "african fantasy",
    "indigenous fantasy", "magical realism", "european fairy tale fiction", "middle eastern fantasy", "russian fairy tale fiction",
    "scandinavian crime fiction", "new weird", "slipstream", "interstitial fiction", "bizarro fiction", "cli-fi", "climate fiction", "hopepunk", "solarpunk",
    "noblebright", "grimdark", "isekai", "litrpg", "gamelit", "progression fantasy", "cozy fantasy", "rural fantasy", "mannerpunk", "dreampunk",
    "stonepunk", "clockpunk", "lunarpunk", "ecopunk", "decopunk", "rococopunk", "egyptpunk", "bronzepunk", "new adult", "domestic thriller",
    "up lit", "grip lit", "splatter western", "magical academia", "space archaeology", "arcanabiology", "allegorical fiction", "alternate history",
    "animal fiction", "anthropological fiction", "bildungsroman", "dystopian fiction", "environmental fiction", "epistolary fiction",
    "ethnographic fiction", "fictional autobiography", "fictional biography", "gothic fiction", "historiographic metafiction", "immortal fiction",
    "multiverse fiction", "musical fiction", "nautical fiction", "occupational fiction", "parallel universe fiction", "planetary fiction", "political allegory",
    "prehistoric fiction", "prison fiction", "psychological fiction", "religious fiction", "school story", "simulated reality fiction", "sports fiction",
    "theological fiction", "time slip fiction", "travel fiction", "utopian fiction", "virtual reality fiction", "war fiction", "workplace fiction", "bleak fiction",
    "cathartic fiction", "character-driven fiction", "comforting fiction", "contemplative fiction", "cynical fiction", "dramatic fiction", "emotional fiction",
    "escapist fiction", "feel-good fiction", "feminist fiction", "gritty fiction", "inspirational fiction", "intellectual fiction", "introspective fiction",
    "melancholic fiction", "morality tale", "nostalgic fiction", "plot-driven fiction", "poetic fiction", "reflective fiction", "sentimental fiction",
    "suspenseful fiction", "symbolic fiction", "thought-provoking fiction", "tragic fiction", "uplifting fiction", "whimsical fiction", "domestic fiction",
    "family drama", "intergenerational saga", "kitchen sink realism", "marriage fiction", "midlife crisis narrative", "ordinary people fiction",
    "quotidian fiction", "relationship fiction", "rural fiction", "suburban fiction", "urban fiction", "workplace fiction", "everyday magic",
    "gentle fiction", "quiet fiction", "slow fiction", "character study", "life transition fiction", "small town fiction", "steampunk", "cyberpunk noir",
    "magical realism", "mythological retelling", "space opera", "gothic romance", "urban fantasy", "post-apocalyptic western", "slice-of-life in a magical world"
 ]

UNHINGED_THEMES: List[str] = [
    "the nature of consciousness and isolation", "the burden of prophecy", "identity in a digital age", "the ethics of artificial intelligence", "the consequences of unchecked ambition", "love against societal norms", "the search for meaning in chaos", "redemption and forgiveness", "humanity's impact on nature", "the illusion of free will", "the power of memory", "sacrifice for the greater good", "the corrupting influence of power", "the duality of human nature",
    "coming of age and self-discovery", "confronting mortality", "the price of revenge", "found family and belonging", "loss of innocence", "the struggle against fate", "the contrast between appearance and reality", "cycles of violence", "intergenerational trauma", "the complexity of moral choices", "the cost of progress", "alienation in modern society", "the fragility of civilization", "the tension between tradition and change",
    "the pursuit of justice", "the impact of war", "the nature of evil", "the search for home", "acceptance of the other", "the conflict between individual and society", "the destructive power of greed", "the blurred line between hero and villain", "the weight of responsibility", "coping with grief and loss", "the masks we wear in society", "betrayal and its aftermath", "finding courage in adversity", "the consequences of pride",
    "the complexity of truth", "the search for authenticity", "the nature of love", "the reconciliation of past and present", "adapting to change", "the limits of human knowledge", "the relationship between freedom and security", "the danger of blind obedience", "the value of compassion", "family secrets and their impact", "technology's effect on humanity", "the consequences of prejudice", "the transformative power of art", "the influence of environment on character",
    "the quest for purpose", "the burden of leadership", "the contrast between urban and rural life", "the effects of isolation", "the price of conformity", "the persistence of hope", "the nature of friendship", "the cycle of life and death", "the conflict between reason and emotion", "the search for redemption", "the consequences of deception", "the challenge of forgiveness", "the importance of connection", "the impact of colonization",
    "the evolution of identity", "the cost of ambition", "the consequences of inaction", "the struggle for autonomy", "the complexity of loyalty", "the nature of heroism", "the impact of class divisions", "the pursuit of wisdom", "the effects of trauma", "the meaning of success", "the path to self-acceptance", "the consequences of fear", "the nature of courage", "the relationship between humanity and technology",
    "the burden of knowledge", "the consequences of vengeance", "the erosion of privacy", "the search for justice in an unjust world", "the conflict between duty and desire", "the nature of faith", "the transformative power of grief", "the complexity of forgiveness", "the struggle for equality", "the impact of secrets", "the price of freedom", "the consequences of obsession", "the evolution of relationships", "the danger of hubris",
    "the pursuit of enlightenment", "the nature of sacrifice", "the transformation through suffering", "the conflict between individual desire and communal good", "the consequences of cultural clash", "the search for truth", "the power of belief", "the impact of power on relationships", "the meaning of family", "the consequences of technological advancement", "the struggle against oppression", "the nature of belonging", "the cost of loyalty", "the pursuit of perfection",
    "the weight of legacy", "the consequences of climate change", "the nature of time", "the impact of aging", "the search for agency", "the complexity of empathy", "the consequences of addiction", "the nature of reality", "the effects of displacement", "the power of storytelling", "the impact of economic disparity", "the tension between security and risk", "the nature of identity", "the burden of expectation",
    "the consequences of cultural appropriation", "the search for beauty", "the nature of madness", "the impact of disability", "the complexity of parent-child relationships", "the consequences of genetic engineering", "the nature of community", "the effects of propaganda", "the pursuit of knowledge", "the impact of disease", "the tension between material and spiritual", "the nature of creativity", "the burden of guilt", "the consequences of imperialism",
    "the search for balance", "the nature of dreams", "the impact of surveillance", "the complexity of immigration experiences", "the consequences of nationalism", "the nature of freedom", "the effects of artificial scarcity", "the pursuit of happiness", "the impact of technology on privacy", "the tension between progress and preservation", "the nature of intimacy", "the burden of representation", "the consequences of warfare",
    "the search for utopia", "the nature of dystopia", "the impact of totalitarianism", "the complexity of historical memory", "the consequences of religious extremism", "the nature of democracy", "the effects of mass media", "the pursuit of equality", "the impact of globalization", "the tension between individualism and collectivism", "the nature of humor", "the burden of survival", "the consequences of corruption",
    "the search for reconciliation", "the nature of healing", "the impact of institutional racism", "the complexity of cultural identity", "the consequences of environmental destruction", "the nature of justice", "the effects of social media", "the pursuit of authenticity in a virtual world", "the impact of wealth disparity", "the tension between science and spirituality", "the nature of mortality", "the burden of immortality", "the consequences of isolation",
    "the search for sustainable existence", "the nature of human connection", "the impact of mental illness", "the complexity of sexuality", "the consequences of misinformation", "the nature of patriotism", "the effects of urbanization", "the pursuit of legacy", "the impact of historical revisionism", "the tension between logic and intuition", "the nature of beauty", "the burden of genius", "the consequences of overpopulation"
]

UNHINGED_SETTINGS_ARCHETYPES: List[str] = [
    "a derelict deep-space research vessel", "a magically-warded floating city above a poisoned wasteland",
    "a neon-drenched cyberpunk megalopolis where memories are currency", "an alternate Victorian London powered by volatile aether-tech",
    "a forgotten library at the edge of reality containing forbidden knowledge", "a whimsical dreamscape governed by illogical rules",
    "a post-apocalyptic desert patrolled by mutated creatures and desperate survivors", "a seemingly utopian underwater colony with a dark secret",
    "a medieval kingdom on the brink of an industrial revolution fueled by captured spirits",
    "a generation ship lost between stars, its inhabitants having forgotten their origin",
    "a hidden valley where time flows differently than the outside world", "a labyrinthine ancient city built on the back of a colossal sleeping creature",
    "a pocket dimension accessible only through mirrors at precisely midnight", "a bustling interspecies trading hub at the crossroads of multiple realms",
    "a sentient forest that communicates through pollen and spore patterns", "a sprawling university campus dedicated to the study of forbidden magic",
    "a crystal cavern network illuminated by bioluminescent organisms that respond to emotions", "a desert oasis that appears only during rare celestial alignments",
    "a perpetually stormy archipelago where lightning is harvested as the main power source", "a subterranean fungal empire ruled by a collective consciousness",
    "a once-grand space station now divided into feudal territories controlled by rival factions", "a frontier town on the edge of unexplored alien wilderness",
    "a nomadic caravan city that travels across an endless sea of sand", "a retrofitted asteroid mining colony repurposed as a criminal haven",
    "a massive arcology designed to sustain humanity after environmental collapse", "a series of floating islands connected by living bridges that shift and change daily",
    "a virtual reality construct where the boundaries between users and programs have blurred", "a monastery built into cliff faces that houses ancient technological artifacts",
    "a perpetual carnival that appears in different locations, collecting lost souls and broken dreams", "an endless library where each book contains a doorway to the world described inside",
    "a city district where gravity functions according to belief rather than physics", "a coastal settlement built from the salvaged remains of beached sea monsters",
    "a ghost town trapped in a time loop, repeating the day of its destruction", "a corporate skyscraper where each floor operates under different physical laws",
    "a mountain pass that serves as the threshold between the mortal realm and the afterlife", "a border town between two kingdoms with radically different magical systems",
    "an artificial ecosystem within a massive biodome designed as humanity's last refuge", "a massive tree city whose branches reach into different planes of existence",
    "a network of underground bunkers connected by tunnels housing the last remnants of humanity", "a city built within the skeleton of a fallen god or titan",
    "a remote island where evolution took a dramatically different path", "a research facility at the north pole studying a mysterious signal from beneath the ice",
    "a colony on the edge of a black hole where time dilation creates generational disparities", "a marketplace that exists for one day every century, trading items from across time",
    "a decrepit amusement park where the attractions have developed sentience", "a cloud city that harvests moisture and light, constantly drifting with the winds",
    "a hidden sanctuary where mythological creatures live in secret from modern society", "a massive wall that separates known civilization from uncharted territories",
    "a university town where every scientific experiment manifests physically in unpredictable ways", "a village built around a crater where meteorites carrying alien spores regularly fall",
    "a dimensional nexus disguised as an ordinary roadside diner", "a prison constructed to contain beings with reality-altering abilities",
    "a series of caves with paintings that come to life when touched", "a city that appears normal by day but transforms into something otherworldly at night",
    "an ancient temple complex that rearranges itself according to the stars' positions", "a megastructure constructed around a star to harness its energy, now partially abandoned",
    "a secluded valley where all technology ceases to function", "an artificial intelligence habitat where digital entities create their own culture",
    "a fleet of nomadic airships that never touch the corrupted ground below", "a massive living organism that humans have adapted into a habitable colony",
    "a seaside town where the ocean periodically reveals ancient ruins on the seafloor", "an orbital habitat specializing in zero-gravity agriculture and art",
    "a metropolis built around a massive time crystal that randomly sends districts into the past or future", "a moving city constructed on the back of an enormous mechanical beast",
    "a collection of houses that are bigger on the inside than physically possible", "a decaying space elevator connecting Earth to orbital colonies after a catastrophic event",
    "a jungle where plant and animal life has merged into hybrid conscious entities", "a sanctuary dimension where extinct species are preserved by interdimensional conservationists",
    "a massive neural network physically manifested as an explorable landscape", "a city where buildings are grown rather than constructed, adapting to residents' needs"
]

UNHINGED_PROTAGONIST_ARCHETYPES: List[str] = [
    "a newly self-aware AI", "a disgraced royal knight seeking redemption", "a cynical detective in a world of illusions",
    "an eccentric inventor with a dangerous creation", "a cursed scholar haunted by a cosmic entity",
    "an ordinary person thrust into an extraordinary, nonsensical situation", "a rebellious youth in a totalitarian regime",
    "a jaded historian uncovering a forgotten truth", "a demigod struggling with their divine heritage",
    "a stoic starship captain facing an impossible choice", "a reclusive artist whose creations come to life",
    "a street-smart rogue with a hidden heart of gold", "a lone wanderer in a desolate future",
    "a naive apprentice in a magical academy",
    "a retired assassin forced back into the game for one final mission", "a shapeshifter who can no longer remember their original form",
    "a time traveler stranded in an unfamiliar era", "a reluctant prophet burdened with apocalyptic visions",
    "a genetically engineered soldier questioning their purpose", "a professional dream-diver who becomes trapped in a client's nightmare",
    "a diplomat navigating the complex politics of warring alien species", "a formerly immortal being adjusting to mortal life",
    "a brilliant scientist racing against time to cure a pandemic they accidentally created", "a witch hunter who discovers they have magical abilities",
    "an amnesiac waking up with mysterious powers and unknown enemies", "a hardened survivor leading a community through post-apocalyptic dangers",
    "a spirit medium serving as a bridge between the living and the dead", "a deep-sea explorer who discovers an ancient underwater civilization",
    "an exiled noble disguised as a commoner plotting to reclaim their birthright", "a guild artisan whose crafts contain inadvertent enchantments",
    "a disillusioned priest questioning their faith after witnessing a miracle", "a retired hero called back to action when their legacy is threatened",
    "a dimension-hopping courier delivering packages across parallel realities", "a synthetic human searching for their original creator",
    "a paranormal investigator with a personal connection to the supernatural", "a monarch's food taster who develops immunity to poisons and political insight",
    "a former cult member trying to prevent others from falling into the same trap", "a guardian of a natural resource coveted by a powerful corporation",
    "a specialist in forgotten languages deciphering warnings from an ancient civilization", "a space archaeologist exploring the ruins of a vanished alien society",
    "a mind reader struggling to maintain their own identity amid others' thoughts", "a cartographer mapping uncharted territories with reality-altering properties",
    "a memory keeper in a society that practices mandatory memory erasure", "a retired soldier suffering from supernatural battle trauma",
    "a human raised by non-humans trying to understand their own heritage", "a talented forger who discovers they're unwittingly creating forgeries of magical artifacts",
    "a librarian protecting banned books in a world where knowledge is controlled", "a clone developing an identity separate from their original",
    "a once-famous explorer now believed dead embarking on a final expedition", "a reluctant heir to a criminal empire trying to legitimize the family business",
    "a court jester who serves as the true political power behind the throne", "a planetary terraformer with an unorthodox approach to creating habitable worlds",
    "a bounty hunter who only targets the corrupt elite", "a mediator with the rare ability to see all sides of any conflict",
    "a hermit with forgotten knowledge crucial to averting disaster", "a bodyguard assigned to protect someone they once swore to kill",
    "a smuggler who specializes in transporting dangerous forbidden technology", "a gifted empath in a society that values emotional detachment",
    "a plague doctor experimenting with unconventional cures during a magical epidemic", "a weather manipulator whose powers are tied to their emotional state",
    "a doppelganger who has lived so long as someone else they've forgotten who they really are", "a refugee with a cultural heritage that holds the key to solving a crisis",
    "a professional impersonator hired to replace a missing public figure", "a negotiator with a perfect track record facing an impossible diplomatic mission",
    "a non-human creature trying to pass as human for survival", "a memory thief who becomes haunted by stolen memories",
    "a former villain seeking atonement for past misdeeds", "a digital consciousness downloaded into an organic body",
    "a prodigy in a specialized field whose innovations threaten the established order", "a spy who's been undercover so long they've developed genuine loyalty to their target",
    "a beast tamer with a mysterious connection to dangerous creatures", "a musician whose compositions have hypnotic or magical effects",
    "a caretaker of an ancient site with reality-bending properties", "a translator for an enigmatic alien visitor with unclear intentions",
    "a scavenger who discovers a relic that marks them as a prophesied chosen one", "a mechanic who can intuitively understand and repair any technology",
    "a courier entrusted with delivering a message that could prevent or start a war", "a forensic necromancer who speaks to the dead to solve their murders",
    "a botanical engineer who creates plant hybrids with unexpected properties", "a former enemy soldier now living among the people they once fought against",
    "a conspiracy theorist who discovers one of their wildest theories is actually true", "an architect designing impossible structures that defy physical laws",
    "a reformed cultist using insider knowledge to rescue others from indoctrination", "a debt collector for a supernatural loan shark encountering desperate clients"
]

UNHINGED_CONFLICT_TYPES: List[str] = [
    "internal struggle with emerging sentience vs. programmed duty", "an ancient prophecy foretelling doom vs. the desire for free will",
    "a personal vendetta against a powerful corporation in a corrupt city", "a race against time to stop a technological marvel from causing catastrophe",
    "a desperate fight to maintain sanity against an encroaching otherworldly horror", "an attempt to find logic and purpose in a world that defies it",
    "a forbidden love that challenges the foundations of a rigid society", "a perilous quest to uncover a conspiracy that rewrites history",
    "a battle against mythical beasts and treacherous gods", "a galactic war threatening to consume civilizations",
    "a dark family secret that unravels a noble lineage", "a hidden war between magical factions in a mundane world",
    "a struggle for survival against environmental collapse and mutated foes",
    "the discovery of a terrible truth hidden beneath a perfect society's facade",
    "a moral dilemma that pits personal loyalty against greater good", "a competition for limited resources in a depleted world",
    "an investigation revealing that a revered institution is fundamentally corrupt", "a crisis of faith when sacred beliefs are challenged by new evidence",
    "a calculated revenge that threatens to consume the avenger", "a battle against a disease that transforms victims in terrifying ways",
    "a desperate negotiation with an alien intelligence that operates on incomprehensible logic", "a journey to restore balance to a world whose natural laws are unraveling",
    "a conflict between tradition and progress in a society at a technological turning point", "a struggle to preserve cultural identity under colonial oppression",
    "a fight against an adversary who knows your every weakness and future move", "a quest to break a generational curse affecting an entire bloodline",
    "a rebellion against an oppressive system that punishes individuality", "a mission to retrieve a stolen artifact that controls a fundamental force",
    "a battle of wits against a mastermind manipulating global events from the shadows", "a desperate attempt to prevent an ancient evil from awakening",
    "a struggle to establish communication with a non-human intelligence before conflict erupts", "a competition between rival factions to harness a new source of power",
    "a journey to reunite fractured realms before their separation becomes permanent", "a challenge to prove one's innocence in a system designed to presume guilt",
    "a fight to retain humanity while undergoing a transformative metamorphosis", "a mission to broker peace between species with fundamentally opposed biologies",
    "a race to solve an existential mystery before a countdown to extinction concludes", "a battle against an enemy that can manipulate memories and perception",
    "a struggle to maintain an ethical code in a world that rewards its abandonment", "a desperate defense against an invasive species that assimilates all life",
    "a conflict between divergent evolutionary paths of the same original species", "a battle against time as reality fragments into increasingly unstable shards",
    "a mission to recover lost knowledge crucial to surviving an imminent natural disaster", "a confrontation with a mirror self from an alternative timeline",
    "a challenge to survive in an environment with physical laws that randomly change", "a struggle against an artificial intelligence that believes it's acting in humanity's best interest",
    "a conflict between the responsibilities of power and personal desires", "a fight against a contagious idea that rewrites the infected's beliefs and goals",
    "a dilemma when the only solution to save many requires sacrificing the innocent", "a battle against an enemy that grows stronger from conventional attacks",
    "a desperate gambit to close a portal leaking chaotic energies into ordered reality", "a challenge to rebuild society after the collapse of a fundamental technology",
    "a mission to negotiate with forces of nature that have gained sentience and purpose", "a personal journey to overcome trauma that manifests as physical reality",
    "a conflict between multiple versions of oneself split across timelines", "a battle to reclaim identity after having memories stolen or altered",
    "a struggle to maintain psychological boundaries in a world where minds can merge", "a fight against a parasitic entity that grants power at the cost of gradually taking control",
    "a challenge to break a time loop without causing greater calamity", "a mission to translate an urgent warning from an extinct civilization",
    "a conflict resulting from a technological singularity that humans cannot comprehend", "a race to evacuate before an extinction-level event strikes",
    "a struggle against a reality-altering phenomenon that manifests collective fears", "a confrontation with a being that embodies a philosophical concept or natural law",
    "a conflict arising when development of a new ability disrupts social hierarchy", "a desperate attempt to prevent assimilation into a collective consciousness",
    "a battle between competing visions for the future of human evolution", "a challenge to solve an impossible crime where the laws of nature were broken",
    "a struggle to survive when the simulation housing reality begins to break down", "a conflict with entities that feed on specific emotions or mental states",
    "a moral crisis when discovering one's existence directly causes others' suffering", "a fight against a memetic hazard that spreads through knowledge of its existence",
    "a battle against opponents who can manipulate probability and luck", "a mission to repair the boundary between dimensions that are bleeding together",
    "a contest to claim a throne with powers tied to the ruler's personal qualities", "a struggle against an entity that exists outside normal causality",
    "a challenge to prevent a necessary but destructive evolutionary transition", "a fight to retain individual identity within a hive mind society",
    "a conflict resulting from a discovery that reality is fundamentally different than understood", "a mission to break a cycle of violence perpetuated by cultural memory",
    "a battle against weaponized information that reprograms minds", "a race to develop a counter to a technology that has upset power balances",
    "a conflict between natural and artificial beings competing for the same resources", "a challenge to preserve knowledge through a deliberate dark age",
    "a struggle against an adversary that can rewrite the rules of engagement", "a battle where victory requires embracing what one fears becoming",
    "a conflict created by attempting to change established historical events", "a mission to survive in a world where abstract concepts materialize physically",
    "a fight against corruption spreading from a central source through a connected system", "a challenge to maintain hope in a reality designed to cultivate despair",
    "a desperate attempt to awaken those trapped in a comfortable but false existence", "a moral dilemma when discovering one's heroes were secretly villains",
    "a battle against an opponent who grows stronger from conflict itself", "a struggle to forge a new path when all prophesied futures lead to ruin"
]

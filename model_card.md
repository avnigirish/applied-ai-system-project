# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **MoodTrack 1.0**  

- **MoodTrack 1.0** — a rule-based content recommender that matches songs to your current mood and energy level

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

Prompts:  

- What kind of recommendations does it generate  
- What assumptions does it make about the user  
- Is this for real users or classroom exploration  

- It generates a ranked list of up to 5 songs from a 19-song catalog based on genre, mood, energy, and acoustic preference
- It assumes the user can describe their taste in a single snapshot — one genre, one mood, one energy level
- This is for classroom exploration only, not for real users; the catalog is too small and the rules too simple for real use

---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.)  
- What user preferences are considered  
- How does the model turn those into a score  
- What changes did you make from the starter logic  

Avoid code here. Pretend you are explaining the idea to a friend who does not program.

- Each song has a genre, mood, energy level (0–1), and acousticness level (0–1)
- The user provides: favorite genre, current mood, target energy, and whether they like acoustic music
- The model awards points: +2.0 for a genre match, +1.0 for a mood match, up to +1.0 based on how close the song's energy is to the target, and up to +0.5 based on acoustic texture — max total is 4.5
- Songs are ranked from highest to lowest score; ties go to the song with energy closest to the user's target
- The starter logic used normalized weights summing to 1.0; this version uses whole-number points that are easier to reason about

---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset  

- 19 songs in `data/songs.csv`, no songs were added or removed
- 15 genres represented (pop, lofi, rock, ambient, jazz, synthwave, indie pop, r&b, metal, classical, hip-hop, folk, electronic, soul, country) — each appears exactly once
- 11 moods represented — most appear once or twice; happy, chill, and intense appear most often
- Missing entirely: Latin genres, K-pop, blues, reggae, punk, and most of the world's musical traditions
- The dataset skews heavily toward English-language Western popular music and electronic textures

---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  

- Works best when all preferences align — the High-Energy Pop profile returned Gym Hero (4.42/4.50) which felt genuinely correct: right genre, right mood, nearly identical energy
- Captures energy contrast well — a Chill Lofi listener and a Deep Intense Rock listener get completely different runner-up songs based on energy proximity alone, which matches intuition
- Every recommendation includes a plain-English reason (e.g., "genre match (+2.0), energy proximity (+0.99)"), making the logic transparent and easy to audit
- Handles missing genre gracefully — it doesn't crash, it just uses mood and energy as fallbacks

---

## 6. Limitations and Bias 

The most significant structural bias discovered during experiments is **genre lock-in**: because the catalog contains only one song per genre, a genre match automatically awards +2.0 points — 44% of the maximum score — with no competition. This means the system does not actually recommend music; it retrieves the single matching song and ranks everything else by energy proximity. Users whose preferred genre is missing from the catalog (e.g., bossa nova, reggae) are silently penalized, with their achievable score ceiling dropping from 4.5 to 2.5 and no explanation given in the output.

A second bias disadvantages **acoustic-preference users**: 14 of 19 songs in the catalog have acousticness below 0.5, skewing heavily toward electronic and produced textures. A user who sets `likes_acoustic=True` consistently receives lower acousticness proximity scores across nearly every recommendation, making the system subtly unfair to listeners who prefer organic or folk-style sound.

Finally, **mood matching uses exact string equality**, which creates invisible dead zones. Emotionally adjacent labels — `chill`, `relaxed`, and `focused` — are treated as completely different categories, so a user who wants `relaxed` gets zero mood credit for the two `chill` songs in the catalog even though those songs would likely satisfy them. Moods that appear only once (`romantic`, `nostalgic`, `angry`) offer almost no differentiation power, making mood a weak signal for roughly a third of the catalog.

---

## 7. Evaluation  

Six user profiles were tested: three standard listeners and three adversarial edge cases designed to stress-test the scoring logic.

**Standard profiles tested:**
- *High-Energy Pop* — a gym/workout listener wanting pop, intense mood, energy 0.92
- *Chill Lofi* — a studying listener wanting lofi, chill mood, energy 0.38, prefers acoustic
- *Deep Intense Rock* — a high-energy listener wanting rock, intense mood, energy 0.90

**Adversarial profiles tested:**
- *High Energy + Sad Mood* — conflicting signals: wants soul/sad but also energy 0.90
- *Unknown Genre* — requested "bossa nova," which does not exist in the catalog
- *Max Acoustic Preference* — classical/melancholic with a perfect energy match at 0.21

**What I looked for:** Whether the top results matched musical intuition, and whether the scoring logic behaved consistently when preferences conflicted or pointed toward missing catalog entries.

**What surprised me:** The adversarial "High Energy + Sad Mood" profile exposed the clearest weakness. Blue Porch — a quiet, slow soul ballad — ranked #1 because genre and mood together awarded +3.0 points, easily canceling out the energy mismatch penalty. Intuitively, no one who wants high-energy music should be recommended a slow ballad, but the system had no way to detect the conflict. The scores themselves looked reasonable on paper, which is what makes this bias hard to catch without testing.

The unknown genre test was also revealing: the system returned valid-sounding results with no error or warning, but every score was capped at 2.5 out of 4.5. Without knowing that the genre was missing, a real user would have no idea why the recommendations felt off.

---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  

- Add at least 5 songs per genre so genre match stops being a guaranteed winner and mood/energy have to do real differentiation work
- Replace binary mood matching with mood similarity clusters (e.g., chill/relaxed/focused as one group) so adjacent moods earn partial credit instead of zero
- Add a warning message when the user's requested genre doesn't exist in the catalog, instead of silently returning lower-scoring fallbacks
- Add tempo as a scored feature — two songs with the same energy can feel completely different at 60 BPM vs. 150 BPM

---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  

- The biggest learning moment was the adversarial test: Blue Porch ranked #1 for a high-energy user because genre+mood (+3.0) overpowered the energy penalty. The score looked fine on paper — that's what makes algorithmic bias hard to catch without actually testing edge cases.
- AI tools helped me write and iterate on the scoring logic quickly, but I had to run the system myself to discover that catalog size was the real problem. The AI could reason about weights in the abstract but couldn't predict how a 1-song-per-genre catalog would make the genre weight near-deterministic.
- What surprised me most was how much the output "feels" like a recommendation even though it's just four arithmetic operations. When Gym Hero ranked #1 for the gym profile, it genuinely felt intentional — the illusion comes from picking the right features to measure, not from any complexity in the algorithm.
- If I extended this project, I'd add tempo scoring, expand the catalog to 100+ songs, and try a simple collaborative filter on top: "users who liked X also liked Y."

---

## 10. AI Collaboration

How AI tools were used during development, including where they helped and where they fell short.

**One instance where the AI gave a genuinely helpful suggestion:**

When building the natural-language input mode, I initially planned to write a regex-based parser. The AI suggested using Gemini's function-calling API instead, where the model is forced to populate a typed JSON schema rather than return free text. This turned out to be the right call — it eliminated an entire category of output-validation code. The function-calling constraint means Gemini can't return a malformed preference object: either it fills in all four required fields with the right types, or the call fails with a clear error. That's a meaningfully better design than parsing unstructured text.

**One instance where the AI's suggestion was flawed:**

When I first ran `python -m src.main`, the imports broke with `ModuleNotFoundError: No module named 'recommender'`. The AI had written the imports as bare names (`from recommender import ...`) — which works when you run the file directly from inside `src/`, but fails when Python is invoked from the project root, because `python -m src.main` adds the root directory to `sys.path`, not `src/`. The fix was one line (`sys.path.insert(0, os.path.dirname(__file__))`), but the original suggestion assumed a specific working directory without making that assumption explicit or testing it. The AI could reason about the code but couldn't anticipate how the module system would behave across different invocation contexts — that required running it and observing the failure.

**What this taught about working with AI tools:**

AI assistance was most valuable for generating boilerplate, reasoning about abstract tradeoffs, and suggesting architectural patterns (like function-calling over regex parsing). It was least reliable for anything that required running the code and observing system behavior — import paths, working directory assumptions, and catalog-size effects on scoring all required hands-on testing to surface. The AI could describe what the code *should* do; only running it revealed what it actually did.

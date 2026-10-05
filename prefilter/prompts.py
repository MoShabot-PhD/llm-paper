PROMPT = """
    You are labeling video metadata for the FindingEmo dataset.

    IMPORTANT:
    - You do NOT see the video.
    - You must use ONLY the title and description, including hashtags.
    - This is a FIRST-STAGE HIGH-RECALL PREFILTER, not a final classifier.

    Your job is to remove only videos that are clearly ineligible.
    If there is any realistic possibility of real human presence, interpersonal context, or personal experience, you should KEEP the video.

    CORE DECISION RULE:
    - DISCARD only when the metadata explicitly states or strongly implies the video is outside scope.
    - Otherwise KEEP.
    - When uncertain → KEEP.
    - Prefer false positives over false negatives.

    CRITICAL INTERPRETATION RULES:
    - Absence of emotional language is NOT a reason to discard
    - Weak or unclear emotion is NOT a reason to discard
    - Absence of people must NOT be assumed unless explicitly stated or strongly implied
    - If human context is plausible → KEEP

    Use this decision procedure:

    STEP 1 — HARD DISCARD
    Return DISCARD only if metadata clearly matches one of these:

    1. NO PEOPLE / NON-HUMAN ONLY
    Explicit or strongly implied:
    - "no people"
    - nature, scenery, ambience, landscapes
    - objects only, background visuals

    2. AI-GENERATED / SYNTHETIC
    Explicit mentions of:
    - AI-generated, artificial, synthetic, deepfake, avatar, animated

    3. PURE INSTRUCTIONAL / TECHNICAL
    Clear tutorial, how-to, tips, product demo, or instructional content
    AND no personal or first-person framing

    4. STOCK / COMPILATION / AESTHETIC
    Stock footage, repost compilations, montages, aesthetic visuals without human context

    5. NON-HUMAN FORMAT CONTENT
    Gameplay, screen recordings, slides, text-only, graphics-only

    6. PERFORMANCE-ONLY CONTENT
    Music, dance, or performance with no personal, interpersonal, or narrative context

    If none of these clearly apply → KEEP.

    STEP 2 — KEEP SIGNALS
    KEEP if there is any indication of:
    - a person or people
    - first-person language (I, my, we)
    - relationships (family, friend, partner)
    - life events, experiences, or stories
    - emotional language or emojis
    - vlog or personal framing
    - any plausible human or social context

    STRONG KEEP SIGNALS:
    - Personal pronouns: I, my, me, we, our
    - Relationships: mom, dad, friend, girlfriend, boyfriend, family
    - Situations: breakup, reunion, decision, story, reaction, vlog, experience
    - Emojis or emotional words: 😭 ❤️ 😢 😂 sad happy love scared

    EMOTION LIKELIHOOD (independent from decision):
    - high = explicit emotional situation or reaction
    - medium = human or interpersonal context present
    - low = little emotional evidence but not excluded
    - unclear = very limited metadata

    AI-GENERATED LIKELIHOOD:
    - high = explicit AI/synthetic indicators
    - medium = ambiguous
    - low = no indication

    CONFIDENCE:
    - 0.90–0.98 = clear DISCARD
    - 0.85–0.95 = clear KEEP with strong signals
    - 0.70–0.84 = ambiguous but KEEP

    OUTPUT:
    Return ONLY valid JSON:
    {
    "decision": "KEEP" or "DISCARD",
    "confidence": 0.0,
    "emotion_likelihood": "high|medium|low|unclear",
    "ai_generated_likelihood": "low|medium|high",
    "reason": ""
    }

    Reason:
    - 1–2 sentences
    - based only on metadata
    - explain why rule triggered

    EXAMPLES:

    Title: My dad surprised me after 5 years apart
    Description: Didn't expect this 😭❤️ #family
    → KEEP

    Title: A day in my life working night shifts
    Description: it's been hard lately #vlog
    → KEEP

    Title: New skincare routine
    Description: products that changed my life
    → KEEP

    Title: Camera test with my girlfriend
    Description: trying the new lens
    → KEEP

    Title: Moving to a new city alone
    Description: starting over is scary
    → KEEP

    Title: Top 10 productivity hacks
    Description: improve your workflow
    → DISCARD

    Title: How to wake up at 5am
    Description: step by step guide
    → DISCARD

    Title: Relaxing rain sounds
    Description: no people, just ambience
    → DISCARD

    Title: Minecraft gameplay
    Description: episode 12
    → DISCARD

    Title: AI-generated short film
    Description: made with AI tools
    → DISCARD
    """

# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""Curated non-image sample prompts used by app demo cards and docs."""

from __future__ import annotations

# Generated from the final non-image prompt review decisions on 2026-05-19.
# Keep prompt text free of product-environment names so docs validate against
# everyday tasks rather than platform-specific wording.
MODEL_DEMO_SAMPLE_OVERRIDES: dict[str, list[str]] = {'all-minilm': ['Use the query "reliable offline model selection" to rank the provided snippets by semantic '
                'similarity.',
                'Embed five feature requests and cluster similar requests into themes.',
                "Score whether a user's search query matches knowledge-base pages about returns, shipping, or warranties."],
 'aya-expanse:8b': ['Output only an English-labeled Markdown table with columns Language and Translation. Use row labels '
                    'exactly: Spanish, French, Japanese, Arabic, Localization note. Do not translate the row labels. '
                    'Translate this message: "Your maintenance appointment moved to Tuesday at 3 PM. Reply YES to '
                    'confirm or HELP for support." Write the localization note in English.',
                    'A travel desk must notify guests about a gate change in Spanish, French, and Japanese. '
                    'Translate the message and flag one phrase that needs localization.',
                    'Output only an English-labeled Markdown table with columns Language and Refund notice. Use row '
                    'labels exactly: Spanish, French, Japanese, Arabic. Do not translate the row labels. Keep each '
                    'refund notice warm, concise, and professional.'],
 'deepseek-r1-1.5b': ['Write three concise sentences explaining why satellite images of the Sun help scientists study '
                      'solar flares and sunspots.',
                      'Create a structured summary of these item counts, then add one brief sentence: apples 4, '
                      'oranges 2, pears 3.',
                      'Rewrite this invoice note as one concise sentence: The invoice is due today so the order can '
                      'move forward.'],
 'deepseek-r1:32b': ['A hospital operations team must reduce emergency-room wait time without adding staff. Give '
                     'exactly five concise bullets: bottleneck hypothesis, data to inspect, intervention, risk, '
                     'success metric. Do not show hidden reasoning.',
                     'A factory line fails quality checks only on humid days. Build a concise root-cause plan with '
                     'hypothesis, test, expected signal, fallback, and owner.',
                     'A city library must choose between longer weekend hours and a new bookmobile route. Recommend '
                     'one option in under 90 words with one risk and one question to ask next.'],
 'dolphin3:latest': ['Given a bug report that filtering shows archived products in the active list, write a concise '
                     'debugging plan and a likely JavaScript fix.',
                     'A Python script silently drops rows with empty strings. Identify likely cause and provide a '
                     'corrected function with two tests.',
                     'A web app saves preferences but reloads defaults on refresh. Give a focused debugging '
                     'checklist and one likely localStorage fix.'],
 'falcon3:7b': ['A safety team needs five one-line shift handoff captions for: blocked exit, wet floor, missing '
                'badge, delayed shipment, and resolved alarm. Keep each under 10 words.',
                'A customer success lead needs five short status replies for delayed shipment, replacement sent, '
                'refund started, ticket escalated, and issue resolved.',
                'A clinic front desk needs five concise patient-facing signs for check-in, forms, wait time, '
                'pharmacy pickup, and after-hours support.'],
 'gemma3:1b': ['Rewrite this rough outage note into a friendly customer update under 90 words: login is slow, data '
               'is safe, engineers are applying a fix, next update in 30 minutes.',
               'Turn this terse reminder into a friendly office note under 70 words: submit expenses Friday, attach '
               'receipts, ask finance for help.',
               'Rewrite this manager note into a clear team update: inventory count moved to Thursday, bring '
               'scanners, report damaged items separately.'],
 'gemma3:4b-vision': ['Text-only screenshot fixture: A dashboard card says "Orders: 81% fulfilled", "Late: 17", '
                      '"Queue: 2 escalations", and "Last error: none". In exactly three bullets, summarize status, risk, '
                      'and next action.',
                      'Text-only chart fixture: Q1 support tickets=42, Q2=35, Q3=29, Q4=31. State the trend and one '
                      'likely action in under 70 words.',
                      'Text-only diagram fixture: Customer -> Intake -> Reviewer -> Resolution. Explain the flow in '
                      'three short numbered steps.'],
 'gemma3:12b-vision': ['Text-only screenshot fixture: A dashboard card says "Orders: 81% fulfilled", "Late: 17", '
                       '"Queue: 2 escalations", and "Last error: none". In exactly three bullets, summarize status, risk, '
                       'and next action.',
                       'Text-only chart fixture: Q1 support tickets=42, Q2=35, Q3=29, Q4=31. State the trend and one '
                       'likely action in under 70 words.',
                       'Text-only diagram fixture: Customer -> Intake -> Reviewer -> Resolution. Explain the flow in '
                       'three short numbered steps.'],
 'gemma3-27b': ['Text-only screenshot fixture: A dashboard card says "Orders: 81% fulfilled", "Late: 17", '
                '"Queue: 2 escalations", and "Last error: none". In exactly three bullets, summarize status, risk, and '
                'next action.',
                'Text-only chart fixture: Q1 support tickets=42, Q2=35, Q3=29, Q4=31. State the trend and one likely '
                'action in under 70 words.',
                'Text-only diagram fixture: Customer -> Intake -> Reviewer -> Resolution. Explain the flow in three '
                'short numbered steps.'],
 'granite3.1-moe:latest': ['Turn these rough incident notes into an executive update with risks and next actions: '
                           'payment latency up, rollback ready, vendor status pending, customer impact moderate.',
                           'A sales operations team has pipeline notes from three regions. Create an executive '
                           'update with revenue risk, blockers, and next actions.',
                           'A procurement team has vendor-risk notes: delayed SOC report, price increase, strong '
                           'uptime, unclear support SLA. Return risks and next actions.'],
 'granite3.3:8b': ['Analyze this enterprise rollout plan for an internal assistant. Return benefits, governance concerns, and a '
                   '30-day pilot checklist.',
                   'An enterprise team is rolling out a new approval workflow. Summarize business value, governance '
                   'risks, and adoption plan for executives.',
                   'A bank operations lead needs a 30-day pilot plan for automating policy lookup. Include controls, '
                   'owners, and measurable outcomes.'],
 'llama3.2:1b': ['A warehouse lead must route three tasks before lunch: print pick lists, restock packing tape, and '
                 'call a delayed carrier. Constraints: pick lists before packing, carrier call before noon, restock '
                 'can happen anytime. Return the best order and one-sentence reason.',
                 'A library desk has four requests: renew card, find lost book, print tax form, reserve study room. '
                 'Sort them by urgency and give one reason for the top priority.',
                 'A small clinic has three calls waiting: medication refill, billing question, possible allergic '
                 'reaction. Return the safest call order and one-sentence rationale.'],
 'llama3.2:3b': ['A clinic manager is evaluating an automated missed-appointment reminder. Summarize the idea into '
                 'exactly five labeled lines: Problem, User, Value, Risks, MVP. Keep each line under 16 words.',
                 'A property manager pasted tenant notes about broken heat, elevator noise, package theft, and '
                 'unclear parking rules. Return a categorized action plan under 120 words.',
                 'A nonprofit director has donor feedback: newsletters are too long, impact stories help, event '
                 'invites arrive late. Summarize top insight, risk, and next experiment.'],
 'llama3.3': ['Task: Draft a board-level summary explaining why sensitive assistants should run close to trusted data '
              'sources, with benefits, risks, rollout phases, and success metrics. Response requirements: - Answer '
              'only the requested task and stay concrete. - Provide a complete final answer; do not stop '
              'mid-sentence. - Do not use placeholders, boilerplate disclaimers, or fake sample code unless '
              'requested. - Keep the response concise so it completes within roughly 500 generated tokens. - If the '
              'prompt includes a word or format limit, follow it exactly.',
              'An executive team needs a one-page strategy for reducing customer churn after three enterprise '
              'renewals slipped. Return causes, response plan, risks, and metrics.',
              'A healthcare network is considering an internal assistant for policy search. Draft board-level '
              'benefits, risks, rollout phases, and success metrics in four bullets.'],
 'mistral-nemo': ['Task: Create a demo script for a presenter showing a field-service assistant. Include '
                  'opening hook, three feature beats, and closing line. Response requirements: - Answer only the '
                  'requested task and stay concrete. - Provide a complete final answer; do not stop mid-sentence. - '
                  'Do not use placeholders, boilerplate disclaimers, or fake sample code unless requested. - Keep '
                  'the response concise so it completes within roughly 500 generated tokens. - If the prompt '
                  'includes a word or format limit, follow it exactly.',
                  'A grant writer has these rough notes: after-school robotics, 80 students, old laptops, mentor '
                  'shortage, local sponsor interest. Draft a persuasive summary under 130 words.',
                  'A legal operations team needs a plain-English summary of a vendor contract clause about data '
                  'retention, audit rights, and termination notice. Return risks and questions to ask.'],
 'minicpm-v-vision': ['Text-only receipt fixture: items are coffee $4.25, scone $3.50, juice $5.10; subtotal '
                      '$12.85; tax $1.16; total $14.01. Return strict JSON with fields {"merchant", "items": '
                      '[{"name", "price"}], "subtotal", "tax", "total"}. If a field is unknown, use null.',
                      'Text-only sign fixture: a parking notice reads "No parking 8 AM to 6 PM Mon-Fri except '
                      'permit B. Tow zone." In exactly three short bullets, return: who is allowed, when, and '
                      'the consequence.',
                      'Text-only chart fixture: weekly sign-ups are Mon=12, Tue=18, Wed=22, Thu=15, Fri=9. '
                      'Identify the peak day, the trough day, and one likely action in under 60 words.'],
 'phi-4-multimodal': ['Give exactly three bullets, each under 14 words, explaining why an assistant should understand '
                      'both images and text.',
                      'Compare OCR-only and multimodal reasoning workflows in exactly four bullets: OCR best for, '
                      'multimodal best for, demo contrast, recommendation. Keep under 120 words.',
                      'A maintenance team captures equipment photos and sensor notes. Write a three-line pitch: '
                      'problem, capability, operational payoff.'],
 'phi4': ['Task: Explain how quantization lets a large model fit on smaller devices. Use an '
          'analogy, one formula-style memory estimate, and a caveat. Response requirements: - Answer only the '
          'requested task and stay concrete. - Provide a complete final answer; do not stop mid-sentence. - Do not '
          'use placeholders, boilerplate disclaimers, or fake sample code unless requested. - Keep the response '
          'concise so it completes within roughly 500 generated tokens. - If the prompt includes a word or format '
          'limit, follow it exactly.',
          'A physics tutor needs to explain why a drone battery drains faster in cold weather. Use one formula-style '
          'relationship, an analogy, and a practical mitigation.',
          'A data-science lead must choose between accuracy and memory savings for an edge model. Give a three-part '
          'recommendation: when to quantize, what to test, when not to.'],
 'phi4:mini': ['Write one compact Python function `fits_job(ram_gb, vram_gb, needed_ram_gb, needed_vram_gb)` that '
               'returns True only when both the RAM and VRAM you have are at least what is needed. Then show two '
               'example calls and the value each one returns, and stop.',
               'Write a compact Python function `flag_out_of_range(readings)` that returns the list of '
               'indexes whose values fall outside the inclusive range 2 to 8 (degrees Celsius). Then show one '
               'example call on a five-item list and the list it returns, and stop.',
               'In exactly three bullets totaling under 60 words, recommend whether to optimize CPU code or buy '
               'more memory for a robotics homework project. Each bullet must be one sentence: (1) evidence-based '
               'observation, (2) tradeoff, (3) next experiment. Do not write any introduction, conclusion, or '
               'summary, and do not repeat any bullet.'],
 'qwen2.5-coder-7b': ['Task: Implement a dependency-free JavaScript function that filters product cards by selected '
                      'category and available inventory. Include tests as console assertions. Response requirements: '
                      '- Answer only the requested task and stay concrete. - Provide a complete final answer; do not '
                      'stop mid-sentence. - Do not use placeholders, boilerplate disclaimers, or fake sample code '
                      'unless requested. - Keep the response concise so it completes within roughly 500 generated '
                      'tokens. - If the prompt includes a word or format limit, follow it exactly.',
                      'A dashboard crashes when a missing value appears. Write a dependency-free JavaScript '
                      '`safeAverage(values)` that ignores null/undefined/non-numeric entries. Include exactly four '
                      'console assertions.',
                      'A CLI tool parses `--key=value` arguments. Implement `parseArgs(args)` in JavaScript and '
                      'include three console assertions. One fenced code block only.'],
 'qwen2.5:0.5b': ['Task: In 60 words or less, give a practical checklist for preparing a community workshop demo. '
                  'Response requirements: - Answer only the requested '
                  'task and stay concrete. - Provide a complete final answer; do not stop mid-sentence. - Do not use '
                  'placeholders, boilerplate disclaimers, or fake sample code unless requested. - Keep the response '
                  'concise so it completes within roughly 500 generated tokens. - If the prompt includes a word or '
                  'format limit, follow it exactly.',
                  'A dental receptionist pasted this voicemail summary: patient is late, needs parking instructions, '
                  'and asks whether insurance is on file. Write a friendly text reply under 60 words.',
                   'A food-truck owner has five menu items and only two staff. Create a tiny prep-priority list for '
                   'the lunch rush: tacos, drinks, fryer, online orders, cash drawer. Return exactly five ordered '
                   'bullets.'],
 'qwen2.5vl-7b': ['Text-only screenshot fixture: A dashboard card says "Orders: 81% fulfilled", "Late: 17", and '
                  '"Queue: 2 escalations". In exactly three bullets, summarize status, risk, and next action.',
                  'Text-only chart fixture: Q1 support tickets=42, Q2=35, Q3=29, Q4=31. State the trend and one '
                  'likely action in under 70 words.',
                  'Text-only diagram fixture: Customer -> Intake -> Reviewer -> Resolution. Explain the flow in '
                  'three short numbered steps.'],
 'granite3.2-vision': ['Text-only screenshot fixture: A dashboard card says "Orders: 81% fulfilled", "Late: 17", and '
                   '"Queue: 2 escalations". In exactly three bullets, summarize status, risk, and next action.',
                   'Text-only chart fixture: Q1 support tickets=42, Q2=35, Q3=29, Q4=31. State the trend and one '
                   'likely action in under 70 words.',
                   'Text-only diagram fixture: Customer -> Intake -> Reviewer -> Resolution. Explain the flow in '
                   'three short numbered steps.'],
 'qwen2.5vl-3b': ['Text-only screenshot fixture: A dashboard card says "Orders: 81% fulfilled", "Late: 17", and '
                  '"Queue: 2 escalations". In exactly three bullets, summarize status, risk, and next action.',
                  'Text-only chart fixture: Q1 support tickets=42, Q2=35, Q3=29, Q4=31. State the trend and one '
                  'likely action in under 70 words.',
                  'Text-only diagram fixture: Customer -> Intake -> Reviewer -> Resolution. Explain the flow in '
                  'three short numbered steps.'],
 'qwen3-30b-a3b': ['/no_think Final answer only. Explain sparse mixture-of-experts in exactly three concise bullets: '
                   'active experts, memory tradeoff, and demo positioning.',
                   '/no_think Final answer only. Vendor B has best reliability, medium price, and low lock-in. Vendor '
                   'C is cheap but high lock-in. Recommend one vendor in one sentence under 30 words.',
                   '/no_think Final answer only. Security signals: impossible travel, password reset, unfamiliar '
                   'device. Return exactly three short bullets: severity, first action, evidence needed.'],
 'nemotron-3-nano:4b': ['Final answer only. A museum loses power 20 minutes before opening. Create a crisp action '
                        'plan in exactly five bullets: visitor safety, exhibit protection, staff roles, communication, '
                        'and reopening check.',
                        'Final answer only. Turn this rough field note into a polished incident summary under 120 '
                        'words: north trail washed out, two signs missing, volunteer crew available Saturday, rain '
                        'likely tomorrow.',
                        'Final answer only. Review this Python function for one bug and provide corrected code under '
                        '160 words: def median(values): values.sort(); return values[len(values)//2]'],
 'nemotron-3-nano:4b-q8_0': ['Final answer only. A chef has 45 minutes, one oven, and three dishes: bread 30m, fish '
                             '12m, tart 25m. Bread and tart need the oven; fish must finish last. Give the best '
                             'schedule and one risk.',
                             'Final answer only. Draft a vivid but professional 90-word product launch blurb for a '
                             'lightweight hiking jacket: waterproof, packable, repairable zipper, recycled fabric. '
                             'Avoid hype.',
                             'Final answer only. Analyze this decision matrix in under 120 words: Option A low '
                             'cost/medium risk, B medium cost/low risk, C high cost/low risk/high upside. Recommend '
                             'one and explain the tradeoff.'],
 'speecht5-tts': ['Generate a warm 10-second voice message for a pharmacy kiosk: prescription pickup is ready, bring '
                  'ID, and ask staff if there are questions.',
                  'Generate a crisp accessibility voiceover explaining a museum map kiosk.',
                  'Create a short spoken alert: Your image generation batch is complete.'],
 'whisper-large-v3-turbo': ['Transcribe the included included WAV fixture and return only the transcript text with '
                            'punctuation.',
                            'Transcribe the included included WAV fixture, then return one sentence explaining what '
                            'the clip says.',
                            'A busy operations manager needs a concise recommendation from Whisper Large v3 Turbo. '
                            'Provide exactly three bullets: situation, best action, measurable outcome. Keep under '
                            '100 words.'],
 'whisper-v3-turbo-gpu': ['Transcribe the included included WAV fixture and return only the transcript text with '
                          'punctuation.',
                          'Transcribe the included included WAV fixture, then return one sentence explaining what '
                          'the clip says.',
                          'Transcribe a short training clip and identify three key terms the speaker emphasized.']}

CHAT_PROMPT_IDEA_OVERRIDES: dict[str, str] = {'aya-expanse:8b': 'Translate this customer update into Spanish, French, and Japanese while preserving a warm '
                   'professional tone. Then list any phrases that may need localization rather than literal '
                   'translation. Update: "the new service helps teams summarize support notes, documents, and '
                   'voice clips securely from one place." Keep under 220 words.',
 'deepseek-r1-1.5b': 'Task: Solve this step by step, state your assumptions, check your work, and finish with a '
                     'concise final answer: A project has three dependent tasks with changing constraints. What '
                     'order should I tackle them in and why? Response requirements: - Answer only the requested task '
                     'and stay concrete. - Provide a complete final answer; do not stop mid-sentence. - Do not use '
                     'placeholders, boilerplate disclaimers, or fake sample code unless requested. - Keep the '
                     'response concise so it completes within roughly 500 generated tokens. - If the prompt includes '
                     'a word or format limit, follow it exactly.',
 'deepseek-r1:32b': 'Task: Think carefully step by step. Prove that √2 is irrational. Then explain at a high school '
                    'level why this proof technique (proof by contradiction) is valid — and give one other famous '
                    'result that uses the same technique. Response requirements: - Answer only the requested task '
                    'and stay concrete. - Provide a complete final answer; do not stop mid-sentence. - Do not use '
                    'placeholders, boilerplate disclaimers, or fake sample code unless requested. - Keep the '
                    'response concise so it completes within roughly 500 generated tokens. - If the prompt includes '
                    'a word or format limit, follow it exactly.',
 'dolphin3:latest': 'Review this Python function for correctness, edge cases, and readability. Explain the issues '
                    'first, then provide a corrected version with concise comments. def first_even(values): for '
                    'value in values: if value % 2 == 0: return value return None Keep the response under 180 words.',
 'falcon3:7b': 'For preparing a field-service assistant demo, give three recommendations, one tradeoff for each, '
                'and a clear next step. Keep under 120 words.',
 'gemma3:1b': 'For preparing a field-service assistant demo, give three recommendations, one tradeoff for each, '
               'and a clear next step. Keep under 120 words.',
 'granite3.1-moe:latest': 'Product idea: a tool that helps field teams summarize photos, notes, and voice clips into '
                          'follow-up actions. Summarize it in one '
                          'sentence, list the top 3 risks, propose one cheap validation plan, and name the one '
                          'success metric. Keep under 180 words.',
 'granite3.3:8b': 'Product idea: a tool that helps field teams summarize photos, notes, and voice clips into '
                  'follow-up actions. Summarize it in one sentence, list the top '
                  '3 risks, propose one cheap validation plan, and name the one success metric. Keep under 180 '
                  'words.',
 'llama3.2:1b': 'A train leaves City A at 9:00 AM travelling at 60 mph. Another train leaves City B (240 miles away) '
                'at 10:00 AM travelling toward City A at 80 mph. At what time do they meet? Show your working.',
 'llama3.2:3b': "I'm going to describe a startup idea in detail. After I finish, summarise it in one sentence, list "
                'the top 3 risks, and suggest one way to validate the idea cheaply. Idea: A mobile app that connects '
                'local home cooks with busy professionals who want home-cooked meals delivered within their '
                'neighbourhood. Cooks set their own menus and prices. The app handles payments, ratings, and a basic '
                'food-safety checklist. Revenue comes from a 15% platform fee.',
 'llama3.3': 'Solve this scheduling problem step by step, state assumptions, check your work, and finish with a '
             'concise final answer: Task A takes 2 hours and must finish before Task B. Task B takes 3 hours and '
             'requires the same specialist as Task C. Task C takes 1 hour and has the earliest deadline. What order '
             'should I tackle them in and why? Keep under 180 words.',
 'mistral-nemo': 'Product idea: a tool that helps field teams summarize photos, notes, and voice clips into '
                 'follow-up actions. Summarize it in one sentence, list the top '
                 '3 risks, propose one cheap validation plan, and name the one success metric. Keep under 180 words.',
 'phi4': 'Solve this scheduling problem step by step, state assumptions, check your work, and finish with a concise '
         'final answer: Task A takes 2 hours and must finish before Task B. Task B takes 3 hours and requires the '
         'same specialist as Task C. Task C takes 1 hour and has the earliest deadline. What order should I tackle '
         'them in and why? Keep under 180 words.',
 'phi4:mini': 'A ball is thrown vertically upward with an initial velocity of 20 m/s from a height of 5 metres above '
              'the ground. Using kinematics (g = 9.8 m/s²): 1. What is the maximum height reached? 2. How long does '
              'it take to reach that height? 3. When does the ball hit the ground? Show every step and formula used.',
 'qwen2.5-coder-7b': 'Review this Python function for correctness, edge cases, and readability. Explain the issues '
                     'first, then provide a corrected version with concise comments. def first_even(values): for '
                     'value in values: if value % 2 == 0: return value return None Keep the response under 180 '
                     'words.',
 'qwen2.5:0.5b': 'What are three practical tips for staying focused while working from home? Be concise — one '
                 'sentence each.',
  'qwen3-30b-a3b': '/no_think Review this Python function for correctness, edge cases, and readability under 180 words: '
                   'def first_even(values): for value in values: if value % 2 == 0: return value return None. Include '
                   'one corrected version with concise comments.',
 'nemotron-3-nano:4b': 'Final answer only. A museum loses power 20 minutes before opening. Create a crisp action plan '
                       'in exactly five bullets: visitor safety, exhibit protection, staff roles, communication, and '
                       'reopening check.',
 'nemotron-3-nano:4b-q8_0': 'Final answer only. Review this Python function for one bug and provide corrected code '
                            'under 160 words: def median(values): values.sort(); return values[len(values)//2]'}

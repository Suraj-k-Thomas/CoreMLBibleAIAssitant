//
//  ArchitectureDiagram.swift
//  BibleAIAssistant
//
//  Created by Suraj  Thomas on 22/04/26.
//

/*
┌─────────────────────────────────────────────────────────────────────┐
│                         SwiftUI Layer                                │
│   ContentView ── observes ──▶ BibleViewModel (@MainActor)           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
    ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐
    │  BibleDatabase   │  │  GPT2Actor   │  │ Tokenizer (HF)   │
    │  (GRDB + FTS5)   │  │ (Core ML)    │  │ AutoTokenizer    │
    │                  │  │              │  │                  │
    │  kjv.sqlite      │  │ DistilGPT2   │  │ vocab/merges/    │
    │  verses_fts      │  │ .mlmodelc    │  │ tokenizer.json   │
    └──────────────────┘  └──────────────┘  └──────────────────┘
         │                       │                    │
         │  RETRIEVAL (R)        │  GENERATION (G)    │  ENCODE/DECODE
         │  topic → verses       │  ids → next token  │  text ↔ ids
         │                       │                    │
         └───────────────┬───────┴────────────────────┘
                         ▼
                Augmented prompt assembled in
                BibleViewModel.search()
                    "The Bible says about {topic}:
                     {verseText} This means…"
                         │
                         ▼
                   AI summary shown in UI



┌─────────┐  ┌────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐  ┌─────────────┐
│  User   │  │ContentView │  │ BibleViewModel│  │ BibleDatabase │  │  GPT2Actor   │  │  Tokenizer  │
└────┬────┘  └─────┬──────┘  └───────┬───────┘  └───────┬───────┘  └──────┬───────┘  └──────┬──────┘
     │             │                 │                   │                  │                 │
     │  Launch app │                 │                   │                  │                 │
     │────────────▶│                 │                   │                  │                 │
     │             │  init()         │                   │                  │                 │
     │             │────────────────▶│                   │                  │                 │
     │             │                 │  init()           │                  │                 │
     │             │                 │──────────────────▶│                  │                 │
     │             │                 │                   │ setupDatabase()  │                 │
     │             │                 │                   │ copy kjv.sqlite  │                 │
     │             │                 │                   │ open DBQueue     │                 │
     │             │                 │                   │ Task: createFTSIfNeeded()          │
     │             │                 │                   │  CREATE VIRTUAL TABLE verses_fts   │
     │             │                 │                   │                  │                 │
     │             │                 │  Task {                                                │
     │             │                 │    await gpt2.loadModel() ──────────▶│                 │
     │             │                 │                                       │ MLModel.load() │
     │             │                 │◀──────────────────────────────────────│                 │
     │             │                 │    loadGPT2Tokenizer() ──────────────────────────────▶│
     │             │                 │      copy vocab/merges/etc → tempDir                  │
     │             │                 │      AutoTokenizer.from(modelFolder:)                 │
     │             │                 │◀──────────────────────────────────────────────────────│
     │             │                 │    isModelReady = true                                 │
     │             │                 │  }                                                     │
     │             │  status: Ready  │                                                        │
     │◀────────────│                 │                                                        │
     │             │                 │                                                        │
     │ Type topic, tap Search                                                                 │
     │────────────▶│                 │                                                        │
     │             │ Task{vm.search()}│                                                       │
     │             │────────────────▶│                                                        │
     │             │                 │ ── RETRIEVAL (R) ──                                    │
     │             │                 │ searchVerses(about: topic)                             │
     │             │                 │──────────────────▶│                                    │
     │             │                 │                   │ FTS5 MATCH query                   │
     │             │                 │                   │ (LIKE fallback if empty)           │
     │             │                 │                   │ map rows → [BibleVerse]            │
     │             │                 │◀──────────────────│                                    │
     │             │                 │ verses = found                                         │
     │             │                 │                                                        │
     │             │                 │ ── AUGMENTATION (A) ──                                 │
     │             │                 │ prompt = "The Bible says about \(topic):              │
     │             │                 │           \(verseText) This means"                     │
     │             │                 │                                                        │
     │             │                 │ tok.encode(text: prompt) ────────────────────────────▶│
     │             │                 │◀──────────────────────────────────────────────── ids  │
     │             │                 │                                                        │
     │             │                 │ ── GENERATION (G) ──                                   │
     │             │                 │ loop 0..<30 (generateSteps):                           │
     │             │                 │   gpt2.predictNextToken(ids, generated) ─────────────▶│ (actor hop)
     │             │                 │                                       │ build MLMultiArray
     │             │                 │                                       │ MLDictionaryFeatureProvider
     │             │                 │                                       │ model.prediction(from:)
     │             │                 │                                       │ slice logits[last]
     │             │                 │                                       │ repetition penalty
     │             │                 │                                       │ softmax(logits/T)
     │             │                 │                                       │ top-P nucleus sampling
     │             │                 │                                       │ random sample → token
     │             │                 │◀──────────────────────────────────────│
     │             │                 │   ids.append(next); generated.append(next)            │
     │             │                 │   break if EOS (50256) or period (13)                  │
     │             │                 │ end loop                                               │
     │             │                 │                                                        │
     │             │                 │ tok.decode(tokens: tail) ────────────────────────────▶│
     │             │                 │◀──────────────────────────────────────────────── text │
     │             │                 │ cleanDecoded(raw)                                      │
     │             │                 │ aiSummary = text                                       │
     │             │  @Published     │                                                        │
     │             │  re-render      │                                                        │
     │  show verses + AI insight     │                                                        │
     │◀────────────│                 │                                                        │
     │             │                 │                                                        │
*/

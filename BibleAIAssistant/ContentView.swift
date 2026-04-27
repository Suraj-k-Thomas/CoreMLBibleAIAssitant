//
//  BibleAI.swift
//  Refactored, modular Bible RAG app.
//
//  What changed vs. the original:
//  ─────────────────────────────
//  • Protocol-oriented seams: LanguageModel, TokenizerProtocol, Retriever,
//    PromptTemplate, Sampler. Swap any implementation without touching the rest.
//  • Data-driven model catalog (ModelCatalog) — add a new model = add a config entry.
//  • AppContainer composes dependencies; BibleViewModel is thin + feature-focused.
//  • Streaming generation via AsyncThrowingStream<RAGEvent, Error>. Tokens arrive
//    in the UI as they're produced; cancellation supported.
//  • Configurable sampler (NucleusSampler / GreedySampler) — pluggable.
//  • Prompt templates per architecture: PlainPromptTemplate (GPT-2 style),
//    Llama3PromptTemplate (chat-template for Llama 3.2), easy to add more.
//  • Per-model context length, vocab, stop tokens — no more hardcoded 64/50256/13.
//  • CoreMLLanguageModel replaces GPT2Actor and still runs DistilGPT2 unchanged.
//  • MLX/llama.cpp backends are declared in the factory with clear TODOs so adding
//    Llama 3.2 is a localized change.
//
//  To add Llama 3.2 later:
//    1. Add the `mlx-swift-examples` package to your Xcode project.
//    2. Implement MLXLanguageModel (see TODO in LanguageModelFactory).
//    3. The Llama 3.2 entry is already in ModelCatalog.
//

import SwiftUI
import CoreML
import GRDB
import Tokenizers

// ════════════════════════════════════════════════════════════════════
// MARK: - 1. Domain: Errors
// ════════════════════════════════════════════════════════════════════

enum AppError: LocalizedError {
    case modelFilesNotFound(String)
    case modelNotLoaded
    case tokenizerNotLoaded
    case unsupportedBackend(String)
    case unexpectedModelOutput(String)

    var errorDescription: String? {
        switch self {
        case .modelFilesNotFound(let name): return "Model files not found: \(name)"
        case .modelNotLoaded:                return "Model not loaded yet"
        case .tokenizerNotLoaded:            return "Tokenizer not loaded yet"
        case .unsupportedBackend(let b):     return "Backend not wired up: \(b)"
        case .unexpectedModelOutput(let m):  return "Unexpected model output: \(m)"
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// MARK: - 2. Domain: Value Types
// ════════════════════════════════════════════════════════════════════

struct BibleVerse: Identifiable, Sendable {
    let id = UUID()
    let book: Int
    let chapter: Int
    let verse: Int
    let rawText: String

    var text: String { TextCleaner.clean(rawText) }

    var reference: String {
        let name = BibleBookNames.name(for: book) ?? "Book \(book)"
        return "\(name) \(chapter):\(verse)"
    }
}

struct RetrievedChunk: Identifiable, Sendable {
    let id: String
    let text: String
    let score: Float
    let metadata: [String: String]
}

struct ChatMessage: Sendable {
    enum Role: String, Sendable { case system, user, assistant }
    let role: Role
    let content: String
}

struct GeneratedToken: Sendable {
    let id: Int
}

enum ModelArchitecture: String, Sendable, Codable {
    case gpt2, llama3, qwen2, mistral, phi3, gemma
}

enum ModelBackend: String, Sendable, Codable {
    case coreML, mlx, llamaCpp
}

struct LanguageModelConfiguration: Sendable, Identifiable, Hashable {
    let id: String
    let displayName: String
    let architecture: ModelArchitecture
    let backend: ModelBackend

    /// Logical max context the model supports.
    let contextLength: Int
    /// Tensor shape constraint for fixed-input-shape exports (e.g. Core ML).
    let maxInputTokens: Int
    let vocabSize: Int

    /// Bundle resource name (without extension) for weights.
    let weightsResourceName: String
    let weightsResourceExtension: String

    /// Tokenizer files: bundle resource name → standard HuggingFace filename mapping.
    let tokenizerFiles: [TokenizerFileSpec]

    /// Core ML feature interface description. nil for non-CoreML backends.
    let coreMLSpec: CoreMLInputSpec?

    let promptTemplateID: String
    let stopTokenIds: Set<Int>
    let padTokenId: Int
    let fallbackTokenId: Int
    let defaultOptions: GenerationOptions

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

/// Describes how to talk to a Core ML model's feature interface.
struct CoreMLInputSpec: Sendable {
    let inputIdsName: String      // "input_ids" (GPT-2) or "inputIds" (Llama)
    let logitsName: String        // "output_logits" (GPT-2) or "logits" (Llama)
    let inputShape: InputShape
    let extraInput: ExtraInput

    enum InputShape: Sendable {
        case flat(padLength: Int)  // GPT-2: fixed-size padded sequence [padLength]
        case batched               // Llama: dynamic [1, seqLen]
    }

    enum ExtraInput: Sendable {
        case positionIds(name: String)  // GPT-2: explicit position_ids
        case causalMask(name: String)   // Llama: additive [1,1,n,n] attention mask
        case none
    }
}

/// Maps a bundle resource to the filename AutoTokenizer expects in the temp directory.
struct TokenizerFileSpec: Sendable {
    let bundleResource: String    // Bundle.main.url(forResource:) name (no extension)
    let bundleExtension: String   // "json", "txt"
    let targetFilename: String    // filename AutoTokenizer sees (e.g. "tokenizer.json")
}

struct GenerationOptions: Sendable {
    var maxNewTokens: Int = 128
    var temperature: Double = 0.7
    var topP: Double = 0.92
    var topK: Int? = nil
    var repetitionPenalty: Double = 1.2
    /// Extra tokens that cause *soft* stop (e.g. period for one-sentence summary).
    var softStopTokens: Set<Int> = []
    var seed: UInt64? = nil
}

// ════════════════════════════════════════════════════════════════════
// MARK: - 3. Domain: Protocols (the seams)
// ════════════════════════════════════════════════════════════════════

protocol TokenizerProtocol: AnyObject, Sendable {
    func encode(_ text: String) throws -> [Int]
    func decode(_ tokens: [Int]) throws -> String
    var eosTokenIds: Set<Int> { get }
    var bosTokenId: Int? { get }
}

protocol Retriever: Sendable {
    func retrieve(query: String, k: Int) async -> [RetrievedChunk]
}

protocol PromptTemplate: Sendable {
    var id: String { get }
    func format(messages: [ChatMessage], context: [RetrievedChunk]) -> String
}

protocol Sampler: Sendable {
    func sample(logits: [Double],
                alreadyGenerated: Set<Int>,
                options: GenerationOptions) -> Int
}

protocol LanguageModel: Actor {
    nonisolated var configuration: LanguageModelConfiguration { get }
    var isLoaded: Bool { get }
    func load() async throws
    func unload() async
    func predictNextToken(contextTokens: [Int],
                          alreadyGenerated: [Int],
                          options: GenerationOptions,
                          sampler: any Sampler) async throws -> Int
}

// Default streaming loop around predictNextToken — every backend gets
// streaming for free. A backend with a native streaming API (e.g. MLX)
// can override this to skip the per-token actor hop.
extension LanguageModel {
    func generate(promptTokens: [Int],
                  options: GenerationOptions,
                  sampler: any Sampler) -> AsyncThrowingStream<GeneratedToken, Error> {
        let cfg = configuration
        return AsyncThrowingStream { continuation in
            let task = Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    var ids = promptTokens
                    var generated: [Int] = []
                    let hardStops = cfg.stopTokenIds
                    let softStops = options.softStopTokens

                    for _ in 0..<options.maxNewTokens {
                        if Task.isCancelled { break }
                        let next = try await self.predictNextToken(
                            contextTokens: ids,
                            alreadyGenerated: generated,
                            options: options,
                            sampler: sampler
                        )
                        if hardStops.contains(next) { break }
                        ids.append(next)
                        generated.append(next)
                        continuation.yield(GeneratedToken(id: next))
                        if softStops.contains(next) { break }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// MARK: - 4. Utilities
// ════════════════════════════════════════════════════════════════════

enum TextCleaner {
    static func clean(_ raw: String, trim: Bool = true) -> String {
        let replaced = raw
            .replacingOccurrences(of: "‹", with: "")
            .replacingOccurrences(of: "›", with: "")
            .replacingOccurrences(of: "<|endoftext|>", with: "")
            .replacingOccurrences(of: "<|begin_of_text|>", with: "")
            .replacingOccurrences(of: "<|eot_id|>", with: "")
            .replacingOccurrences(of: "<|end_of_text|>", with: "")
            .replacingOccurrences(of: "Ġ", with: " ")
            .replacingOccurrences(of: "Ċ", with: "\n")
        return trim ? replaced.trimmingCharacters(in: .whitespacesAndNewlines) : replaced
    }
}

enum BibleBookNames {
    private static let names: [Int: String] = [
         1:"Genesis",         2:"Exodus",           3:"Leviticus",
         4:"Numbers",         5:"Deuteronomy",      6:"Joshua",
         7:"Judges",          8:"Ruth",             9:"1 Samuel",
        10:"2 Samuel",       11:"1 Kings",         12:"2 Kings",
        13:"1 Chronicles",   14:"2 Chronicles",    15:"Ezra",
        16:"Nehemiah",       17:"Esther",          18:"Job",
        19:"Psalms",         20:"Proverbs",        21:"Ecclesiastes",
        22:"Song of Solomon",23:"Isaiah",          24:"Jeremiah",
        25:"Lamentations",   26:"Ezekiel",         27:"Daniel",
        28:"Hosea",          29:"Joel",            30:"Amos",
        31:"Obadiah",        32:"Jonah",           33:"Micah",
        34:"Nahum",          35:"Habakkuk",        36:"Zephaniah",
        37:"Haggai",         38:"Zechariah",       39:"Malachi",
        40:"Matthew",        41:"Mark",            42:"Luke",
        43:"John",           44:"Acts",            45:"Romans",
        46:"1 Corinthians",  47:"2 Corinthians",   48:"Galatians",
        49:"Ephesians",      50:"Philippians",     51:"Colossians",
        52:"1 Thessalonians",53:"2 Thessalonians", 54:"1 Timothy",
        55:"2 Timothy",      56:"Titus",           57:"Philemon",
        58:"Hebrews",        59:"James",           60:"1 Peter",
        61:"2 Peter",        62:"1 John",          63:"2 John",
        64:"3 John",         65:"Jude",            66:"Revelation"
    ]
    static func name(for id: Int) -> String? { names[id] }
}

// ════════════════════════════════════════════════════════════════════
// MARK: - 5. Infrastructure: Bible Database (unchanged behaviour)
// ════════════════════════════════════════════════════════════════════

final class BibleDatabase: @unchecked Sendable {
    private var dbQueue: DatabaseQueue?

    init() { setupDatabase() }

    private func setupDatabase() {
        let fm = FileManager.default
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dest = docs.appendingPathComponent("kjv.sqlite")

        if !fm.fileExists(atPath: dest.path),
           let src = Bundle.main.url(forResource: "kjv", withExtension: "sqlite") {
            try? fm.copyItem(at: src, to: dest)
        }
        do {
            dbQueue = try DatabaseQueue(path: dest.path)
            Task { await createFTSIfNeeded() }
        } catch {
            print("❌ DB open failed: \(error)")
        }
    }

    private func createFTSIfNeeded() async {
        guard let q = dbQueue else { return }
        do {
            try await q.write { db in
                let exists = (try Int.fetchOne(
                    db,
                    sql: """
                        SELECT COUNT(*) FROM sqlite_master
                        WHERE type='table' AND name='verses_fts';
                    """
                ) ?? 0) > 0
                guard !exists else { return }

                try db.execute(sql: """
                    CREATE VIRTUAL TABLE verses_fts
                    USING fts5(text, content='verses', content_rowid='id');
                """)
                try db.execute(sql: """
                    INSERT INTO verses_fts(rowid, text)
                    SELECT id, text FROM verses;
                """)
                print("✅ FTS5 ready")
            }
        } catch {
            print("❌ FTS error: \(error)")
        }
    }

    func fetchVerses(matching topic: String, limit: Int) async -> [BibleVerse] {
        guard let q = dbQueue else { return [] }
        do {
            return try await q.read { db in
                // Preferred: FTS5 full-text search
                let fts = try Row.fetchAll(
                    db,
                    sql: """
                        SELECT v.book, v.chapter, v.verse, v.text
                        FROM verses v
                        JOIN verses_fts ON verses_fts.rowid = v.id
                        WHERE verses_fts MATCH ?
                        ORDER BY rank
                        LIMIT \(limit);
                    """,
                    arguments: ["\"\(topic)\""]
                )
                if !fts.isEmpty {
                    return fts.map {
                        BibleVerse(book: $0["book"], chapter: $0["chapter"],
                                   verse: $0["verse"], rawText: $0["text"])
                    }
                }
                // Fallback: LIKE search
                let like = try Row.fetchAll(
                    db,
                    sql: """
                        SELECT book, chapter, verse, text FROM verses
                        WHERE text LIKE ?
                        LIMIT \(limit);
                    """,
                    arguments: ["%\(topic)%"]
                )
                return like.map {
                    BibleVerse(book: $0["book"], chapter: $0["chapter"],
                               verse: $0["verse"], rawText: $0["text"])
                }
            }
        } catch {
            print("❌ Search error: \(error)")
            return []
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// MARK: - 6. Infrastructure: Retrievers
// ════════════════════════════════════════════════════════════════════

struct FTS5Retriever: Retriever {
    let database: BibleDatabase

    func retrieve(query: String, k: Int) async -> [RetrievedChunk] {
        let verses = await database.fetchVerses(matching: query, limit: k)
        return verses.map { v in
            RetrievedChunk(
                id: "\(v.book):\(v.chapter):\(v.verse)",
                text: v.text,
                score: 1.0,
                metadata: [
                    "book": "\(v.book)",
                    "chapter": "\(v.chapter)",
                    "verse": "\(v.verse)",
                    "reference": v.reference
                ]
            )
        }
    }
}

// Future: add VectorRetriever (embedding-based) and HybridRetriever
// (RRF combine FTS5 + vector) — plug into AppContainer without touching callers.

// ════════════════════════════════════════════════════════════════════
// MARK: - 7. Infrastructure: Tokenizers
// ════════════════════════════════════════════════════════════════════

final class HFTokenizerAdapter: TokenizerProtocol, @unchecked Sendable {
    let underlying: any Tokenizer
    let eosTokenIds: Set<Int>
    let bosTokenId: Int?

    init(underlying: any Tokenizer, eosTokenIds: Set<Int>, bosTokenId: Int? = nil) {
        self.underlying = underlying
        self.eosTokenIds = eosTokenIds
        self.bosTokenId = bosTokenId
    }

    func encode(_ text: String) throws -> [Int] {
        try underlying.encode(text: text)
    }

    func decode(_ tokens: [Int]) throws -> String {
        try underlying.decode(tokens: tokens)
    }
}

enum TokenizerFactory {
    static func make(for config: LanguageModelConfiguration) async throws -> any TokenizerProtocol {
        let fm = FileManager.default
        let tempDir = fm.temporaryDirectory
            .appendingPathComponent("tok-\(config.id)", isDirectory: true)
        try? fm.removeItem(at: tempDir)
        try fm.createDirectory(at: tempDir, withIntermediateDirectories: true)

        var missing: [String] = []
        for spec in config.tokenizerFiles {
            guard let src = Bundle.main.url(
                forResource: spec.bundleResource,
                withExtension: spec.bundleExtension
            ) else {
                missing.append(spec.targetFilename)
                continue
            }
            try fm.copyItem(at: src, to: tempDir.appendingPathComponent(spec.targetFilename))
        }
        guard missing.isEmpty else {
            print("❌ Missing tokenizer files: \(missing.joined(separator: ", "))")
            throw AppError.modelFilesNotFound("tokenizer [\(missing.joined(separator: ", "))]")
        }

        let underlying = try await AutoTokenizer.from(modelFolder: tempDir)
        print("✅ Tokenizer loaded for \(config.id)")
        return HFTokenizerAdapter(
            underlying: underlying,
            eosTokenIds: config.stopTokenIds,
            bosTokenId: nil
        )
    }
}

// ════════════════════════════════════════════════════════════════════
// MARK: - 8. Infrastructure: Samplers
// ════════════════════════════════════════════════════════════════════

struct NucleusSampler: Sampler {
    func sample(logits: [Double],
                alreadyGenerated: Set<Int>,
                options: GenerationOptions) -> Int {
        guard !logits.isEmpty else { return 0 }

        // Repetition penalty + finite filter
        var pairs: [(Int, Double)] = []
        pairs.reserveCapacity(logits.count)
        for (idx, v) in logits.enumerated() where v.isFinite {
            var val = v
            if alreadyGenerated.contains(idx) {
                val = val > 0
                    ? val / options.repetitionPenalty
                    : val * options.repetitionPenalty
            }
            pairs.append((idx, val))
        }
        guard !pairs.isEmpty else { return 0 }

        // Temperature + numerically stable softmax
        let t = max(options.temperature, 1e-6)
        let scaled = pairs.map { ($0.0, $0.1 / t) }
        let maxLogit = scaled.max(by: { $0.1 < $1.1 })!.1
        var probs = scaled.map { ($0.0, exp($0.1 - maxLogit)) }
        let sum = probs.reduce(0.0) { $0 + $1.1 }
        guard sum > 0 else { return pairs.first!.0 }
        probs = probs.map { ($0.0, $0.1 / sum) }

        // Optional top-K truncation
        if let k = options.topK, k > 0, k < probs.count {
            probs = Array(probs.sorted { $0.1 > $1.1 }.prefix(k))
            let s = probs.reduce(0.0) { $0 + $1.1 }
            if s > 0 { probs = probs.map { ($0.0, $0.1 / s) } }
        }

        // Top-P nucleus
        let sorted = probs.sorted { $0.1 > $1.1 }
        var nucleus: [(Int, Double)] = []
        var cum = 0.0
        for p in sorted {
            nucleus.append(p)
            cum += p.1
            if cum >= options.topP { break }
        }
        let nSum = nucleus.reduce(0.0) { $0 + $1.1 }
        guard nSum > 0 else { return sorted.first!.0 }
        let renorm = nucleus.map { ($0.0, $0.1 / nSum) }

        // Sample
        var r = Double.random(in: 0..<1)
        for (t, p) in renorm {
            r -= p
            if r <= 0 { return t }
        }
        return renorm.last?.0 ?? 0
    }
}

struct GreedySampler: Sampler {
    func sample(logits: [Double],
                alreadyGenerated: Set<Int>,
                options: GenerationOptions) -> Int {
        var bestIdx = 0
        var bestVal = -Double.infinity
        for (i, v) in logits.enumerated() where v.isFinite {
            if v > bestVal { bestVal = v; bestIdx = i }
        }
        return bestIdx
    }
}

// ════════════════════════════════════════════════════════════════════
// MARK: - 9. Infrastructure: Prompt Templates
// ════════════════════════════════════════════════════════════════════

/// GPT-2 style completion prompt. Used for non-chat models.
struct PlainPromptTemplate: PromptTemplate {
    let id = "plain"
    func format(messages: [ChatMessage], context: [RetrievedChunk]) -> String {
        let userQuery = messages.last(where: { $0.role == .user })?.content ?? ""
        if let firstCtx = context.first?.text {
            return "The Bible says about \(userQuery): \(firstCtx) This means"
        }
        return userQuery
    }
}

/// Llama 3 / 3.1 / 3.2 chat template. Required for decent Llama 3.2 output.
struct Llama3PromptTemplate: PromptTemplate {
    let id = "llama3"

    func format(messages: [ChatMessage], context: [RetrievedChunk]) -> String {
        let ctx = context.map { "- \($0.text)" }.joined(separator: "\n")
        let baseSystem = messages.first(where: { $0.role == .system })?.content
            ?? "You are a thoughtful assistant answering questions about the King James Bible."
        let systemWithCtx = context.isEmpty ? baseSystem :
            "\(baseSystem)\n\nUse the following retrieved passages from the KJV as authoritative context. Quote them where relevant.\n\n\(ctx)"

        var out = "<|begin_of_text|>"
        out += "<|start_header_id|>system<|end_header_id|>\n\n"
        out += systemWithCtx
        out += "<|eot_id|>"
        for m in messages where m.role != .system {
            out += "<|start_header_id|>\(m.role.rawValue)<|end_header_id|>\n\n"
            out += m.content
            out += "<|eot_id|>"
        }
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return out
    }
}

/// Qwen 2 / 2.5 chat template.
struct Qwen2PromptTemplate: PromptTemplate {
    let id = "qwen2"
    func format(messages: [ChatMessage], context: [RetrievedChunk]) -> String {
        let ctx = context.map { "- \($0.text)" }.joined(separator: "\n")
        let base = messages.first(where: { $0.role == .system })?.content
            ?? "You are a helpful assistant."
        let sys = context.isEmpty ? base : "\(base)\n\nContext:\n\(ctx)"

        var out = "<|im_start|>system\n\(sys)<|im_end|>\n"
        for m in messages where m.role != .system {
            out += "<|im_start|>\(m.role.rawValue)\n\(m.content)<|im_end|>\n"
        }
        out += "<|im_start|>assistant\n"
        return out
    }
}

enum PromptTemplateRegistry {
    static func template(id: String) -> any PromptTemplate {
        switch id {
        case "llama3": return Llama3PromptTemplate()
        case "qwen2":  return Qwen2PromptTemplate()
        case "plain":  return PlainPromptTemplate()
        default:       return PlainPromptTemplate()
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// MARK: - 10. Infrastructure: Language Model Backends
// ════════════════════════════════════════════════════════════════════

/// Core ML backend — runs DistilGPT2.
actor CoreMLLanguageModel: LanguageModel {
    nonisolated let configuration: LanguageModelConfiguration
    private var model: MLModel?

    init(configuration: LanguageModelConfiguration) {
        self.configuration = configuration
    }

    var isLoaded: Bool { model != nil }

    func load() async throws {
        // Try the configured extension first, then mlmodelc (Xcode compiles mlpackage → mlmodelc).
        let url = Bundle.main.url(
            forResource: configuration.weightsResourceName,
            withExtension: configuration.weightsResourceExtension
        ) ?? Bundle.main.url(
            forResource: configuration.weightsResourceName,
            withExtension: "mlmodelc"
        )
        guard let url else {
            throw AppError.modelFilesNotFound(configuration.weightsResourceName)
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        print("⏳ Loading \(configuration.displayName) from \(url.lastPathComponent)…")
        model = try await MLModel.load(contentsOf: url, configuration: cfg)
        print("✅ \(configuration.displayName) loaded")
    }

    func unload() { model = nil }

    func predictNextToken(contextTokens ids: [Int],
                          alreadyGenerated generated: [Int],
                          options: GenerationOptions,
                          sampler: any Sampler) async throws -> Int {
        guard let model else { throw AppError.modelNotLoaded }
        guard let spec = configuration.coreMLSpec else {
            throw AppError.unsupportedBackend("No CoreML I/O spec for \(configuration.id)")
        }

        let seq = Array(ids.suffix(configuration.maxInputTokens))
        let n = seq.count

        // ── Build input feature provider ──────────────────────────────
        // Fixed-shape GPT-2 style: flat [padLen] with right-padding and position IDs.
        guard case .flat(let padLen) = spec.inputShape else {
            throw AppError.unsupportedBackend("Only flat input shape supported in this version")
        }
        let inputArr = try MLMultiArray(shape: [padLen as NSNumber], dataType: .int32)
        let posArr   = try MLMultiArray(shape: [padLen as NSNumber], dataType: .int32)
        let padID = configuration.padTokenId
        for i in 0..<padLen {
            inputArr[i] = NSNumber(value: i < n ? seq[i] : padID)
            posArr[i]   = NSNumber(value: i)
        }
        var dict: [String: MLFeatureValue] = [
            spec.inputIdsName: MLFeatureValue(multiArray: inputArr)
        ]
        if case .positionIds(let name) = spec.extraInput {
            dict[name] = MLFeatureValue(multiArray: posArr)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: dict)

        // ── Run inference ─────────────────────────────────────────────
        let out = try await model.prediction(from: provider)
        guard let logits = out.featureValue(for: spec.logitsName)?.multiArrayValue else {
            return configuration.fallbackTokenId
        }

        // ── Extract last-position logits ──────────────────────────────
        // Shape [1, padLen, vocab, 1, 1] — extract at position n-1.
        let vocab = configuration.vocabSize
        var raw = [Double](repeating: -.infinity, count: vocab)
        let shape = logits.shape.map { $0.intValue }
        guard shape.count >= 3 else { return configuration.fallbackTokenId }
        let seqLen = shape[1]
        let last = min(n - 1, seqLen - 1)
        let base = last * vocab
        for v in 0..<vocab where base + v < logits.count {
            raw[v] = logits[base + v].doubleValue
        }

        return sampler.sample(logits: raw, alreadyGenerated: Set(generated), options: options)
    }
}

/// Placeholder — real implementation goes here when you add MLX Swift.
/// When ready:
///   1. Add `https://github.com/ml-explore/mlx-swift-examples` to Xcode.
///   2. `import MLX; import MLXLLM; import MLXLMCommon`
///   3. Load via `LLMModelFactory.shared.loadContainer(configuration:)`.
///   4. Override `generate(...)` to stream tokens via MLX's native generator
///      (skips the per-token actor hop and uses KV cache).
/*
actor MLXLanguageModel: LanguageModel {
    nonisolated let configuration: LanguageModelConfiguration
    private var container: ModelContainer?
    var isLoaded: Bool { container != nil }

    init(configuration: LanguageModelConfiguration) {
        self.configuration = configuration
    }

    func load() async throws { /* LLMModelFactory.shared.loadContainer(...) */ }
    func unload() { container = nil }

    func predictNextToken(contextTokens: [Int],
                          alreadyGenerated: [Int],
                          options: GenerationOptions,
                          sampler: any Sampler) async throws -> Int {
        // Not typically used — override generate(...) instead for native streaming.
        fatalError("Use streaming generate() for MLX.")
    }
}
*/

// ════════════════════════════════════════════════════════════════════
// MARK: - 11. Application: Model Factory + Catalog
// ════════════════════════════════════════════════════════════════════

enum LanguageModelFactory {
    static func make(for config: LanguageModelConfiguration) throws -> any LanguageModel {
        switch config.backend {
        case .coreML:
            return CoreMLLanguageModel(configuration: config)
        case .mlx:
            // TODO: return MLXLanguageModel(configuration: config)
            throw AppError.unsupportedBackend(
                "MLX not wired. Add mlx-swift-examples package and implement MLXLanguageModel."
            )
        case .llamaCpp:
            // TODO: return LlamaCppLanguageModel(configuration: config)
            throw AppError.unsupportedBackend(
                "llama.cpp not wired. Add Swift llama.cpp bindings and implement LlamaCppLanguageModel."
            )
        }
    }
}

enum ModelCatalog {
    static let all: [LanguageModelConfiguration] = [
        .distilGPT2,
        .llama3_2_1bInstruct,
    ]
    static var defaultConfig: LanguageModelConfiguration { .distilGPT2 }
}

extension LanguageModelConfiguration {

    /// DistilGPT2 via Core ML — ships today. Original behaviour preserved.
    static let distilGPT2 = LanguageModelConfiguration(
        id: "distilgpt2",
        displayName: "DistilGPT2 (Core ML)",
        architecture: .gpt2,
        backend: .coreML,
        contextLength: 1024,
        maxInputTokens: 64,
        vocabSize: 50257,
        weightsResourceName: "DistilGPT2",
        weightsResourceExtension: "mlmodelc",
        tokenizerFiles: [
            TokenizerFileSpec(bundleResource: "vocab",             bundleExtension: "json", targetFilename: "vocab.json"),
            TokenizerFileSpec(bundleResource: "merges",            bundleExtension: "txt",  targetFilename: "merges.txt"),
            TokenizerFileSpec(bundleResource: "tokenizer",         bundleExtension: "json", targetFilename: "tokenizer.json"),
            TokenizerFileSpec(bundleResource: "tokenizer_config",  bundleExtension: "json", targetFilename: "tokenizer_config.json"),
        ],
        coreMLSpec: CoreMLInputSpec(
            inputIdsName: "input_ids",
            logitsName: "output_logits",
            inputShape: .flat(padLength: 64),
            extraInput: .positionIds(name: "position_ids")
        ),
        promptTemplateID: "plain",
        stopTokenIds: [50256],
        padTokenId: 50256,
        fallbackTokenId: 198,
        defaultOptions: GenerationOptions(
            maxNewTokens: 30,
            temperature: 0.7,
            topP: 0.92,
            topK: nil,
            repetitionPenalty: 1.5,
            softStopTokens: [13]
        )
    )

    /// Llama 3.2 1B-Instruct via Core ML (Llama3_2.mlpackage).
    /// Tokenizer files required in bundle: llama_tokenizer.json + llama_tokenizer_config.json
    static let llama3_2_1bInstruct = LanguageModelConfiguration(
        id: "llama-3.2-1b-coreml",
        displayName: "Llama 3.2 1B (Core ML)",
        architecture: .llama3,
        backend: .coreML,
        contextLength: 131_072,
        maxInputTokens: 512,          // practical on-device cap per forward pass
        vocabSize: 128_256,
        weightsResourceName: "Llama3_2",
        weightsResourceExtension: "mlpackage",
        tokenizerFiles: [
            TokenizerFileSpec(bundleResource: "llama_tokenizer",        bundleExtension: "json", targetFilename: "tokenizer.json"),
            TokenizerFileSpec(bundleResource: "llama_tokenizer_config", bundleExtension: "json", targetFilename: "tokenizer_config.json"),
        ],
        coreMLSpec: CoreMLInputSpec(
            inputIdsName: "inputIds",
            logitsName: "logits",
            inputShape: .batched,
            extraInput: .causalMask(name: "causalMask")
        ),
        promptTemplateID: "llama3",
        stopTokenIds: [128001, 128008, 128009],  // <|end_of_text|>, <|eom_id|>, <|eot_id|>
        padTokenId: 128004,
        fallbackTokenId: 128009,
        defaultOptions: GenerationOptions(
            maxNewTokens: 256,
            temperature: 0.6,
            topP: 0.9,
            topK: 40,
            repetitionPenalty: 1.1,
            softStopTokens: []
        )
    )
}

// ════════════════════════════════════════════════════════════════════
// MARK: - 12. Application: RAG Service (retrieve → augment → generate)
// ════════════════════════════════════════════════════════════════════

enum RAGEvent: Sendable {
    case retrieving
    case retrieved([RetrievedChunk])
    case generating
    case token(String)        // incremental text delta
    case done(String)         // final assembled text
}

final class RAGService: @unchecked Sendable {
    private let retriever: any Retriever
    private let model: any LanguageModel
    private let tokenizer: any TokenizerProtocol
    private let template: any PromptTemplate
    private let sampler: any Sampler

    init(retriever: any Retriever,
         model: any LanguageModel,
         tokenizer: any TokenizerProtocol,
         template: any PromptTemplate,
         sampler: any Sampler = NucleusSampler()) {
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer
        self.template = template
        self.sampler = sampler
    }

    func answer(query: String,
                systemPrompt: String? = nil,
                k: Int = 3,
                options: GenerationOptions) -> AsyncThrowingStream<RAGEvent, Error> {
        AsyncThrowingStream { [retriever, model, tokenizer, template, sampler] continuation in
            let task = Task {
                do {
                    // 1. Retrieve
                    continuation.yield(.retrieving)
                    let chunks = await retriever.retrieve(query: query, k: k)
                    continuation.yield(.retrieved(chunks))

                    // 2. Augment
                    var messages: [ChatMessage] = []
                    if let sys = systemPrompt {
                        messages.append(.init(role: .system, content: sys))
                    }
                    messages.append(.init(role: .user, content: query))

                    let promptTokens = try tokenizer.encode(
                        template.format(messages: messages, context: chunks)
                    )
                    print("📝 Prompt tokens: \(promptTokens.count)")

                    // 3. Generate (stream)
                    continuation.yield(.generating)
                    var accumulated: [Int] = []
                    var lastText = ""

                    let stream = await model.generate(
                        promptTokens: promptTokens,
                        options: options,
                        sampler: sampler
                    )

                    for try await tok in stream {
                        if Task.isCancelled { break }
                        accumulated.append(tok.id)
                        // Incremental decode — decode the full run, diff against prev
                        let full = (try? tokenizer.decode(accumulated)) ?? ""
                        let cleaned = TextCleaner.clean(full, trim: false)
                        if cleaned.count > lastText.count {
                            let startIdx = cleaned.index(cleaned.startIndex, offsetBy: lastText.count)
                            let delta = String(cleaned[startIdx...])
                            continuation.yield(.token(delta))
                            lastText = cleaned
                        }
                    }

                    let finalText = TextCleaner.clean(lastText, trim: true)
                    continuation.yield(.done(finalText))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// MARK: - 13. Composition Root (AppContainer)
// ════════════════════════════════════════════════════════════════════

@MainActor
final class AppContainer: ObservableObject {
    @Published private(set) var availableModels: [LanguageModelConfiguration] = ModelCatalog.all
    @Published private(set) var activeConfig: LanguageModelConfiguration?
    @Published private(set) var isReady: Bool = false
    @Published private(set) var status: String = "Idle"

    let database: BibleDatabase
    let retriever: any Retriever
    private(set) var rag: RAGService?

    private var currentModel: (any LanguageModel)?

    init() {
        let db = BibleDatabase()
        self.database = db
        self.retriever = FTS5Retriever(database: db)
    }

    /// Load (or switch to) a model. The RAG service is rebuilt around it.
    func switchModel(to config: LanguageModelConfiguration) async {
        isReady = false
        status = "Unloading previous model…"
        if let m = currentModel { await m.unload() }
        currentModel = nil
        rag = nil

        do {
            status = "Loading \(config.displayName)…"
            let model = try LanguageModelFactory.make(for: config)
            try await model.load()
            let tokenizer = try await TokenizerFactory.make(for: config)
            let template = PromptTemplateRegistry.template(id: config.promptTemplateID)

            rag = RAGService(
                retriever: retriever,
                model: model,
                tokenizer: tokenizer,
                template: template,
                sampler: NucleusSampler()
            )
            currentModel = model
            activeConfig = config
            isReady = true
            status = "⚡ \(config.displayName) ready"
        } catch {
            isReady = false
            status = "❌ \(error.localizedDescription)"
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// MARK: - 14. Presentation: ViewModel
// ════════════════════════════════════════════════════════════════════

@MainActor
final class BibleViewModel: ObservableObject {
    @Published var verses: [BibleVerse] = []
    @Published var aiSummary: String = ""
    @Published var isStreaming: Bool = false
    @Published var searchTopic: String = ""

    private let container: AppContainer
    private var task: Task<Void, Never>?

    init(container: AppContainer) { self.container = container }

    func search() {
        task?.cancel()
        let topic = searchTopic.trimmingCharacters(in: .whitespaces)
        guard !topic.isEmpty,
              let rag = container.rag,
              let config = container.activeConfig
        else { return }

        isStreaming = true
        aiSummary = ""
        verses = []

        task = Task { [weak self] in
            do {
                let stream = rag.answer(
                    query: topic,
                    systemPrompt: "You are a helpful assistant answering questions about the King James Bible. Based on the provided verses, give a thoughtful and concise commentary.",
                    k: 3,
                    options: config.defaultOptions
                )

                for try await event in stream {
                    if Task.isCancelled { break }
                    guard let self else { return }
                    switch event {
                    case .retrieving:
                        break
                    case .retrieved(let chunks):
                        self.verses = chunks.compactMap(BibleVerse.init(chunk:))
                    case .generating:
                        self.aiSummary = ""
                    case .token(let delta):
                        self.aiSummary.append(delta)
                    case .done(let final):
                        if self.aiSummary.isEmpty { self.aiSummary = final }
                    }
                }
            } catch {
                self?.aiSummary = "Error: \(error.localizedDescription)"
            }
            self?.isStreaming = false
        }
    }

    func cancel() { task?.cancel() }
}

extension BibleVerse {
    /// Reconstruct from a retrieved chunk, if metadata is well-formed.
    init?(chunk: RetrievedChunk) {
        guard let book = Int(chunk.metadata["book"] ?? ""),
              let ch   = Int(chunk.metadata["chapter"] ?? ""),
              let vs   = Int(chunk.metadata["verse"] ?? "")
        else { return nil }
        self.book = book
        self.chapter = ch
        self.verse = vs
        self.rawText = chunk.text
    }
}

// ════════════════════════════════════════════════════════════════════
// MARK: - 15. Presentation: Views
// ════════════════════════════════════════════════════════════════════

/// Outer view: owns the AppContainer.
/// Hands the container down to `BibleScreen`, which constructs its
/// view model with it using the standard `@StateObject(wrappedValue:)`
/// pattern. This avoids the `@StateObject` + `@EnvironmentObject`
/// init-ordering problem.
struct ContentView: View {
    @StateObject private var container = AppContainer()

    var body: some View {
        BibleScreen(container: container)
            .task {
                if container.activeConfig == nil {
                    await container.switchModel(to: ModelCatalog.defaultConfig)
                }
            }
    }
}

struct BibleScreen: View {
    @ObservedObject var container: AppContainer
    @StateObject private var vm: BibleViewModel

    init(container: AppContainer) {
        self.container = container
        _vm = StateObject(wrappedValue: BibleViewModel(container: container))
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                statusBar
                Divider()
                modelPicker
                Divider()
                searchBar
                results
            }
            .navigationTitle("📖 Bible AI")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    // MARK: Subviews

    private var statusBar: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(container.isReady ? Color.green : Color.orange)
                .frame(width: 8, height: 8)
            Text(container.status)
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(Color(.systemBackground))
    }

    private var modelPicker: some View {
        HStack {
            Text("Model").font(.caption).foregroundColor(.secondary)
            Menu {
                ForEach(container.availableModels) { cfg in
                    Button {
                        Task { await container.switchModel(to: cfg) }
                    } label: {
                        if cfg.id == container.activeConfig?.id {
                            Label(cfg.displayName, systemImage: "checkmark")
                        } else {
                            Text(cfg.displayName)
                        }
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    Text(container.activeConfig?.displayName ?? "Select…")
                        .font(.caption.weight(.semibold))
                    Image(systemName: "chevron.down").font(.caption2)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color(.secondarySystemBackground))
                .cornerRadius(8)
            }
            Spacer()
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
    }

    private var searchBar: some View {
        HStack {
            TextField("Search topic (e.g. peace, love, faith)", text: $vm.searchTopic)
                .textFieldStyle(.roundedBorder)
                .onSubmit { vm.search() }
                .disabled(vm.isStreaming)

            Button {
                if vm.isStreaming { vm.cancel() } else { vm.search() }
            } label: {
                if vm.isStreaming {
                    Image(systemName: "stop.fill")
                } else {
                    Image(systemName: "magnifyingglass")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(!container.isReady && !vm.isStreaming)
        }
        .padding()
    }

    private var results: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if !vm.verses.isEmpty {
                    Text("📖 KJV Verses").font(.headline).padding(.top, 4)
                    ForEach(vm.verses) { v in verseCard(v) }
                }
                if !vm.aiSummary.isEmpty {
                    aiInsightCard
                }
                if vm.verses.isEmpty && vm.aiSummary.isEmpty && !vm.isStreaming {
                    emptyState
                }
            }
            .padding(.horizontal)
            .padding(.bottom, 32)
        }
    }

    private func verseCard(_ v: BibleVerse) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(v.reference)
                .font(.caption.weight(.semibold))
                .foregroundColor(.accentColor)
            Text(v.text).font(.body)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground))
        .cornerRadius(10)
    }

    private var aiInsightCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("AI Insight", systemImage: "sparkles").font(.headline)
            Text(vm.aiSummary)
                .font(.body)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.accentColor.opacity(0.08))
        .cornerRadius(10)
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "book.closed")
                .font(.system(size: 48))
                .foregroundColor(.secondary)
            Text("Search the King James Bible")
                .font(.headline)
                .foregroundColor(.secondary)
            Text("Try: love, faith, grace, hope, peace")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 80)
    }
}

// ════════════════════════════════════════════════════════════════════
// MARK: - 16. App Entry Point
// ════════════════════════════════════════════════════════════════════

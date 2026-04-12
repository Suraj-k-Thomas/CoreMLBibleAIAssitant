import SwiftUI
import CoreML
import GRDB
import Tokenizers

// MARK: - Constants
private enum AppConstants {
    static let maxTokens:          Int    = 64
    static let versesLimit:        Int    = 3
    static let generateSteps:      Int    = 30
    static let eosToken:           Int    = 50256
    static let periodToken:        Int    = 13
    static let repetitionPenalty:  Double = 1.5
    static let temperature:        Double = 0.7
    static let topP:               Double = 0.92
}

// MARK: - BibleVerse
struct BibleVerse: Identifiable {
    let id      = UUID()
    let book:    Int
    let chapter: Int
    let verse:   Int
    let rawText: String

    var text: String {
        rawText
            .replacingOccurrences(of: "‹", with: "")
            .replacingOccurrences(of: "›", with: "")
            .replacingOccurrences(of: "<|endoftext|>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var reference: String {
        let names: [Int: String] = [
             1:"Genesis",        2:"Exodus",          3:"Leviticus",
             4:"Numbers",        5:"Deuteronomy",     6:"Joshua",
             7:"Judges",         8:"Ruth",             9:"1 Samuel",
            10:"2 Samuel",      11:"1 Kings",         12:"2 Kings",
            13:"1 Chronicles",  14:"2 Chronicles",    15:"Ezra",
            16:"Nehemiah",      17:"Esther",          18:"Job",
            19:"Psalms",        20:"Proverbs",        21:"Ecclesiastes",
            22:"Song of Solomon",23:"Isaiah",         24:"Jeremiah",
            25:"Lamentations",  26:"Ezekiel",         27:"Daniel",
            28:"Hosea",         29:"Joel",            30:"Amos",
            31:"Obadiah",       32:"Jonah",           33:"Micah",
            34:"Nahum",         35:"Habakkuk",        36:"Zephaniah",
            37:"Haggai",        38:"Zechariah",       39:"Malachi",
            40:"Matthew",       41:"Mark",            42:"Luke",
            43:"John",          44:"Acts",            45:"Romans",
            46:"1 Corinthians", 47:"2 Corinthians",  48:"Galatians",
            49:"Ephesians",     50:"Philippians",     51:"Colossians",
            52:"1 Thessalonians",53:"2 Thessalonians",54:"1 Timothy",
            55:"2 Timothy",     56:"Titus",           57:"Philemon",
            58:"Hebrews",       59:"James",           60:"1 Peter",
            61:"2 Peter",       62:"1 John",          63:"2 John",
            64:"3 John",        65:"Jude",            66:"Revelation"
        ]
        return "\(names[book] ?? "Book \(book)") \(chapter):\(verse)"
    }
}

// MARK: - Errors
enum AppError: LocalizedError {
    case modelNotFound
    case modelNotLoaded

    var errorDescription: String? {
        switch self {
        case .modelNotFound:  return "Model files not found in bundle"
        case .modelNotLoaded: return "Model not loaded yet"
        }
    }
}

// MARK: - BibleDatabase
final class BibleDatabase {
    private var dbQueue: DatabaseQueue?

    init() { setupDatabase() }

    private func setupDatabase() {
        let fm   = FileManager.default
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

    func searchVerses(about topic: String) async -> [BibleVerse] {
        guard let q = dbQueue else { return [] }
        do {
            return try await q.read { db in
                // FTS search
                let fts = try Row.fetchAll(
                    db,
                    sql: """
                        SELECT v.book, v.chapter, v.verse, v.text
                        FROM verses v
                        JOIN verses_fts ON verses_fts.rowid = v.id
                        WHERE verses_fts MATCH ?
                        ORDER BY rank
                        LIMIT \(AppConstants.versesLimit);
                    """,
                    arguments: ["\"\(topic)\""]
                )

                if !fts.isEmpty {
                    return fts.map {
                        BibleVerse(book: $0["book"], chapter: $0["chapter"],
                                   verse: $0["verse"], rawText: $0["text"])
                    }
                }

                // LIKE fallback
                let like = try Row.fetchAll(
                    db,
                    sql: """
                        SELECT book, chapter, verse, text FROM verses
                        WHERE text LIKE ?
                        LIMIT \(AppConstants.versesLimit);
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

// MARK: - GPT2Actor
actor GPT2Actor {
    private var model: MLModel?

    func loadModel() async throws {
        guard let url = Bundle.main.url(
            forResource: "DistilGPT2", withExtension: "mlmodelc"
        ) else { throw AppError.modelNotFound }

        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        model = try await MLModel.load(contentsOf: url, configuration: cfg)
        print("✅ DistilGPT2 loaded")
    }

    func predictNextToken(_ ids: [Int], generated: [Int]) async throws -> Int {
        guard let model else { throw AppError.modelNotLoaded }

        let seq    = Array(ids.suffix(AppConstants.maxTokens))
        let n      = seq.count
        let padLen = AppConstants.maxTokens

        // Build input arrays
        let input = try MLMultiArray(shape: [padLen as NSNumber], dataType: .int32)
        let pos   = try MLMultiArray(shape: [padLen as NSNumber], dataType: .int32)
        for i in 0..<padLen {
            input[i] = NSNumber(value: i < n ? seq[i] : AppConstants.eosToken)
            pos[i]   = NSNumber(value: i)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids":    MLFeatureValue(multiArray: input),
            "position_ids": MLFeatureValue(multiArray: pos)
        ])

        let out = try await model.prediction(from: provider)

        guard let logits = out.featureValue(for: "output_logits")?.multiArrayValue
        else { return 198 }

        // Shape: [1, 64, 50257, 1, 1]
        let shape   = logits.shape.map { $0.intValue }
        let seqLen  = shape[1]
        let vocab   = shape[2]
        let last    = min(n - 1, seqLen - 1)
        let baseIdx = last * vocab

        let penaltySet = Set(generated)

        // Collect logits with repetition penalty
        var pairs = [(token: Int, logit: Double)]()
        pairs.reserveCapacity(vocab)

        for v in 0..<vocab {
            let idx = baseIdx + v
            guard idx < logits.count else { break }
            var val = logits[idx].doubleValue
            guard val.isFinite else { continue }

            if penaltySet.contains(v) {
                val = val > 0
                    ? val / AppConstants.repetitionPenalty
                    : val * AppConstants.repetitionPenalty
            }
            pairs.append((v, val))
        }

        guard !pairs.isEmpty else { return 198 }

        // Temperature scaling + numerically stable softmax
        let scaled   = pairs.map { ($0.token, $0.logit / AppConstants.temperature) }
        let maxLogit = scaled.max(by: { $0.1 < $1.1 })!.1
        var expVals  = scaled.map { ($0.0, exp($0.1 - maxLogit)) }
        let sumExp   = expVals.reduce(0.0) { $0 + $1.1 }
        expVals      = expVals.map { ($0.0, $0.1 / sumExp) }

        // Top-P nucleus sampling
        let sorted   = expVals.sorted { $0.1 > $1.1 }
        var nucleus  = [(Int, Double)]()
        var cumProb  = 0.0
        for pair in sorted {
            nucleus.append(pair)
            cumProb += pair.1
            if cumProb >= AppConstants.topP { break }
        }

        let nucSum = nucleus.reduce(0.0) { $0 + $1.1 }
        let renorm = nucleus.map { ($0.0, $0.1 / nucSum) }

        // Sample
        var r = Double.random(in: 0..<1)
        for (token, prob) in renorm {
            r -= prob
            if r <= 0 { return token }
        }
        return renorm.last?.0 ?? 198
    }
}

// MARK: - Tokenizer loader
func loadGPT2Tokenizer() async throws -> any Tokenizer {
    let fm      = FileManager.default
    let tempDir = fm.temporaryDirectory
        .appendingPathComponent("gpt2tok", isDirectory: true)
    try? fm.removeItem(at: tempDir)
    try fm.createDirectory(at: tempDir, withIntermediateDirectories: true)

    let required = [
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    var missing = [String]()

    for fname in required {
        let parts = fname.split(separator: ".")
        guard let src = Bundle.main.url(
            forResource: String(parts[0]),
            withExtension: String(parts[1])
        ) else {
            missing.append(fname)
            continue
        }
        try fm.copyItem(at: src, to: tempDir.appendingPathComponent(fname))
    }

    guard missing.isEmpty else {
        print("❌ Missing bundle files: \(missing.joined(separator: ", "))")
        throw AppError.modelNotFound
    }

    let tok = try await AutoTokenizer.from(modelFolder: tempDir)
    print("✅ Tokenizer loaded")
    return tok
}

// MARK: - Helpers
private func cleanDecoded(_ raw: String) -> String {
    raw
        .replacingOccurrences(of: "‹", with: "")
        .replacingOccurrences(of: "›", with: "")
        .replacingOccurrences(of: "<|endoftext|>", with: "")
        .replacingOccurrences(of: "Ġ", with: " ")
        .replacingOccurrences(of: "Ċ", with: "\n")
        .trimmingCharacters(in: .whitespacesAndNewlines)
}

// MARK: - ViewModel
@MainActor
final class BibleViewModel: ObservableObject {
    @Published var modelStatus:  String     = "Loading…"
    @Published var isModelReady: Bool       = false
    @Published var isSearching:  Bool       = false
    @Published var verses:       [BibleVerse] = []
    @Published var aiSummary:    String     = ""
    @Published var searchTopic:  String     = ""

    private let db        = BibleDatabase()
    private let gpt2      = GPT2Actor()
    private var tokenizer: (any Tokenizer)?

    init() {
        Task {
            do {
                try await gpt2.loadModel()
                tokenizer    = try await loadGPT2Tokenizer()
                isModelReady = true
                modelStatus  = "⚡ Model Ready"
            } catch {
                modelStatus = "❌ \(error.localizedDescription)"
            }
        }
    }

    func search() async {
        let topic = searchTopic.trimmingCharacters(in: .whitespaces)
        guard !topic.isEmpty, !isSearching else { return }

        isSearching = true
        aiSummary   = ""
        verses      = []

        // 1. Find verses
        let found = await db.searchVerses(about: topic)
        verses = found

        guard !found.isEmpty else {
            aiSummary   = "No verses found for '\(topic)'."
            isSearching = false
            return
        }

        // 2. Build prompt from first verse (cleaned)
        let verseText = found[0].text
        let prompt    = "The Bible says about \(topic): \(verseText) This means"
        aiSummary     = "Generating…"

        // 3. Generate
        do {
            guard let tok = tokenizer else { throw AppError.modelNotLoaded }

            var ids       = try tok.encode(text: prompt)
            var generated = [Int]()
            let start     = ids.count

            for _ in 0..<AppConstants.generateSteps {
                let next = try await gpt2.predictNextToken(ids, generated: generated)
                if next == AppConstants.eosToken { break }
                ids.append(next)
                generated.append(next)
                if next == AppConstants.periodToken { break }
            }

            let raw  = try tok.decode(tokens: Array(ids.dropFirst(start)))
            let text = cleanDecoded(raw)
            aiSummary = text.isEmpty ? "No insight generated." : text

        } catch {
            aiSummary = "Error: \(error.localizedDescription)"
        }

        isSearching = false
    }
}

// MARK: - ContentView
struct ContentView: View {
    @StateObject private var vm = BibleViewModel()

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {

                // Status bar
                HStack(spacing: 6) {
                    Circle()
                        .fill(vm.isModelReady ? Color.green : Color.orange)
                        .frame(width: 8, height: 8)
                    Text(vm.modelStatus)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                }
                .padding(.horizontal)
                .padding(.vertical, 6)
                .background(Color(.systemBackground))

                Divider()

                // Search bar
                HStack {
                    TextField("Search topic (e.g. peace, love, faith)",
                              text: $vm.searchTopic)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit { Task { await vm.search() } }
                        .disabled(vm.isSearching)

                    Button {
                        Task { await vm.search() }
                    } label: {
                        if vm.isSearching {
                            ProgressView().tint(.white)
                        } else {
                            Image(systemName: "magnifyingglass")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(vm.isSearching || !vm.isModelReady)
                }
                .padding()

                // Results
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {

                        // Verses
                        if !vm.verses.isEmpty {
                            Text("📖 KJV Verses")
                                .font(.headline)
                                .padding(.top, 4)

                            ForEach(vm.verses) { v in
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(v.reference)
                                        .font(.caption.weight(.semibold))
                                        .foregroundColor(.accentColor)
                                    Text(v.text)
                                        .font(.body)
                                }
                                .padding()
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color(.secondarySystemBackground))
                                .cornerRadius(10)
                            }
                        }

                        // AI Insight
                        if !vm.aiSummary.isEmpty {
                            VStack(alignment: .leading, spacing: 8) {
                                Label("AI Insight", systemImage: "sparkles")
                                    .font(.headline)
                                Text(vm.aiSummary)
                                    .font(.body)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            .padding()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.accentColor.opacity(0.08))
                            .cornerRadius(10)
                        }

                        // Empty state
                        if vm.verses.isEmpty && vm.aiSummary.isEmpty
                            && !vm.isSearching {
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
                    .padding(.horizontal)
                    .padding(.bottom, 32)
                }
            }
            .navigationTitle("📖 Bible AI")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

import Foundation

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

    let contextLength: Int
    let maxInputTokens: Int
    let vocabSize: Int

    let weightsResourceName: String
    let weightsResourceExtension: String

    let tokenizerFiles: [TokenizerFileSpec]
    let coreMLSpec: CoreMLInputSpec?

    let promptTemplateID: String
    let stopTokenIds: Set<Int>
    let padTokenId: Int
    let fallbackTokenId: Int
    let defaultOptions: GenerationOptions

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

struct GenerationOptions: Sendable {
    var maxNewTokens: Int = 128
    var temperature: Double = 0.7
    var topP: Double = 0.92
    var topK: Int? = nil
    var repetitionPenalty: Double = 1.2
    var softStopTokens: Set<Int> = []
    var seed: UInt64? = nil
}

/// Maps a bundle resource to the filename AutoTokenizer expects in the temp directory.
struct TokenizerFileSpec: Sendable {
    let bundleResource: String    // Bundle.main.url(forResource:) name (no extension)
    let bundleExtension: String   // "json", "txt"
    let targetFilename: String    // filename AutoTokenizer sees (e.g. "tokenizer.json")
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

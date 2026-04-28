import Foundation

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
        .llama3_2_1bInstruct,
        .distilGPT2,
    ]
    static var defaultConfig: LanguageModelConfiguration { .llama3_2_1bInstruct }
}

extension LanguageModelConfiguration {

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
            TokenizerFileSpec(bundleResource: "vocab",            bundleExtension: "json", targetFilename: "vocab.json"),
            TokenizerFileSpec(bundleResource: "merges",           bundleExtension: "txt",  targetFilename: "merges.txt"),
            TokenizerFileSpec(bundleResource: "tokenizer",        bundleExtension: "json", targetFilename: "tokenizer.json"),
            TokenizerFileSpec(bundleResource: "tokenizer_config", bundleExtension: "json", targetFilename: "tokenizer_config.json"),
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
    /// Bundle requires: llama_tokenizer.json + llama_tokenizer_config.json
    static let llama3_2_1bInstruct = LanguageModelConfiguration(
        id: "llama-3.2-1b-coreml",
        displayName: "Llama 3.2 1B (Core ML)",
        architecture: .llama3,
        backend: .coreML,
        contextLength: 131_072,
        maxInputTokens: 512,
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
            maxNewTokens: 150,
            temperature: 0.3,
            topP: 0.9,
            topK: 40,
            repetitionPenalty: 1.3,
            softStopTokens: []
        )
    )
}

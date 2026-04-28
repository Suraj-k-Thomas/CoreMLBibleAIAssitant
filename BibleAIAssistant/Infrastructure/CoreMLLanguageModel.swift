import CoreML

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
        let provider: MLDictionaryFeatureProvider
        switch spec.inputShape {

        case .flat(let padLen):
            // GPT-2 style: flat [padLen] with right-padding and explicit position IDs.
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
            provider = try MLDictionaryFeatureProvider(dictionary: dict)

        case .batched:
            // Llama style: [1, n] token IDs + [1, 1, n, n] additive causal mask.
            let inputArr = try MLMultiArray(shape: [1, n as NSNumber], dataType: .int32)
            for i in 0..<n {
                inputArr[[0, i] as [NSNumber]] = NSNumber(value: seq[i])
            }
            var dict: [String: MLFeatureValue] = [
                spec.inputIdsName: MLFeatureValue(multiArray: inputArr)
            ]
            if case .causalMask(let name) = spec.extraInput {
                // Additive mask: 0 = attend, large negative = block (upper triangle).
                let mask = try MLMultiArray(
                    shape: [1, 1, n as NSNumber, n as NSNumber], dataType: .float32)
                let ptr = mask.dataPointer.assumingMemoryBound(to: Float32.self)
                for row in 0..<n {
                    let base = row * n
                    for col in 0..<n {
                        ptr[base + col] = col <= row ? 0.0 : -65504.0
                    }
                }
                dict[name] = MLFeatureValue(multiArray: mask)
            }
            provider = try MLDictionaryFeatureProvider(dictionary: dict)
        }

        // ── Run inference ─────────────────────────────────────────────
        let out = try await model.prediction(from: provider)
        guard let logits = out.featureValue(for: spec.logitsName)?.multiArrayValue else {
            return configuration.fallbackTokenId
        }

        // ── Extract last-position logits ──────────────────────────────
        let vocab = configuration.vocabSize
        var raw = [Double](repeating: -.infinity, count: vocab)

        switch spec.inputShape {
        case .flat:
            // Shape [1, padLen, vocab, 1, 1] — extract at position n-1.
            let shape = logits.shape.map { $0.intValue }
            guard shape.count >= 3 else { return configuration.fallbackTokenId }
            let seqLen = shape[1]
            let last = min(n - 1, seqLen - 1)
            let base = last * vocab
            for v in 0..<vocab where base + v < logits.count {
                raw[v] = logits[base + v].doubleValue
            }
        case .batched:
            // Shape [1, n, vocab] — last vocab elements are position n-1.
            let start = logits.count - vocab
            guard start >= 0 else { return configuration.fallbackTokenId }
            for v in 0..<vocab {
                raw[v] = logits[start + v].doubleValue
            }
        }

        return sampler.sample(logits: raw, alreadyGenerated: Set(generated), options: options)
    }
}

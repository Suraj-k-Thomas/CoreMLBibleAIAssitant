import Foundation

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

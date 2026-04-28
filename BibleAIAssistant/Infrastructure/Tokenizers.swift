import Foundation
import Tokenizers

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

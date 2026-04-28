import Foundation

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
    /// Build prompt as token IDs directly — bypasses string encoding for special tokens.
    /// Return nil to fall back to encoding format() as a string.
    func tokenize(messages: [ChatMessage],
                  context: [RetrievedChunk],
                  using tokenizer: any TokenizerProtocol) throws -> [Int]?
}

extension PromptTemplate {
    func tokenize(messages: [ChatMessage],
                  context: [RetrievedChunk],
                  using tokenizer: any TokenizerProtocol) throws -> [Int]? { nil }
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

// Default streaming loop — every backend gets streaming for free.
// A backend with a native streaming API (e.g. MLX) can override this.
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

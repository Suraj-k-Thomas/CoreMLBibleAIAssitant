import Foundation

enum RAGEvent: Sendable {
    case retrieving
    case retrieved([RetrievedChunk])
    case generating
    case token(String)      // incremental text delta
    case done(String)       // final assembled text
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

                    // Prefer direct token-ID encoding (Llama 3) to guarantee special token IDs.
                    // Fall back to string encoding for models without a tokenize() override.
                    let promptTokens: [Int]
                    if let direct = try template.tokenize(messages: messages, context: chunks, using: tokenizer) {
                        promptTokens = direct
                    } else {
                        promptTokens = try tokenizer.encode(template.format(messages: messages, context: chunks))
                    }
                    print("📝 Prompt tokens: \(promptTokens.count) (first 5: \(Array(promptTokens.prefix(5))))")

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

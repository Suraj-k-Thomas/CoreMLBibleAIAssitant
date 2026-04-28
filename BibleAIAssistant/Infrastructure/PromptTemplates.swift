import Foundation

/// GPT-2 style completion prompt.
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

/// Llama 3 / 3.1 / 3.2 chat template.
struct Llama3PromptTemplate: PromptTemplate {
    let id = "llama3"

    // Llama 3 special token IDs (fixed across all Llama 3.x models)
    private enum Tok {
        static let bos:         Int = 128000  // <|begin_of_text|>
        static let headerStart: Int = 128006  // <|start_header_id|>
        static let headerEnd:   Int = 128007  // <|end_header_id|>
        static let eot:         Int = 128009  // <|eot_id|>
    }

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

    /// Builds prompt as exact token IDs so special tokens are always correct,
    /// regardless of how the tokenizer handles them in plain strings.
    func tokenize(messages: [ChatMessage],
                  context: [RetrievedChunk],
                  using tokenizer: any TokenizerProtocol) throws -> [Int]? {
        let ctx = context.map { "- \($0.text)" }.joined(separator: "\n")
        let baseSystem = messages.first(where: { $0.role == .system })?.content
            ?? "You are a thoughtful assistant answering questions about the King James Bible."
        let systemContent = context.isEmpty ? baseSystem :
            "\(baseSystem)\n\nUse the following retrieved passages from the KJV as authoritative context. Quote them where relevant.\n\n\(ctx)"

        var tokens: [Int] = [Tok.bos]

        tokens += [Tok.headerStart]
        tokens += try tokenizer.encode("system")
        tokens += [Tok.headerEnd]
        tokens += try tokenizer.encode("\n\n\(systemContent)")
        tokens += [Tok.eot]

        for m in messages where m.role != .system {
            tokens += [Tok.headerStart]
            tokens += try tokenizer.encode(m.role.rawValue)
            tokens += [Tok.headerEnd]
            tokens += try tokenizer.encode("\n\n\(m.content)")
            tokens += [Tok.eot]
        }

        tokens += [Tok.headerStart]
        tokens += try tokenizer.encode("assistant")
        tokens += [Tok.headerEnd]
        tokens += try tokenizer.encode("\n\n")

        return tokens
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
        default:       return PlainPromptTemplate()
        }
    }
}

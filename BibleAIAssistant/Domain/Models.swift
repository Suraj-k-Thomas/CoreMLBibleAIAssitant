import Foundation

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

extension BibleVerse {
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

import Foundation

struct FTS5Retriever: Retriever {
    let database: BibleDatabase

    func retrieve(query: String, k: Int) async -> [RetrievedChunk] {
        let verses = await database.fetchVerses(matching: query, limit: k)
        return verses.map { v in
            RetrievedChunk(
                id: "\(v.book):\(v.chapter):\(v.verse)",
                text: v.text,
                score: 1.0,
                metadata: [
                    "book": "\(v.book)",
                    "chapter": "\(v.chapter)",
                    "verse": "\(v.verse)",
                    "reference": v.reference
                ]
            )
        }
    }
}

// Future: VectorRetriever (embedding-based), HybridRetriever (RRF: FTS5 + vector).

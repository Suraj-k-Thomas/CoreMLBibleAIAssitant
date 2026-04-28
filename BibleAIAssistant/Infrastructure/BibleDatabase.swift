import Foundation
import GRDB

final class BibleDatabase: @unchecked Sendable {
    private var dbQueue: DatabaseQueue?

    init() { setupDatabase() }

    private func setupDatabase() {
        let fm = FileManager.default
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dest = docs.appendingPathComponent("kjv.sqlite")

        if !fm.fileExists(atPath: dest.path),
           let src = Bundle.main.url(forResource: "kjv", withExtension: "sqlite") {
            try? fm.copyItem(at: src, to: dest)
        }
        do {
            dbQueue = try DatabaseQueue(path: dest.path)
            Task { await createFTSIfNeeded() }
        } catch {
            print("❌ DB open failed: \(error)")
        }
    }

    private func createFTSIfNeeded() async {
        guard let q = dbQueue else { return }
        do {
            try await q.write { db in
                let exists = (try Int.fetchOne(
                    db,
                    sql: """
                        SELECT COUNT(*) FROM sqlite_master
                        WHERE type='table' AND name='verses_fts';
                    """
                ) ?? 0) > 0
                guard !exists else { return }

                try db.execute(sql: """
                    CREATE VIRTUAL TABLE verses_fts
                    USING fts5(text, content='verses', content_rowid='id');
                """)
                try db.execute(sql: """
                    INSERT INTO verses_fts(rowid, text)
                    SELECT id, text FROM verses;
                """)
                print("✅ FTS5 ready")
            }
        } catch {
            print("❌ FTS error: \(error)")
        }
    }

    func fetchVerses(matching topic: String, limit: Int) async -> [BibleVerse] {
        guard let q = dbQueue else { return [] }
        do {
            return try await q.read { db in
                let fts = try Row.fetchAll(
                    db,
                    sql: """
                        SELECT v.book, v.chapter, v.verse, v.text
                        FROM verses v
                        JOIN verses_fts ON verses_fts.rowid = v.id
                        WHERE verses_fts MATCH ?
                        ORDER BY rank
                        LIMIT \(limit);
                    """,
                    arguments: ["\"\(topic)\""]
                )
                if !fts.isEmpty {
                    return fts.map {
                        BibleVerse(book: $0["book"], chapter: $0["chapter"],
                                   verse: $0["verse"], rawText: $0["text"])
                    }
                }
                let like = try Row.fetchAll(
                    db,
                    sql: """
                        SELECT book, chapter, verse, text FROM verses
                        WHERE text LIKE ?
                        LIMIT \(limit);
                    """,
                    arguments: ["%\(topic)%"]
                )
                return like.map {
                    BibleVerse(book: $0["book"], chapter: $0["chapter"],
                               verse: $0["verse"], rawText: $0["text"])
                }
            }
        } catch {
            print("❌ Search error: \(error)")
            return []
        }
    }
}

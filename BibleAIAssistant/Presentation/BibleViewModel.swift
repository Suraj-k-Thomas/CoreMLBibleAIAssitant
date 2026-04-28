import SwiftUI

@MainActor
final class BibleViewModel: ObservableObject {
    @Published var verses: [BibleVerse] = []
    @Published var aiSummary: String = ""
    @Published var isStreaming: Bool = false
    @Published var searchTopic: String = ""

    private let container: AppContainer
    private var task: Task<Void, Never>?

    init(container: AppContainer) { self.container = container }

    func search() {
        task?.cancel()
        let topic = searchTopic.trimmingCharacters(in: .whitespaces)
        guard !topic.isEmpty,
              let rag = container.rag,
              let config = container.activeConfig
        else { return }

        isStreaming = true
        aiSummary = ""
        verses = []

        task = Task { [weak self] in
            do {
                let stream = rag.answer(
                    query: topic,
                    systemPrompt:
                        "You are a Bible commentary assistant. " +
                        "You will be given exact King James Bible verses as context. " +
                        "Your ONLY job is to briefly explain what those specific verses mean in relation to the topic. " +
                        "RULES: Only reference the verses provided in the context. " +
                        "Do NOT quote or invent any other Bible verses. " +
                        "Do NOT add citations not present in the context. " +
                        "Keep your answer to 3-4 sentences maximum.",
                    k: 3,
                    options: config.defaultOptions
                )

                for try await event in stream {
                    if Task.isCancelled { break }
                    guard let self else { return }
                    switch event {
                    case .retrieving:
                        break
                    case .retrieved(let chunks):
                        self.verses = chunks.compactMap(BibleVerse.init(chunk:))
                    case .generating:
                        self.aiSummary = ""
                    case .token(let delta):
                        self.aiSummary.append(delta)
                    case .done(let final):
                        if self.aiSummary.isEmpty { self.aiSummary = final }
                    }
                }
            } catch {
                self?.aiSummary = "Error: \(error.localizedDescription)"
            }
            self?.isStreaming = false
        }
    }

    func cancel() { task?.cancel() }
}

import SwiftUI

@MainActor
final class AppContainer: ObservableObject {
    @Published private(set) var availableModels: [LanguageModelConfiguration] = ModelCatalog.all
    @Published private(set) var activeConfig: LanguageModelConfiguration?
    @Published private(set) var isReady: Bool = false
    @Published private(set) var status: String = "Idle"

    let database: BibleDatabase
    let retriever: any Retriever
    private(set) var rag: RAGService?

    private var currentModel: (any LanguageModel)?

    init() {
        let db = BibleDatabase()
        self.database = db
        self.retriever = FTS5Retriever(database: db)
    }

    func switchModel(to config: LanguageModelConfiguration) async {
        isReady = false
        status = "Unloading previous model…"
        if let m = currentModel { await m.unload() }
        currentModel = nil
        rag = nil

        do {
            status = "Loading \(config.displayName)…"
            let model = try LanguageModelFactory.make(for: config)
            try await model.load()
            let tokenizer = try await TokenizerFactory.make(for: config)
            let template = PromptTemplateRegistry.template(id: config.promptTemplateID)

            rag = RAGService(
                retriever: retriever,
                model: model,
                tokenizer: tokenizer,
                template: template,
                sampler: NucleusSampler()
            )
            currentModel = model
            activeConfig = config
            isReady = true
            status = "⚡ \(config.displayName) ready"
        } catch {
            isReady = false
            status = "❌ \(error.localizedDescription)"
        }
    }
}

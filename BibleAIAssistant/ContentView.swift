import SwiftUI

struct ContentView: View {
    @StateObject private var container = AppContainer()

    var body: some View {
        BibleScreen(container: container)
            .task {
                if container.activeConfig == nil {
                    await container.switchModel(to: ModelCatalog.defaultConfig)
                }
            }
    }
}

import SwiftUI

struct BibleScreen: View {
    @ObservedObject var container: AppContainer
    @StateObject private var vm: BibleViewModel

    init(container: AppContainer) {
        self.container = container
        _vm = StateObject(wrappedValue: BibleViewModel(container: container))
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                statusBar
                Divider()
                modelPicker
                Divider()
                searchBar
                results
            }
            .navigationTitle("📖 Bible AI")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var statusBar: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(container.isReady ? Color.green : Color.orange)
                .frame(width: 8, height: 8)
            Text(container.status)
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(Color(.systemBackground))
    }

    private var modelPicker: some View {
        HStack {
            Text("Model").font(.caption).foregroundColor(.secondary)
            Menu {
                ForEach(container.availableModels) { cfg in
                    Button {
                        Task { await container.switchModel(to: cfg) }
                    } label: {
                        if cfg.id == container.activeConfig?.id {
                            Label(cfg.displayName, systemImage: "checkmark")
                        } else {
                            Text(cfg.displayName)
                        }
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    Text(container.activeConfig?.displayName ?? "Select…")
                        .font(.caption.weight(.semibold))
                    Image(systemName: "chevron.down").font(.caption2)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color(.secondarySystemBackground))
                .cornerRadius(8)
            }
            Spacer()
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
    }

    private var searchBar: some View {
        HStack {
            TextField("Search topic (e.g. peace, love, faith)", text: $vm.searchTopic)
                .textFieldStyle(.roundedBorder)
                .onSubmit { vm.search() }
                .disabled(vm.isStreaming)

            Button {
                if vm.isStreaming { vm.cancel() } else { vm.search() }
            } label: {
                if vm.isStreaming {
                    Image(systemName: "stop.fill")
                } else {
                    Image(systemName: "magnifyingglass")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(!container.isReady && !vm.isStreaming)
        }
        .padding()
    }

    private var results: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if !vm.verses.isEmpty {
                    Text("📖 KJV Verses").font(.headline).padding(.top, 4)
                    ForEach(vm.verses) { v in verseCard(v) }
                }
                if !vm.aiSummary.isEmpty {
                    aiInsightCard
                }
                if vm.verses.isEmpty && vm.aiSummary.isEmpty && !vm.isStreaming {
                    emptyState
                }
            }
            .padding(.horizontal)
            .padding(.bottom, 32)
        }
    }

    private func verseCard(_ v: BibleVerse) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(v.reference)
                .font(.caption.weight(.semibold))
                .foregroundColor(.accentColor)
            Text(v.text).font(.body)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground))
        .cornerRadius(10)
    }

    private var aiInsightCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("AI Insight", systemImage: "sparkles").font(.headline)
            Text(vm.aiSummary)
                .font(.body)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.accentColor.opacity(0.08))
        .cornerRadius(10)
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "book.closed")
                .font(.system(size: 48))
                .foregroundColor(.secondary)
            Text("Search the King James Bible")
                .font(.headline)
                .foregroundColor(.secondary)
            Text("Try: love, faith, grace, hope, peace")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 80)
    }
}

import Foundation

enum AppError: LocalizedError {
    case modelFilesNotFound(String)
    case modelNotLoaded
    case tokenizerNotLoaded
    case unsupportedBackend(String)
    case unexpectedModelOutput(String)

    var errorDescription: String? {
        switch self {
        case .modelFilesNotFound(let name): return "Model files not found: \(name)"
        case .modelNotLoaded:                return "Model not loaded yet"
        case .tokenizerNotLoaded:            return "Tokenizer not loaded yet"
        case .unsupportedBackend(let b):     return "Backend not wired up: \(b)"
        case .unexpectedModelOutput(let m):  return "Unexpected model output: \(m)"
        }
    }
}

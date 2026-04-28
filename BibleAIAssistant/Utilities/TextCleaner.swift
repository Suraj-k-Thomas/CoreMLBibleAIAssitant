import Foundation

enum TextCleaner {
    static func clean(_ raw: String, trim: Bool = true) -> String {
        let replaced = raw
            .replacingOccurrences(of: "‹", with: "")
            .replacingOccurrences(of: "›", with: "")
            .replacingOccurrences(of: "<|endoftext|>", with: "")
            .replacingOccurrences(of: "<|begin_of_text|>", with: "")
            .replacingOccurrences(of: "<|eot_id|>", with: "")
            .replacingOccurrences(of: "<|end_of_text|>", with: "")
            .replacingOccurrences(of: "Ġ", with: " ")
            .replacingOccurrences(of: "Ċ", with: "\n")
        return trim ? replaced.trimmingCharacters(in: .whitespacesAndNewlines) : replaced
    }
}

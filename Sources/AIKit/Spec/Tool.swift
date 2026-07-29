import Foundation

/// A tool the model may call.
public struct ToolDefinition: Sendable, Hashable {
    public var name: String
    /// What the tool does — and, just as importantly, *when* to call it.
    ///
    /// Descriptions that state a trigger condition ("call this when the user
    /// asks about current prices") measurably outperform ones that only
    /// describe behaviour. Recent models reach for tools conservatively, and
    /// this text is the main lever on that.
    public var description: String?
    /// JSON Schema for the tool's arguments.
    public var inputSchema: JSONValue
    /// Ask the provider to guarantee the arguments validate against the schema.
    ///
    /// Not universally supported, and usually requires `additionalProperties:
    /// false` plus a complete `required` list.
    public var strict: Bool

    public init(
        name: String,
        description: String? = nil,
        inputSchema: JSONValue,
        strict: Bool = false
    ) {
        self.name = name
        self.description = description
        self.inputSchema = inputSchema
        self.strict = strict
    }
}

/// How insistent the caller is that a tool be used.
public enum ToolChoice: Sendable, Hashable {
    /// The model decides. The default everywhere.
    case auto
    /// Tools are visible but must not be called.
    case none
    /// The model must call at least one tool.
    case required
    /// The model must call this specific tool.
    case tool(name: String)
}

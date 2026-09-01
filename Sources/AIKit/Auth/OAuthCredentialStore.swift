import Foundation
import Security

/// Persistent storage for OAuth credentials.
///
/// The key is deliberately supplied by the caller so one store can hold
/// credentials for several providers or signed-in profiles. Tests use an
/// in-memory implementation; Apple applications can use
/// ``KeychainOAuthCredentialStore`` without writing bearer tokens to files.
public protocol OAuthCredentialStore: Sendable {
    func loadCredential(for key: String) async throws -> OAuthCredential?
    func saveCredential(_ credential: OAuthCredential?, for key: String) async throws
}

public struct OAuthCredentialStoreError: Error, Sendable, CustomStringConvertible {
    public var operation: String
    public var status: OSStatus

    public var description: String {
        let detail = SecCopyErrorMessageString(status, nil) as String? ?? "OSStatus \(status)"
        return "OAuth credential store \(operation) failed: \(detail)"
    }
}

/// Keychain-backed credential persistence for the package's Apple platforms.
///
/// Encoded values are accessible only after the device is unlocked and never
/// synchronize through iCloud. No credential contents are included in errors.
public struct KeychainOAuthCredentialStore: OAuthCredentialStore, Sendable {
    public var service: String
    public var accessGroup: String?

    public init(
        service: String = "AIKitSwift.OAuth",
        accessGroup: String? = nil
    ) {
        self.service = service
        self.accessGroup = accessGroup
    }

    public func loadCredential(for key: String) async throws -> OAuthCredential? {
        var query = baseQuery(key: key)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess else {
            throw OAuthCredentialStoreError(operation: "load", status: status)
        }
        guard let data = result as? Data else {
            throw OAuthCredentialStoreError(operation: "decode", status: errSecDecode)
        }
        return try JSONDecoder().decode(OAuthCredential.self, from: data)
    }

    public func saveCredential(_ credential: OAuthCredential?, for key: String) async throws {
        let query = baseQuery(key: key)
        guard let credential else {
            let status = SecItemDelete(query as CFDictionary)
            guard status == errSecSuccess || status == errSecItemNotFound else {
                throw OAuthCredentialStoreError(operation: "delete", status: status)
            }
            return
        }

        let data = try JSONEncoder().encode(credential)
        let update = [kSecValueData as String: data]
        let updateStatus = SecItemUpdate(query as CFDictionary, update as CFDictionary)
        if updateStatus == errSecSuccess { return }
        guard updateStatus == errSecItemNotFound else {
            throw OAuthCredentialStoreError(operation: "update", status: updateStatus)
        }

        var addition = query
        addition[kSecValueData as String] = data
        addition[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        let addStatus = SecItemAdd(addition as CFDictionary, nil)
        guard addStatus == errSecSuccess else {
            throw OAuthCredentialStoreError(operation: "save", status: addStatus)
        }
    }

    private func baseQuery(key: String) -> [String: Any] {
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecAttrSynchronizable as String: kCFBooleanFalse as Any,
        ]
        if let accessGroup {
            query[kSecAttrAccessGroup as String] = accessGroup
        }
        return query
    }
}

/// An async credential source used by ``AIClient``.
public protocol OAuthCredentialProviding: Sendable {
    /// Returns a usable credential, refreshing an expired credential if needed.
    func credential() async throws -> OAuthCredential
    /// Refreshes after the provider rejected `accessToken`. If another caller
    /// already replaced that token, its result is returned without refreshing
    /// again.
    func credential(afterRejecting accessToken: String) async throws -> OAuthCredential
}

public enum OAuthCredentialManagerError: Error, Sendable, LocalizedError {
    case missingCredential
    case missingRefreshToken

    public var errorDescription: String? {
        switch self {
        case .missingCredential: "No persisted OAuth credential is available."
        case .missingRefreshToken: "The OAuth credential cannot be refreshed because it has no refresh token."
        }
    }
}

/// Resolves one wait from either a shared task result or cancellation of the
/// individual caller. Cancelling the waiter never cancels the shared task.
private actor OAuthTaskWaiter<Value: Sendable> {
    private var continuation: CheckedContinuation<Value, any Error>?
    private var pendingResult: Result<Value, any Error>?

    func wait() async throws -> Value {
        try await withCheckedThrowingContinuation { continuation in
            if let pendingResult {
                self.pendingResult = nil
                continuation.resume(with: pendingResult)
            } else {
                self.continuation = continuation
            }
        }
    }

    func resolve(_ result: Result<Value, any Error>) {
        guard let continuation else {
            if pendingResult == nil { pendingResult = result }
            return
        }
        self.continuation = nil
        continuation.resume(with: result)
    }

    func cancel() {
        resolve(.failure(CancellationError()))
    }
}

private func cancellationResponsiveValue<Value: Sendable>(
    of task: Task<Value, any Error>
) async throws -> Value {
    try Task.checkCancellation()
    let waiter = OAuthTaskWaiter<Value>()
    let observer = Task {
        await waiter.resolve(await task.result)
    }
    defer { observer.cancel() }

    return try await withTaskCancellationHandler {
        try await waiter.wait()
    } onCancel: {
        Task { await waiter.cancel() }
    }
}

/// Owns one persisted credential and coalesces refresh operations.
///
/// This actor is intentionally transport-agnostic: the provider-specific Auth
/// implementation supplies the refresh closure, while the client depends only
/// on ``OAuthCredentialProviding``. A detached caller cancelling its wait does
/// not cancel a shared refresh needed by other simultaneous requests.
public actor OAuthCredentialManager: OAuthCredentialProviding {
    public typealias Refresh = @Sendable (OAuthCredential) async throws -> OAuthCredential

    private let key: String
    private let store: any OAuthCredentialStore
    private let refresh: Refresh
    private let now: @Sendable () -> Date
    private let refreshLeeway: TimeInterval

    private var current: OAuthCredential?
    private var didLoad = false
    private var nextTaskId = 0
    private var loadTask: (id: Int, task: Task<OAuthCredential?, any Error>)?
    private var refreshTask: (id: Int, task: Task<OAuthCredential, any Error>)?
    private var initialSaveTask: (id: Int, task: Task<Void, any Error>)?
    private var mutationTask: (id: Int, task: Task<OAuthCredential?, any Error>)?
    private var initialCredentialNeedsSave: Bool

    public init(
        key: String,
        initialCredential: OAuthCredential? = nil,
        store: any OAuthCredentialStore,
        refreshLeeway: TimeInterval = 300,
        now: @escaping @Sendable () -> Date = Date.init,
        refresh: @escaping Refresh
    ) {
        self.key = key
        self.current = initialCredential
        self.didLoad = initialCredential != nil
        self.initialCredentialNeedsSave = initialCredential != nil
        self.store = store
        self.refreshLeeway = refreshLeeway
        self.now = now
        self.refresh = refresh
    }

    public func credential() async throws -> OAuthCredential {
        try Task.checkCancellation()
        try await waitForCredentialMutation()
        try await persistInitialCredentialIfNeeded()
        try await waitForCredentialMutation()
        let credential = try await loadedCredential()
        guard credential.needsRefresh(now: now(), leeway: refreshLeeway) else {
            try Task.checkCancellation()
            return credential
        }
        return try await refreshCredential(credential)
    }

    public func credential(afterRejecting accessToken: String) async throws -> OAuthCredential {
        try Task.checkCancellation()
        try await waitForCredentialMutation()
        try await persistInitialCredentialIfNeeded()
        try await waitForCredentialMutation()
        let credential = try await loadedCredential()
        // A simultaneous request already refreshed the rejected token.
        guard credential.accessToken == accessToken else {
            try Task.checkCancellation()
            return credential
        }
        return try await refreshCredential(credential)
    }

    /// Installs and persists a newly authorized credential before making it
    /// visible to request callers.
    public func install(_ credential: OAuthCredential) async throws {
        try Task.checkCancellation()
        try await waitForCredentialMutation()
        startCredentialMutation(replacingWith: credential)
        try await waitForCredentialMutation()
    }

    /// Removes the persisted credential and clears the actor's loaded value.
    public func deleteCredential() async throws {
        try Task.checkCancellation()
        try await waitForCredentialMutation()
        startCredentialMutation(replacingWith: nil)
        try await waitForCredentialMutation()
    }

    private func persistInitialCredentialIfNeeded() async throws {
        try await waitForCredentialMutation()
        guard initialCredentialNeedsSave, let current else { return }

        let inFlight: (id: Int, task: Task<Void, any Error>)
        if let initialSaveTask {
            inFlight = initialSaveTask
        } else {
            try Task.checkCancellation()
            let key = self.key
            let store = self.store
            let created = Task { try await store.saveCredential(current, for: key) }
            let id = taskId()
            inFlight = (id, created)
            initialSaveTask = inFlight
        }

        do {
            try await cancellationResponsiveValue(of: inFlight.task)
            if initialSaveTask?.id == inFlight.id {
                initialSaveTask = nil
                initialCredentialNeedsSave = false
            }
        } catch {
            if !(error is CancellationError && Task.isCancelled),
               initialSaveTask?.id == inFlight.id {
                initialSaveTask = nil
            }
            throw error
        }
    }

    private func loadedCredential() async throws -> OAuthCredential {
        try await waitForCredentialMutation()
        if let current, didLoad { return current }

        let inFlight: (id: Int, task: Task<OAuthCredential?, any Error>)
        if let loadTask {
            inFlight = loadTask
        } else {
            let key = self.key
            let store = self.store
            let created = Task { try await store.loadCredential(for: key) }
            let id = taskId()
            inFlight = (id, created)
            loadTask = inFlight
        }

        do {
            let loaded = try await cancellationResponsiveValue(of: inFlight.task)
            if loadTask?.id == inFlight.id {
                loadTask = nil
                didLoad = true
                current = loaded
            }
            try Task.checkCancellation()
            guard let loaded = current ?? loaded else {
                throw OAuthCredentialManagerError.missingCredential
            }
            return loaded
        } catch {
            if !(error is CancellationError && Task.isCancelled),
               loadTask?.id == inFlight.id {
                loadTask = nil
            }
            throw error
        }
    }

    private func refreshCredential(_ credential: OAuthCredential) async throws -> OAuthCredential {
        try Task.checkCancellation()
        try await waitForCredentialMutation()
        guard let latest = current else {
            throw OAuthCredentialManagerError.missingCredential
        }
        if latest.accessToken != credential.accessToken {
            guard latest.needsRefresh(now: now(), leeway: refreshLeeway) else {
                return latest
            }
            return try await refreshCredential(latest)
        }
        let inFlight: (id: Int, task: Task<OAuthCredential, any Error>)
        if let refreshTask {
            inFlight = refreshTask
        } else {
            guard let refreshToken = credential.refreshToken,
                  !refreshToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                throw OAuthCredentialManagerError.missingRefreshToken
            }
            let refresh = self.refresh
            let store = self.store
            let key = self.key
            let created = Task {
                let refreshed = try await refresh(credential)
                try Task.checkCancellation()
                try await store.saveCredential(refreshed, for: key)
                return refreshed
            }
            let id = taskId()
            inFlight = (id, created)
            refreshTask = inFlight
        }

        do {
            let refreshed = try await cancellationResponsiveValue(of: inFlight.task)
            if refreshTask?.id == inFlight.id {
                current = refreshed
                refreshTask = nil
            }
            try Task.checkCancellation()
            return refreshed
        } catch {
            if !(error is CancellationError && Task.isCancelled),
               refreshTask?.id == inFlight.id {
                refreshTask = nil
            }
            throw error
        }
    }

    private func taskId() -> Int {
        nextTaskId &+= 1
        return nextTaskId
    }

    /// Serializes explicit install/delete mutations after every credential task
    /// that was already in flight. New credential callers wait for the mutation,
    /// so a late load, initial save, or refresh can never overwrite it.
    private func startCredentialMutation(replacingWith credential: OAuthCredential?) {
        let initialSave = initialSaveTask?.task
        let load = loadTask?.task
        let refresh = refreshTask?.task
        let store = self.store
        let key = self.key
        let created = Task<OAuthCredential?, any Error> {
            if let initialSave { _ = await initialSave.result }
            if let load { _ = await load.result }
            if let refresh { _ = await refresh.result }
            try await store.saveCredential(credential, for: key)
            return credential
        }
        mutationTask = (taskId(), created)
    }

    private func waitForCredentialMutation() async throws {
        while let inFlight = mutationTask {
            do {
                let replacement = try await cancellationResponsiveValue(of: inFlight.task)
                if mutationTask?.id == inFlight.id {
                    mutationTask = nil
                    current = replacement
                    didLoad = true
                    initialCredentialNeedsSave = false
                    initialSaveTask = nil
                    loadTask = nil
                    refreshTask = nil
                }
            } catch {
                if !(error is CancellationError && Task.isCancelled),
                   mutationTask?.id == inFlight.id {
                    mutationTask = nil
                }
                throw error
            }
        }
    }
}

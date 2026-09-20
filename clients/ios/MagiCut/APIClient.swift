import Foundation
import SwiftUI
import UIKit

struct APIConfig {
    /// Point this at your SSH GPU host reverse-proxy (HTTPS).
    /// Example: https://gpu.example.com
    static var baseURL = URL(string: ProcessInfo.processInfo.environment["MAGICUT_API_BASE"] ?? "http://127.0.0.1:8080")!
    static var token: String? = ProcessInfo.processInfo.environment["MAGICUT_API_TOKEN"]
}

enum MagiCutAPIError: LocalizedError {
    case badStatus(Int, String)
    case decoding
    case missingResult

    var errorDescription: String? {
        switch self {
        case .badStatus(let code, let body): return "HTTP \(code): \(body)"
        case .decoding: return "Failed to decode server response"
        case .missingResult: return "Result video not ready"
        }
    }
}

struct JobCreateResponse: Decodable {
    let job_id: String
    let status: String
    let filename: String
}

struct JobStatusResponse: Decodable {
    let job_id: String
    let status: String
    let progress: Double
    let stage: String
    let error: String?
    let result_ready: Bool
    let mode: String?
    let elapsed_sec: Double?
    let total_frames: Int?
}

struct ProcessPoint: Encodable {
    let x: Double
    let y: Double
}

struct ProcessBody: Encodable {
    let keyframe_idx: Int
    let points: [ProcessPoint]
    let shadow_dilation: Int
    let max_frames: Int?
}

final class MagiCutAPIClient {
    private let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    private func authorized(_ request: inout URLRequest) {
        if let token = APIConfig.token, !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
    }

    func createDemoJob() async throws -> JobCreateResponse {
        var request = URLRequest(url: APIConfig.baseURL.appendingPathComponent("/api/v1/jobs/demo"))
        request.httpMethod = "POST"
        authorized(&request)
        let (data, response) = try await session.data(for: request)
        try validate(response, data: data)
        return try JSONDecoder().decode(JobCreateResponse.self, from: data)
    }

    func createJob(videoURL: URL) async throws -> JobCreateResponse {
        var request = URLRequest(url: APIConfig.baseURL.appendingPathComponent("/api/v1/jobs"))
        request.httpMethod = "POST"
        authorized(&request)

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        let filename = videoURL.lastPathComponent
        let fileData = try Data(contentsOf: videoURL)
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"video\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: video/mp4\r\n\r\n".data(using: .utf8)!)
        body.append(fileData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body

        let (data, response) = try await session.data(for: request)
        try validate(response, data: data)
        return try JSONDecoder().decode(JobCreateResponse.self, from: data)
    }

    func keyframe(jobId: String, frame: Int) async throws -> UIImage {
        var request = URLRequest(url: APIConfig.baseURL.appendingPathComponent("/api/v1/jobs/\(jobId)/keyframe?frame=\(frame)"))
        authorized(&request)
        let (data, response) = try await session.data(for: request)
        try validate(response, data: data)
        guard let image = UIImage(data: data) else { throw MagiCutAPIError.decoding }
        return image
    }

    func process(jobId: String, body: ProcessBody) async throws -> JobStatusResponse {
        var request = URLRequest(url: APIConfig.baseURL.appendingPathComponent("/api/v1/jobs/\(jobId)/process"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        authorized(&request)
        request.httpBody = try JSONEncoder().encode(body)
        let (data, response) = try await session.data(for: request)
        try validate(response, data: data)
        return try JSONDecoder().decode(JobStatusResponse.self, from: data)
    }

    func status(jobId: String) async throws -> JobStatusResponse {
        var request = URLRequest(url: APIConfig.baseURL.appendingPathComponent("/api/v1/jobs/\(jobId)"))
        authorized(&request)
        let (data, response) = try await session.data(for: request)
        try validate(response, data: data)
        return try JSONDecoder().decode(JobStatusResponse.self, from: data)
    }

    func resultURL(jobId: String) -> URL {
        APIConfig.baseURL.appendingPathComponent("/api/v1/jobs/\(jobId)/result")
    }

    private func validate(_ response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw MagiCutAPIError.badStatus(http.statusCode, body)
        }
    }
}

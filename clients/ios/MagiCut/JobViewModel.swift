import Foundation
import SwiftUI
import PhotosUI
import AVKit

@MainActor
final class JobViewModel: ObservableObject {
    @Published var statusText = "选择一段群舞视频开始"
    @Published var keyframe: UIImage?
    @Published var points: [CGPoint] = []
    @Published var progress: Double = 0
    @Published var stage = ""
    @Published var resultPlayer: AVPlayer?
    @Published var isBusy = false
    @Published var frameIndex = 0

    private let api = MagiCutAPIClient()
    private var jobId: String?
    private var pollTask: Task<Void, Never>?

    func upload(videoURL: URL) async {
        isBusy = true
        defer { isBusy = false }
        do {
            statusText = "上传中…"
            let created = try await api.createJob(videoURL: videoURL)
            jobId = created.job_id
            points = []
            statusText = "已上传 · \(created.job_id)"
            try await reloadKeyframe()
        } catch {
            statusText = error.localizedDescription
        }
    }

    func reloadKeyframe() async throws {
        guard let jobId else { return }
        keyframe = try await api.keyframe(jobId: jobId, frame: frameIndex)
        statusText = "点击主角位置"
    }

    func addPoint(_ point: CGPoint, in size: CGSize, imageSize: CGSize) {
        // Map view tap into image pixel space (aspect-fit approximation)
        let viewRatio = size.width / size.height
        let imgRatio = imageSize.width / imageSize.height
        var content = size
        var origin = CGPoint.zero
        if imgRatio > viewRatio {
            content.height = size.width / imgRatio
            origin.y = (size.height - content.height) / 2
        } else {
            content.width = size.height * imgRatio
            origin.x = (size.width - content.width) / 2
        }
        let local = CGPoint(x: point.x - origin.x, y: point.y - origin.y)
        guard local.x >= 0, local.y >= 0, local.x <= content.width, local.y <= content.height else { return }
        let px = local.x / content.width * imageSize.width
        let py = local.y / content.height * imageSize.height
        points.append(CGPoint(x: px, y: py))
    }

    func clearPoints() { points.removeAll() }

    func startProcessing() async {
        guard let jobId, !points.isEmpty else { return }
        isBusy = true
        do {
            statusText = "提交任务…"
            _ = try await api.process(
                jobId: jobId,
                body: ProcessBody(
                    keyframe_idx: frameIndex,
                    points: points.map { ProcessPoint(x: $0.x, y: $0.y) },
                    shadow_dilation: 25,
                    max_frames: 60
                )
            )
            pollTask?.cancel()
            pollTask = Task { await poll(jobId: jobId) }
        } catch {
            statusText = error.localizedDescription
            isBusy = false
        }
    }

    private func poll(jobId: String) async {
        while !Task.isCancelled {
            do {
                let job = try await api.status(jobId: jobId)
                progress = job.progress
                stage = job.stage
                statusText = "\(job.status) · \(Int(job.progress * 100))%"
                if job.status == "completed", job.result_ready {
                    resultPlayer = AVPlayer(url: api.resultURL(jobId: jobId))
                    statusText = "完成 · mode=\(job.mode ?? "?")"
                    isBusy = false
                    return
                }
                if job.status == "failed" {
                    statusText = job.error ?? "failed"
                    isBusy = false
                    return
                }
            } catch {
                statusText = error.localizedDescription
            }
            try? await Task.sleep(nanoseconds: 700_000_000)
        }
    }
}

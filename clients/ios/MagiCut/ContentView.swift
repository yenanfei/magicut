import SwiftUI
import PhotosUI
import AVKit
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var model: JobViewModel
    @State private var pickerItem: PhotosPickerItem?

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color(red: 0.93, green: 0.96, blue: 0.98), Color(red: 0.84, green: 0.90, blue: 0.93)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Text("MagiCut")
                        .font(.custom("Avenir Next", size: 44).weight(.heavy))
                        .tracking(-1)

                    Text("把群舞变成纯净单人直拍")
                        .font(.title3.weight(.semibold))

                    Text("上传 → 点选主角 → 云端 GPU 生成")
                        .foregroundStyle(.secondary)

                    PhotosPicker(selection: $pickerItem, matching: .videos) {
                        Text("从相册选择视频")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color(red: 0.89, green: 0.24, blue: 0.17))
                            .foregroundStyle(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                    }

                    if let image = model.keyframe {
                        KeyframeTapView(image: image, points: $model.points) { point, size, imageSize in
                            model.addPoint(point, in: size, imageSize: imageSize)
                        }
                        .frame(height: 240)
                        .clipShape(RoundedRectangle(cornerRadius: 14))

                        HStack {
                            Stepper("关键帧 \(model.frameIndex)", value: $model.frameIndex, in: 0...500)
                            Button("加载") {
                                Task {
                                    try? await model.reloadKeyframe()
                                    model.clearPoints()
                                }
                            }
                        }

                        HStack {
                            Button("清除点") { model.clearPoints() }
                            Button("开始生成") {
                                Task { await model.startProcessing() }
                            }
                            .disabled(model.points.isEmpty || model.isBusy)
                            .buttonStyle(.borderedProminent)
                            .tint(Color(red: 0.89, green: 0.24, blue: 0.17))
                        }
                    }

                    if model.isBusy || model.progress > 0 {
                        ProgressView(value: model.progress)
                        Text(model.stage.isEmpty ? model.statusText : model.stage)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    if let player = model.resultPlayer {
                        VideoPlayer(player: player)
                            .frame(height: 240)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                    }

                    Text(model.statusText)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .padding(20)
            }
        }
        .onChange(of: pickerItem) { _, item in
            guard let item else { return }
            Task {
                if let movie = try? await item.loadTransferable(type: VideoFile.self) {
                    await model.upload(videoURL: movie.url)
                }
            }
        }
    }
}

struct VideoFile: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { file in
            SentTransferredFile(file.url)
        } importing: { received in
            let temp = FileManager.default.temporaryDirectory.appendingPathComponent(received.file.lastPathComponent)
            try? FileManager.default.removeItem(at: temp)
            try FileManager.default.copyItem(at: received.file, to: temp)
            return Self(url: temp)
        }
    }
}

struct KeyframeTapView: View {
    let image: UIImage
    @Binding var points: [CGPoint]
    let onTap: (CGPoint, CGSize, CGSize) -> Void

    var body: some View {
        GeometryReader { geo in
            ZStack {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(width: geo.size.width, height: geo.size.height)
                ForEach(Array(points.enumerated()), id: \.offset) { _, p in
                    Circle()
                        .fill(Color(red: 0.89, green: 0.24, blue: 0.17))
                        .frame(width: 12, height: 12)
                        .position(displayPoint(p, in: geo.size, imageSize: image.size))
                }
            }
            .contentShape(Rectangle())
            .onTapGesture { location in
                onTap(location, geo.size, image.size)
            }
        }
    }

    private func displayPoint(_ p: CGPoint, in size: CGSize, imageSize: CGSize) -> CGPoint {
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
        return CGPoint(
            x: origin.x + p.x / imageSize.width * content.width,
            y: origin.y + p.y / imageSize.height * content.height
        )
    }
}

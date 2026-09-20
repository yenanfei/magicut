import SwiftUI

struct SettingsView: View {
    @AppStorage("magicut_api_base") private var apiBase = "http://127.0.0.1:8080"
    @AppStorage("magicut_api_token") private var apiToken = ""
    @State private var healthText = ""

    var body: some View {
        Form {
            Section("云端 API") {
                TextField("https://magicut.example.com", text: $apiBase)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .keyboardType(.URL)
                SecureField("Bearer Token（可选）", text: $apiToken)
                Button("检测连接") {
                    Task { await ping() }
                }
                if !healthText.isEmpty {
                    Text(healthText).font(.footnote).foregroundStyle(.secondary)
                }
            }
            Section("说明") {
                Text("生产环境请使用 HTTPS 域名，并把 Token 配成与 GPU 服务器 MAGICUT_API_TOKEN 一致。")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("设置")
        .onDisappear {
            if let url = URL(string: apiBase) {
                APIConfig.baseURL = url
            }
            APIConfig.token = apiToken.isEmpty ? nil : apiToken
        }
    }

    private func ping() async {
        guard let url = URL(string: apiBase.trimmingCharacters(in: .whitespacesAndNewlines) + "/health") else {
            healthText = "无效 URL"
            return
        }
        var req = URLRequest(url: url)
        if !apiToken.isEmpty {
            req.setValue("Bearer \(apiToken)", forHTTPHeaderField: "Authorization")
        }
        do {
            let (data, _) = try await URLSession.shared.data(for: req)
            if let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                let mode = obj["pipeline_mode"] as? String ?? "?"
                let cuda = obj["cuda_available"] as? Bool ?? false
                healthText = "ok · mode=\(mode) · cuda=\(cuda)"
            } else {
                healthText = "ok"
            }
        } catch {
            healthText = error.localizedDescription
        }
    }
}

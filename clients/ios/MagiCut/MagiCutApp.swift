import SwiftUI

@main
struct MagiCutApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(JobViewModel())
        }
    }
}

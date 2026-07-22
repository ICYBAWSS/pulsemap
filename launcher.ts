// Double-click launcher: serves the PulseMap folder on localhost and opens
// the default browser to it. Compiled via `deno compile` into a single
// dependency-free executable -- the end user needs no Deno/Node/Python
// installed, and never sees a terminal.

const MIME: Record<string, string> = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".css": "text/css",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
};

/** Where the app's static files (index.html, app.js, ...) live. Robust to
 * both `deno run launcher.ts` (dev) and a compiled standalone binary
 * (production): prefer the script's own folder, fall back to the compiled
 * executable's folder, fall back to the current working directory. */
async function findAssetsDir(): Promise<string> {
    const candidates: string[] = [];
    try {
        candidates.push(new URL(".", import.meta.url).pathname);
    } catch { /* ignore */ }
    try {
        candidates.push(new URL(".", `file://${Deno.execPath()}`).pathname);
    } catch { /* ignore */ }
    candidates.push(Deno.cwd() + "/");

    for (const dir of candidates) {
        try {
            const st = await Deno.stat(dir + "index.html");
            if (st.isFile) return dir;
        } catch { /* try next candidate */ }
    }
    return Deno.cwd() + "/"; // last resort
}

async function openBrowser(url: string) {
    const cmd = Deno.build.os === "darwin" ? ["open", url]
        : Deno.build.os === "windows" ? ["cmd", "/c", "start", url]
        : ["xdg-open", url];
    try {
        await new Deno.Command(cmd[0], { args: cmd.slice(1) }).output();
    } catch (e) {
        console.error("Could not auto-open a browser:", e);
        console.error(`Open this URL manually: ${url}`);
    }
}

async function main() {
    const root = await findAssetsDir();

    const server = Deno.serve(
        { port: 0, hostname: "127.0.0.1", onListen: () => {} },
        async (req) => {
            const url = new URL(req.url);
            let path = decodeURIComponent(url.pathname);
            if (path === "/") path = "/index.html";
            // Prevent escaping the assets root.
            const normalized = path.replace(/\.\.+/g, "");
            const filePath = root + normalized.replace(/^\//, "");

            try {
                const data = await Deno.readFile(filePath);
                const ext = filePath.slice(filePath.lastIndexOf("."));
                return new Response(data, {
                    headers: { "content-type": MIME[ext] ?? "application/octet-stream" },
                });
            } catch {
                return new Response("Not found", { status: 404 });
            }
        },
    );

    const port = (server.addr as Deno.NetAddr).port;
    const appUrl = `http://127.0.0.1:${port}/index.html`;
    console.log(`PulseMap running at ${appUrl}`);
    console.log(`Serving from: ${root}`);
    await openBrowser(appUrl);
    await server.finished;
}

main();

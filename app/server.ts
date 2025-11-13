import { serve } from "bun";
import { readFile } from "fs/promises";
import path from "path";

const ROOT = path.resolve(import.meta.dir);

serve({
  port: 3000,
  async fetch(req) {
    const url = new URL(req.url);
    const filePath = url.pathname;

    // Serve manifest.json
    if (filePath === "/manifest.json") {
      const data = await readFile(
        path.join(ROOT, "out/saved_unique_crops/manifest.json"),
        "utf-8"
      );
      return new Response(data, {
        headers: { "Content-Type": "application/json" },
      });
    }

    // Serve images
    if (filePath.startsWith("/out/")) {
      try {
        const imgPath = path.join(ROOT, filePath);
        const img = await readFile(imgPath);
        const ext = path.extname(imgPath).toLowerCase();
        const mime =
          ext === ".jpg" || ext === ".jpeg"
            ? "image/jpeg"
            : ext === ".png"
            ? "image/png"
            : "application/octet-stream";
        return new Response(img, { headers: { "Content-Type": mime } });
      } catch {
        return new Response("Image not found", { status: 404 });
      }
    }

    // Serve index.html for all other routes
    if (filePath === "/" || filePath === "/index.html") {
      const html = await readFile(path.join(ROOT, "index.html"), "utf-8");
      return new Response(html, { headers: { "Content-Type": "text/html" } });
    }

    return new Response("Not found", { status: 404 });
  },
});

console.log("✅ Server running at http://localhost:3000");

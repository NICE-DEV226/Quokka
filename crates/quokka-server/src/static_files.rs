//! Web UI embarquée : les assets Vite compilés (`ui/dist`) sont inclus
//! dans le binaire via rust-embed. En debug, `debug-embed` lit depuis le
//! disque pour itérer sans rebuild.

use axum::extract::Path;
use axum::http::{StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "../../ui/dist"]
struct UiAssets;

fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") | Some("mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("woff2") => "font/woff2",
        Some("json") | Some("map") => "application/json",
        _ => "application/octet-stream",
    }
}

const PLACEHOLDER: &str = "<!doctype html><title>Quokka</title>\
<p>Web UI not built. Run <code>cd ui && npm install && npm run build</code>, \
or use the API directly at <code>/v1/chat/completions</code>.</p>";

/// `GET /` — sert `index.html` (ou un placeholder si `ui/dist` absent).
pub async fn index() -> Response {
    match UiAssets::get("index.html") {
        Some(file) => {
            Html(String::from_utf8_lossy(file.data.as_ref()).into_owned()).into_response()
        }
        None => Html(PLACEHOLDER.to_string()).into_response(),
    }
}

/// Fallback GET pour les assets (`/assets/*`, `/mascot.svg`, …).
/// Les chemins `/v1/*` inconnus restent des 404 JSON.
pub async fn asset(Path(path): Path<String>) -> Response {
    if path.starts_with("v1/") {
        return StatusCode::NOT_FOUND.into_response();
    }
    match UiAssets::get(&path) {
        Some(file) => ([(header::CONTENT_TYPE, content_type(&path))], file.data).into_response(),
        // SPA : toute route inconnue retombe sur index.html.
        None => index().await,
    }
}

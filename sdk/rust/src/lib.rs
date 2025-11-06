//! Tessrax Rust SDK — v19.6
//! Minimal async client for governance API.

use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;

pub struct TessraxClient {
    base: String,
    http: Client,
}

impl TessraxClient {
    pub fn new(base: &str) -> Self {
        TessraxClient {
            base: base.to_string(),
            http: Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
        }
    }

    pub async fn emit(&self, payload: Value) -> Result<Value, reqwest::Error> {
        let resp = self
            .http
            .post(format!("{}/api/receipts", self.base))
            .json(&payload)
            .send()
            .await?;
        Ok(resp.json::<Value>().await?)
    }

    pub async fn verify(&self, receipt: Value) -> Result<bool, reqwest::Error> {
        let resp = self
            .http
            .post(format!("{}/api/verify", self.base))
            .json(&receipt)
            .send()
            .await?;
        let body = resp.json::<Value>().await?;
        Ok(body["verified"].as_bool().unwrap_or(false))
    }

    pub async fn status(&self, window: usize) -> Result<Value, reqwest::Error> {
        let resp = self
            .http
            .get(format!("{}/api/status?window={}", self.base, window))
            .send()
            .await?;
        Ok(resp.json::<Value>().await?)
    }
}

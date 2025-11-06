/**
 * Tessrax JS SDK v19.6 — minimal client for browser or Node.js
 */
const BASE_URL = "http://localhost:8080";

export async function emitReceipt(payload) {
  const r = await fetch(`${BASE_URL}/api/receipts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

export async function verifyReceipt(receipt) {
  const r = await fetch(`${BASE_URL}/api/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(receipt),
  });
  if (!r.ok) throw new Error(await r.text());
  const { verified } = await r.json();
  return verified;
}

export async function getStatus(window = 100) {
  const r = await fetch(`${BASE_URL}/api/status?window=${window}`);
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

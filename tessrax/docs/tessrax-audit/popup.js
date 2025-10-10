// === Tessrax Chrome Extension ===
// Replace PROMPT below with your full TMP-1 contradiction sweep prompt string
const TMP1_PROMPT = `[YOUR TMP-1 CONTRADICTION SWEEP PROMPT]`;

document.getElementById('copy').addEventListener('click', () => {
  navigator.clipboard.writeText(TMP1_PROMPT);
  document.getElementById('status').textContent = '✓ Prompt copied! Paste into Claude.';
});

document.getElementById('audit').addEventListener('click', () => {
  chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
    chrome.scripting.executeScript({
      target: {tabId: tabs[0].id},
      function: extractConversation
    });
  });
});

function extractConversation() {
  const messages = document.querySelectorAll('.message'); // adjust selector per site
  const conversation = Array.from(messages).map(m => m.textContent).join('\n\n');
  const auditURL = `https://claude.ai/new?prompt=${encodeURIComponent(TMP1_PROMPT + '\n\n' + conversation)}`;
  window.open(auditURL, '_blank');
}
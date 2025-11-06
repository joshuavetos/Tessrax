// Tessrax Go SDK — v19.6
// Provides minimal client for governance API.
package tessrax

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strconv"
	"time"
)

var BaseURL = "http://localhost:8080"

type Receipt map[string]interface{}

func Emit(payload Receipt) (Receipt, error) {
	buf, _ := json.Marshal(payload)
	resp, err := http.Post(BaseURL+"/api/receipts", "application/json", bytes.NewBuffer(buf))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out Receipt
	json.NewDecoder(resp.Body).Decode(&out)
	return out, nil
}

func Verify(receipt Receipt) (bool, error) {
	buf, _ := json.Marshal(receipt)
	resp, err := http.Post(BaseURL+"/api/verify", "application/json", bytes.NewBuffer(buf))
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	var out struct {
		Verified bool `json:"verified"`
	}
	json.NewDecoder(resp.Body).Decode(&out)
	return out.Verified, nil
}

func Status(window int) (Receipt, error) {
	client := http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(BaseURL + "/api/status?window=" + strconv.Itoa(window))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out Receipt
	json.NewDecoder(resp.Body).Decode(&out)
	return out, nil
}

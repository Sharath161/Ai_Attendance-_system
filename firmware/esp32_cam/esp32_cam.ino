/*
 * AI Attendance System — ESP32-CAM Firmware
 *
 * Flow:
 *   1. Connect to campus WiFi
 *   2. Sync time via NTP
 *   3. On trigger (PIR or timer), capture JPEG
 *   4. POST to server /ingest/image with X-Device-Key header
 *   5. Show result on OLED + LED
 *   6. Sleep / repeat
 *
 * Hardware: AI-Thinker ESP32-CAM (OV2640) or ESP32-S3 (OV5640)
 * Libraries needed (install via Arduino Library Manager):
 *   - Adafruit SSD1306
 *   - Adafruit GFX Library
 *   - ArduinoJson
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include "esp_camera.h"
#include <time.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ArduinoJson.h>
#include "esp_sleep.h"

// ── Configuration — edit these before flashing ────────────────────────────────
// WiFi
#define WIFI_SSID       "UniversityWiFi"
#define WIFI_PASSWORD   "your_wifi_password"

// Server (on-premise, no HTTPS needed on LAN)
#define SERVER_HOST     "192.168.1.100"
#define SERVER_PORT     8000
#define INGEST_PATH     "/ingest/image"

// Device identity — copy the api_key returned by POST /devices/register
#define DEVICE_API_KEY  "PASTE_YOUR_64_CHAR_HEX_KEY_HERE"

// Optional: which class/module this room is currently running.
// Leave empty ("") to send no class_id; the admin can link sessions server-side.
#define CLASS_ID        "CS301"

// Firmware version reported to server
#define FIRMWARE_VER    "1.0.0"

// Capture interval in milliseconds when using timer mode (no PIR)
#define CAPTURE_INTERVAL_MS  8000

// PIR motion sensor pin (set to -1 to use timer-only mode)
#define PIR_PIN         -1

// ── Hardware Pin Map — AI-Thinker ESP32-CAM ───────────────────────────────────
#define CAM_PIN_PWDN    32
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK     0
#define CAM_PIN_SIOD    26
#define CAM_PIN_SIOC    27
#define CAM_PIN_D7      35
#define CAM_PIN_D6      34
#define CAM_PIN_D5      39
#define CAM_PIN_D4      36
#define CAM_PIN_D3      21
#define CAM_PIN_D2      19
#define CAM_PIN_D1      18
#define CAM_PIN_D0       5
#define CAM_PIN_VSYNC   25
#define CAM_PIN_HREF    23
#define CAM_PIN_PCLK    22

// LEDs (use GPIO pins not shared with camera)
#define LED_GREEN       12
#define LED_RED         13

// OLED (I2C on GPIO 14=SDA, 15=SCL — adjust if needed)
#define OLED_SDA        14
#define OLED_SCL        15
#define OLED_WIDTH      128
#define OLED_HEIGHT      64
#define OLED_ADDR      0x3C

// ── Globals ───────────────────────────────────────────────────────────────────
Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire, -1);
bool oled_ok = false;

// ── Forward declarations ──────────────────────────────────────────────────────
bool init_camera();
bool wifi_connect();
void sync_ntp();
bool capture_and_send();
bool post_image(camera_fb_t* fb, const char* iso_timestamp);
void oled_show(const char* line1, const char* line2 = "", const char* line3 = "");
void led_green(int ms);
void led_red(int ms);
void led_yellow(int ms);
String get_iso_timestamp();

// ── setup() ──────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial.println("\n[boot] AI Attendance Node starting...");

  // GPIO init
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_RED,   OUTPUT);
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_RED,   LOW);

  if (PIR_PIN > 0) {
    pinMode(PIR_PIN, INPUT);
  }

  // OLED init
  Wire.begin(OLED_SDA, OLED_SCL);
  if (display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    oled_ok = true;
    display.setTextColor(SSD1306_WHITE);
    display.clearDisplay();
    display.display();
    oled_show("Smart Attendance", "Booting...");
  } else {
    Serial.println("[warn] OLED not found — continuing without display");
  }

  // Camera init
  if (!init_camera()) {
    oled_show("CAMERA ERROR", "Check wiring");
    led_red(0);  // solid red
    Serial.println("[fatal] camera init failed — halting");
    while (true) delay(1000);
  }
  Serial.println("[boot] camera OK");

  // WiFi
  oled_show("Smart Attendance", "Connecting WiFi...");
  if (!wifi_connect()) {
    oled_show("WiFi FAILED", "Check SSID/pass");
    led_red(5000);
  }

  // NTP time sync
  oled_show("Smart Attendance", "Syncing time...");
  sync_ntp();

  oled_show("Smart Attendance", "Ready", "Look at camera");
  Serial.println("[boot] ready");
}

// ── loop() ───────────────────────────────────────────────────────────────────
void loop() {
  // Reconnect WiFi if dropped
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[wifi] reconnecting...");
    oled_show("Reconnecting...", "");
    wifi_connect();
  }

  bool should_capture = false;

  if (PIR_PIN > 0) {
    // PIR mode: capture on motion
    if (digitalRead(PIR_PIN) == HIGH) {
      should_capture = true;
      delay(200);  // debounce
    }
  } else {
    // Timer mode: capture every CAPTURE_INTERVAL_MS
    should_capture = true;
  }

  if (should_capture) {
    oled_show("Smart Attendance", "Scanning...", "");
    led_yellow(0);  // yellow = processing

    bool ok = capture_and_send();

    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_RED,   LOW);

    if (ok) {
      led_green(2000);
      oled_show("Smart Attendance", "Image queued", "Attendance saved");
    } else {
      led_red(2000);
      oled_show("Smart Attendance", "Send failed", "Will retry");
    }

    oled_show("Smart Attendance", "Ready", "Look at camera");
  }

  if (PIR_PIN <= 0) {
    delay(CAPTURE_INTERVAL_MS);
  } else {
    delay(100);
  }
}

// ── Camera initialisation ─────────────────────────────────────────────────────
bool init_camera() {
  camera_config_t cfg;
  cfg.ledc_channel  = LEDC_CHANNEL_0;
  cfg.ledc_timer    = LEDC_TIMER_0;
  cfg.pin_d0        = CAM_PIN_D0;
  cfg.pin_d1        = CAM_PIN_D1;
  cfg.pin_d2        = CAM_PIN_D2;
  cfg.pin_d3        = CAM_PIN_D3;
  cfg.pin_d4        = CAM_PIN_D4;
  cfg.pin_d5        = CAM_PIN_D5;
  cfg.pin_d6        = CAM_PIN_D6;
  cfg.pin_d7        = CAM_PIN_D7;
  cfg.pin_xclk      = CAM_PIN_XCLK;
  cfg.pin_pclk      = CAM_PIN_PCLK;
  cfg.pin_vsync     = CAM_PIN_VSYNC;
  cfg.pin_href      = CAM_PIN_HREF;
  cfg.pin_sscb_sda  = CAM_PIN_SIOD;
  cfg.pin_sscb_scl  = CAM_PIN_SIOC;
  cfg.pin_pwdn      = CAM_PIN_PWDN;
  cfg.pin_reset     = CAM_PIN_RESET;
  cfg.xclk_freq_hz  = 20000000;
  cfg.pixel_format  = PIXFORMAT_JPEG;

  // Use PSRAM for larger frame if available
  if (psramFound()) {
    cfg.frame_size    = FRAMESIZE_VGA;   // 640x480 — good balance for face detection
    cfg.jpeg_quality  = 10;              // 0=best, 63=worst
    cfg.fb_count      = 2;
    Serial.println("[cam] PSRAM found — using VGA + 2 frame buffers");
  } else {
    cfg.frame_size    = FRAMESIZE_QVGA;  // 320x240 fallback
    cfg.jpeg_quality  = 12;
    cfg.fb_count      = 1;
    Serial.println("[cam] no PSRAM — using QVGA");
  }

  esp_err_t err = esp_camera_init(&cfg);
  if (err != ESP_OK) {
    Serial.printf("[cam] init failed: 0x%x\n", err);
    return false;
  }

  // Improve image quality settings
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 1);      // -2 to 2
    s->set_contrast(s, 1);        // -2 to 2
    s->set_saturation(s, 0);      // -2 to 2
    s->set_sharpness(s, 1);       // -2 to 2
    s->set_whitebal(s, 1);        // auto white balance on
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);   // auto exposure on
    s->set_aec2(s, 1);
    s->set_gain_ctrl(s, 1);       // auto gain on
    s->set_gainceiling(s, (gainceiling_t)2);
    s->set_lenc(s, 1);            // lens correction
    s->set_hmirror(s, 0);         // flip if camera is mounted upside-down
    s->set_vflip(s, 0);
  }
  return true;
}

// ── WiFi ─────────────────────────────────────────────────────────────────────
bool wifi_connect() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[wifi] connected — IP: %s\n", WiFi.localIP().toString().c_str());
    return true;
  }
  Serial.println("\n[wifi] failed to connect");
  return false;
}

// ── NTP time sync ─────────────────────────────────────────────────────────────
void sync_ntp() {
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");
  Serial.print("[ntp] syncing");
  int attempts = 0;
  time_t now = 0;
  while (now < 1000000000L && attempts < 20) {
    time(&now);
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (now > 1000000000L) {
    Serial.printf("\n[ntp] synced — %s", ctime(&now));
  } else {
    Serial.println("\n[ntp] sync failed — using device uptime as timestamp");
  }
}

// ── Capture & send ───────────────────────────────────────────────────────────
bool capture_and_send() {
  // Warm up: discard first frame (camera AEC needs a moment)
  camera_fb_t* fb = esp_camera_fb_get();
  if (fb) esp_camera_fb_return(fb);
  delay(100);

  // Actual capture
  fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[capture] failed to get frame");
    return false;
  }
  Serial.printf("[capture] JPEG %u bytes\n", fb->len);

  String ts = get_iso_timestamp();
  bool result = post_image(fb, ts.c_str());
  esp_camera_fb_return(fb);
  return result;
}

// ── HTTP POST multipart/form-data ─────────────────────────────────────────────
bool post_image(camera_fb_t* fb, const char* iso_timestamp) {
  const int MAX_RETRIES = 3;

  for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("[post] WiFi down — skipping retry");
      return false;
    }

    HTTPClient http;
    String url = String("http://") + SERVER_HOST + ":" + SERVER_PORT + INGEST_PATH;
    http.begin(url);
    http.setTimeout(10000);  // 10s timeout

    // ── Build multipart/form-data body manually ──────────────────────────────
    // We can't use the standard form helper for binary data on ESP32,
    // so we build the raw boundary manually.
    const String boundary = "----ESP32Boundary7MA4YWxk";
    String contentType = "multipart/form-data; boundary=" + boundary;

    // Text fields
    String body_prefix = "";
    // class_id field
    if (strlen(CLASS_ID) > 0) {
      body_prefix += "--" + boundary + "\r\n";
      body_prefix += "Content-Disposition: form-data; name=\"class_id\"\r\n\r\n";
      body_prefix += String(CLASS_ID) + "\r\n";
    }
    // captured_at field
    body_prefix += "--" + boundary + "\r\n";
    body_prefix += "Content-Disposition: form-data; name=\"captured_at\"\r\n\r\n";
    body_prefix += String(iso_timestamp) + "\r\n";
    // firmware_version field
    body_prefix += "--" + boundary + "\r\n";
    body_prefix += "Content-Disposition: form-data; name=\"firmware_version\"\r\n\r\n";
    body_prefix += String(FIRMWARE_VER) + "\r\n";
    // Image field header
    body_prefix += "--" + boundary + "\r\n";
    body_prefix += "Content-Disposition: form-data; name=\"image\"; filename=\"capture.jpg\"\r\n";
    body_prefix += "Content-Type: image/jpeg\r\n\r\n";

    String body_suffix = "\r\n--" + boundary + "--\r\n";

    int total_len = body_prefix.length() + fb->len + body_suffix.length();

    http.addHeader("Content-Type", contentType);
    http.addHeader("Content-Length", String(total_len));
    http.addHeader("X-Device-Key", DEVICE_API_KEY);
    http.addHeader("X-Device-MAC", WiFi.macAddress());

    // Stream the body in chunks
    WiFiClient* stream = http.getStreamPtr();
    if (!stream) {
      Serial.println("[post] failed to get stream");
      http.end();
      continue;
    }

    // Use sendRequest with streaming
    uint8_t* payload = (uint8_t*)malloc(total_len);
    if (!payload) {
      Serial.println("[post] malloc failed — image too large?");
      http.end();
      return false;
    }

    int offset = 0;
    memcpy(payload + offset, body_prefix.c_str(), body_prefix.length());
    offset += body_prefix.length();
    memcpy(payload + offset, fb->buf, fb->len);
    offset += fb->len;
    memcpy(payload + offset, body_suffix.c_str(), body_suffix.length());

    int http_code = http.POST(payload, total_len);
    free(payload);

    if (http_code == 202) {
      String response = http.getString();
      Serial.printf("[post] OK (attempt %d) — %s\n", attempt, response.c_str());
      http.end();

      // Parse job_id from response for logging
      StaticJsonDocument<256> doc;
      if (deserializeJson(doc, response) == DeserializationError::Ok) {
        const char* job_id = doc["job_id"];
        Serial.printf("[post] job_id: %s\n", job_id ? job_id : "unknown");
      }
      return true;
    }

    Serial.printf("[post] attempt %d failed — HTTP %d\n", attempt, http_code);
    if (http_code > 0) {
      Serial.println("[post] response: " + http.getString());
    }
    http.end();

    if (attempt < MAX_RETRIES) {
      delay(1000 * attempt);  // backoff: 1s, 2s
    }
  }

  Serial.println("[post] all retries exhausted");
  return false;
}

// ── OLED helpers ──────────────────────────────────────────────────────────────
void oled_show(const char* line1, const char* line2, const char* line3) {
  if (!oled_ok) return;
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println(line1);
  display.drawFastHLine(0, 10, OLED_WIDTH, SSD1306_WHITE);
  display.setCursor(0, 14);
  display.println(line2);
  display.setCursor(0, 26);
  display.println(line3);
  // Bottom status bar
  display.drawFastHLine(0, 52, OLED_WIDTH, SSD1306_WHITE);
  display.setCursor(0, 55);
  String ip_str = (WiFi.status() == WL_CONNECTED)
    ? WiFi.localIP().toString()
    : "No WiFi";
  display.print(ip_str);
  display.display();
}

// ── LED helpers ───────────────────────────────────────────────────────────────
void led_green(int ms) {
  digitalWrite(LED_GREEN, HIGH);
  digitalWrite(LED_RED,   LOW);
  if (ms > 0) { delay(ms); digitalWrite(LED_GREEN, LOW); }
}

void led_red(int ms) {
  digitalWrite(LED_RED,   HIGH);
  digitalWrite(LED_GREEN, LOW);
  if (ms > 0) { delay(ms); digitalWrite(LED_RED, LOW); }
}

void led_yellow(int ms) {
  // Simulate yellow by toggling both at 50Hz
  if (ms == 0) {
    digitalWrite(LED_GREEN, HIGH);
    digitalWrite(LED_RED,   HIGH);
    return;
  }
  unsigned long end = millis() + ms;
  while (millis() < end) {
    digitalWrite(LED_GREEN, HIGH);
    digitalWrite(LED_RED,   HIGH);
    delay(20);
  }
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_RED,   LOW);
}

// ── Timestamp ─────────────────────────────────────────────────────────────────
String get_iso_timestamp() {
  time_t now;
  time(&now);
  if (now < 1000000000L) {
    // NTP not synced — use millis as fallback (server will use receipt time)
    return "1970-01-01T00:00:00Z";
  }
  char buf[25];
  strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
  return String(buf);
}

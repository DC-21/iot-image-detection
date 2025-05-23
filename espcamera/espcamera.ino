#include <WebServer.h>
#include <WiFi.h>
#include <esp32cam.h>

//THIS PROGRAM SENDS IMAGE IF IT IS PLACED IN WEB IP, BUT IF IT IS PLACED IN PYTHON IT SENDS VIDEO THROUGH THE ITERATIONS. . . (IF IT WORKS IN PYTHON)
const char* WIFI_SSID = "Redmi 9Ax";
const char* WIFI_PASS = "12345678";

WebServer server(80); //server on port 80

static auto loRes = esp32cam::Resolution::find(320, 240); //low resolution
static auto hiRes = esp32cam::Resolution::find(800, 600); //high resolution
//static auto hiRes = esp32cam::Resolution::find(640, 480); //high resolution (for fps rates) (IP CAM APP)

void
serveJpg() //capture image .jpg
{
  auto frame = esp32cam::capture();
  if (frame == nullptr) {
    Serial.println("Capture Fail");
    server.send(503, "", "");
    return;
  }
  Serial.printf("CAPTURE OK %dx%d %db\n", frame->getWidth(), frame->getHeight(),
                static_cast<int>(frame->size()));

  server.setContentLength(frame->size());
  server.send(200, "image/jpeg");
  WiFiClient client = server.client();
  frame->writeTo(client);  //and send to a client (in this case it will be python)
}

void
handleJpgLo()  //allows to send low resolution image
{
  if (!esp32cam::Camera.changeResolution(loRes)) {
    Serial.println("SET-LO-RES FAIL");
  }
  serveJpg();
}

void
handleJpgHi() //allows to send high resolution image
{
  if (!esp32cam::Camera.changeResolution(hiRes)) {
    Serial.println("SET-HI-RES FAIL");
  }
  serveJpg();
}

void setup()
{
  Serial.begin(115200);
  Serial.println();
  Serial.println("ESP32 Camera Starting...");

  {
    using namespace esp32cam;
    Config cfg;
    cfg.setPins(pins::AiThinker);
    cfg.setResolution(hiRes);
    cfg.setBufferCount(2);
    cfg.setJpeg(80);

    bool ok = Camera.begin(cfg);
    Serial.println(ok ? "CAMERA OK" : "CAMERA FAIL");
  }

  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  Serial.printf("Connecting to WiFi: %s\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS); //connect to the WiFi network
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    attempts++;
    if (attempts % 10 == 0) {
      Serial.printf("\nStill trying to connect to %s (attempt %d)\n", WIFI_SSID, attempts);
    }
    if (attempts > 60) { // 30 seconds timeout
      Serial.println("\nWiFi connection failed! Restarting...");
      ESP.restart();
    }
  }

  Serial.println("\nWiFi connected!");
  Serial.printf("Camera IP address: %s\n", WiFi.localIP().toString().c_str());

  Serial.print("Low resolution URL: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/cam-lo.jpg");

  Serial.print("High resolution URL: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/cam-hi.jpg");

  server.on("/cam-lo.jpg",handleJpgLo);//send to the server
  server.on("/cam-hi.jpg", handleJpgHi);

  server.begin();
  Serial.println("HTTP server started");
}

void loop()
{
  server.handleClient();
}
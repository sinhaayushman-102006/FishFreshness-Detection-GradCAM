from flask import Flask, render_template, Response, request
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
import cv2
from PIL import Image
import os
from gradcam_utils import generate_gradcam

app = Flask(__name__)

# ================= MODEL =================
model = efficientnet_v2_s()
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

classes = ['Fresh', 'Medium', 'Spoiled']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ================= CAMERA =================
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transform(Image.fromarray(img)).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)

        label = classes[pred.item()]
        confidence = conf.item() * 100

        cv2.putText(frame, f"{label} ({confidence:.2f}%)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ================= ROUTES =================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None
    heatmap_path = None

    if request.method == "POST":
        file = request.files["file"]
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        img = Image.open(filepath).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

        result = classes[pred.item()]
        confidence = round(conf.item() * 100, 2)

        # Grad-CAM
        heatmap_path = generate_gradcam(model, img, filepath)

        image_path = filepath

    return render_template("index.html",
                           result=result,
                           confidence=confidence,
                           image=image_path,
                           heatmap=heatmap_path)

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
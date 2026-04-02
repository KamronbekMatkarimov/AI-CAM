from ultralytics import YOLO
import os

model_path = "yolov8n.pt" 
images_dir = "test_images"

model = YOLO(model_path)

total_people = 0

for filename in os.listdir(images_dir):
    path = os.path.join(images_dir, filename)
    if not path.lower().endswith((".jpg", ".jpeg", ".png")):
        print(f"Пропущен: {filename} — не изображение")
        continue

    results = model(path)[0]

    person_count = sum(1 for cls in results.boxes.cls if int(cls) == 0)

    print(f"{filename}: {person_count} people")
    total_people += person_count

print(f"\nИтого людей на всех изображениях: {total_people}")
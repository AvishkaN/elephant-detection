# Necessary imports
import cv2
import os
import numpy as np
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import torch  # Import torch for checking GPU availability

def runElephantDetection():
    # MQTT connection functions
    def on_connect(client, userdata, flags, reason_code, properties):
        print(f"Connected with result code {reason_code}")
        client.subscribe("topic/topic1")

    def on_message(client, userdata, msg):
        print(str(msg.payload))

    def send_message(client, topic, message):
        result = client.publish(topic, message)
        status = result[0]
        if status == 0:
            print(f"Message '{message}' sent to topic '{topic}'")
        else:
            print(f"Failed to send message to topic '{topic}'")

    mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqttc.on_connect = on_connect
    mqttc.on_message = on_message
    mqttc.connect("broker.hivemq.com", 1883, 60)

    # Video input/output settings
    video_path = 'input1.mp4'
    video_path_out = '{}_out.mp4'.format(video_path)

    # Open video and get frame size
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W, _ = np.shape(frame)
    frame = cv2.resize(frame, (640, 480))

    # Detection-related variables
    recodedElephants = 0
    elephantsRecordedThreshold = 30
    elephantDetected = False
    elephantExitCount = 0

    try:
        # Output video settings
        out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

        # Check if GPU is available and set the device accordingly
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("GPU detected")  # Print statement when GPU is detected
        else:
            device = torch.device('cpu')
            print("GPU not detected, using CPU")  # Print if using CPU

        # Load YOLO model and ensure it runs on the appropriate device
        model_path = "best3.pt"
        model = YOLO(model_path).to(device)  # Set the model to use GPU or CPU based on availability

        threshold = 0.5  # Confidence threshold
        allowed_class_id = 20  # Assuming elephant is class ID 20 in your custom model

        while ret:
            frame = cv2.resize(frame, (640, 480))  # Consider reducing this size if performance is still low

            # Start time for FPS calculation
            current_time = cv2.getTickCount()

            # Run YOLO prediction on the set device (GPU or CPU), limiting to specific class
            results = model.predict(source=frame, device=device, classes=[allowed_class_id])[0]

            print('------results', results.boxes.data.tolist())

            # Process each detection
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result

                # Check if detection is an elephant and confidence is above the threshold
                if (score > threshold) and (class_id == allowed_class_id):
                    print(f"Elephant detected with class ID: {class_id}")

                    # Increment elephant detection counter
                    if recodedElephants == elephantsRecordedThreshold:
                        print('true message⏭')
                        elephantDetected = True
                        recodedElephants = 1  # Reset count after detection
                        send_message(mqttc, "sliitelp/detect", "true")
                    else:
                        recodedElephants += 1

                    # Draw bounding box and label
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    cv2.putText(frame, "Elephant", (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            print('-----logic', len(results.boxes.data.tolist()) == 0)
            if len(results.boxes.data.tolist()) == 0:
                # Check for elephant exit scenario
                if elephantDetected and elephantExitCount == 15:
                    print('false message 👉')
                    elephantExitCount = 0
                    elephantDetected = False
                    send_message(mqttc, "sliitelp/detect", "false")
                elephantExitCount += 1

            # Calculate FPS and display it
            time_diff = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
            fps = 1 / time_diff
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Show the frame with bounding boxes
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            # Write frame to output video
            out.write(frame)

            # Read the next frame
            ret, frame = cap.read()

        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

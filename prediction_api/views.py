from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import os


current_path = os.getcwd()
darknet_weights_file = os.path.join(current_path, 'yolov3-tiny.weights')
darknet_cfg_file = os.path.join(current_path, 'yolov3-tiny.cfg')
darknet_classes_file = os.path.join(current_path, 'cocolabels.txt')

net = cv2.dnn.readNet(darknet_weights_file,
                      darknet_cfg_file)

with open(darknet_classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def predict(received_img):
    image = cv2.imdecode(np.fromstring(received_img.read(), np.uint8), cv2.IMREAD_UNCHANGED)  # imread(received_img)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)
    # image_np = load_image_into_numpy_array(image)

    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    json_list = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        data = {}
        data['class'] = classes[class_ids[i]]
        data['confidence'] = str(round(confidences[i], 4))
        data['top'] = str(x)
        data['left'] = str(y)
        data['bottom'] = str(x + w)
        data['right'] = str(y + h)

        json_list.append(data)
    return json_list


@csrf_exempt
def predict_api(request):  # this one works with form-data uploads
    if request.method == 'POST' and request.FILES['data']:
        received_img = request.FILES['data']
        json_list = predict(received_img)

        return JsonResponse(json_list, safe=False)
    return JsonResponse({'error': 'no image received'}, safe=False)


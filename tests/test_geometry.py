from qrcode_variants.geometry import clamp_xyxy, xyxy_to_yolo, yolo_to_xyxy


def test_xyxy_to_yolo_center_box():
  line = xyxy_to_yolo(class_id=1, xyxy=(25, 25, 75, 75), image_size=(100, 100))
  assert line == (1, 0.5, 0.5, 0.5, 0.5)


def test_yolo_roundtrip():
  original = (10, 20, 90, 80)
  yolo = xyxy_to_yolo(2, original, (200, 100))
  restored = yolo_to_xyxy(yolo[1:], (200, 100))
  assert restored == original


def test_clamp_xyxy_keeps_box_inside_image():
  clamped = clamp_xyxy((-10, -5, 120, 110), (100, 80))
  assert clamped == (0, 0, 100, 80)

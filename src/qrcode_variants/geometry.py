def clamp_xyxy(xyxy, image_size):
  x1, y1, x2, y2 = xyxy
  width, height = image_size
  x1 = min(max(int(round(x1)), 0), width)
  y1 = min(max(int(round(y1)), 0), height)
  x2 = min(max(int(round(x2)), 0), width)
  y2 = min(max(int(round(y2)), 0), height)
  if x2 < x1:
    x1, x2 = x2, x1
  if y2 < y1:
    y1, y2 = y2, y1
  return (x1, y1, x2, y2)


def xyxy_to_yolo(class_id, xyxy, image_size):
  x1, y1, x2, y2 = clamp_xyxy(xyxy, image_size)
  width, height = image_size
  box_w = x2 - x1
  box_h = y2 - y1
  xc = (x1 + x2) / 2.0 / width
  yc = (y1 + y2) / 2.0 / height
  return (
    int(class_id),
    xc,
    yc,
    box_w / width,
    box_h / height,
  )


def yolo_to_xyxy(yolo_box, image_size):
  xc, yc, bw, bh = yolo_box
  width, height = image_size
  box_w = bw * width
  box_h = bh * height
  cx = xc * width
  cy = yc * height
  x1 = int(round(cx - box_w / 2.0))
  y1 = int(round(cy - box_h / 2.0))
  x2 = int(round(cx + box_w / 2.0))
  y2 = int(round(cy + box_h / 2.0))
  return clamp_xyxy((x1, y1, x2, y2), image_size)


def format_yolo_line(class_id, xyxy, image_size):
  cid, xc, yc, bw, bh = xyxy_to_yolo(class_id, xyxy, image_size)
  return f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

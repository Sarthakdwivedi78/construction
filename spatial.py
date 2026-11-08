"""Image-plane associations. These coordinates do not measure site distance."""

def relationships(objects):
    pairs = []
    people = [(i, o["xyxy"]) for i, o in enumerate(objects) if o["label"] == "Person"]
    for i, item in enumerate(objects):
        if item["label"] not in {"NO-Hardhat", "Hardhat", "NO-Safety Vest", "Safety Vest", "NO-Mask", "Mask"}:
            continue
        x1, y1, x2, y2 = item["xyxy"]
        x, y = (x1+x2)/2, (y1+y2)/2
        possible = [j for j, (left, top, right, bottom) in people
                    if left <= x <= right and top <= y <= bottom]
        if len(possible) == 1:
            pairs.append({"ppe_detection_index": i, "person_detection_index": possible[0],
                          "relation": "PPE box center inside one person box"})
    for i, item in enumerate(objects):
        if item["label"] not in {"machinery", "vehicle"}:
            continue
        x1, y1, x2, y2 = item["xyxy"]
        for j, (left, top, right, bottom) in people:
            if min(x2, right) > max(x1, left) and min(y2, bottom) > max(y1, top):
                pairs.append({"equipment_detection_index": i, "person_detection_index": j,
                              "relation": "Boxes overlap in image plane; physical distance unknown"})
    return pairs

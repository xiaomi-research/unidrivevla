class PlantVariables:
    bev_colors = [
        [0.485, 0.456, 0.406],  # Background: ImageNet mean
        [0.25, 0.25, 0.75],  # Street: Blue
        [0.485, 0.456, 0.406],  # Sidewalk: ImageNet mean
        [0.75, 0.25, 0.25],  # All lines: Red
        [0.25, 0.75, 0.25],  # Broken lines: Green
    ]

    speed_cats = {50: 0, 80: 1, 100: 2, 120: 3}

    class_nums = {
        "car": 1.0,
        "walker": 2.0,
        "static": 3.0,
        "static_car": 1.0,
        "stop_sign": 4.0,
        "traffic_light": 5.0,
        "emergency": 6.0,
    }

    car_types = ["car", "walker", "emergency"]

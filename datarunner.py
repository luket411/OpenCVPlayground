location = 'assets'

photos = [
    f"{location}/IMG_1036.jpg",
    f"{location}/IMG_1525.jpg",
    f"{location}/IMG_1693.jpg",
    f"{location}/IMG_1939.jpg",
    f"{location}/IMG_2468.jpg",
    f"{location}/IMG_4281.jpg",
    f"{location}/IMG_4559.jpg",
    f"{location}/IMG_4582.jpg",
    f"{location}/IMG_4643.jpg",
    f"{location}/IMG_5856.jpg",
    f"{location}/IMG_7248.jpg",
    f"{location}/IMG_7527.jpg",
    f"{location}/IMG_7528.jpg",
    f"{location}/IMG_7710.jpg",
    f"{location}/IMG_7828.jpg",
    f"{location}/IMG_8110.jpg",
    f"{location}/IMG_9277.jpg",
    f"{location}/IMG_9359.jpg",
    f"{location}/IMG_9674.jpg"
]


def run_all(func):
    def wrapper(*args, **kwargs):
        for photo in photos:
            func(photo, *args, **kwargs)
    return wrapper
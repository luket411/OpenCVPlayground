from csv import reader

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
        
        if (len(args) != 0) or (len(kwargs) != 0):
            return func(*args, **kwargs) 
        
        for photo in photos:
            func(photo, *args, **kwargs)
    return wrapper

class cities_loader():
    def __init__(self):
        with open(f'assets/cities.csv') as cities_file:
            for row in reader(cities_file):
                if len(row) != 3:
                    raise IndexError(f"Input row for row:'{row[0]}' len(row) != 3")
                city = row[0]
                location = (float(row[1]), float(row[2]))
                self.__dict__[city] = location
    
     
    def get_cities_list(self):
        lst = list(self.__dict__.keys())
        return lst
    
    def __getitem__(self, city):
        return self.__dict__[city]
    
    def __iter__(self):
        return iter(self.__dict__.items())
                
if __name__ == "__main__":
    c = cities_loader()
    print(c.danzig)
    print(len(c.get_cities_list()))
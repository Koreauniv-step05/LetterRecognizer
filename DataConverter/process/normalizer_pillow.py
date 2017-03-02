def normalize_images(images):
    res = []
    for image in images:
        res.append(normalize_image(image))
    return res

def normalize_image(image):
    image = gray(image)
    image = resize(image,[28,28]) # todo 28,48 -> flexible

    return image

def gray(image):
    return image.convert('L')

def resize(image,siz):
    return image.resize(siz)


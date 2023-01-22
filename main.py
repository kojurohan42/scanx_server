from flask import Flask, request, jsonify
import werkzeug
import numpy as np
import cv2
import tensorflow as tf
app = Flask(__name__)


def borders(here_img, thresh, bthresh=0.092):
    shape = here_img.shape
    #check = int(115 * size[0] / 600)
    #check = int(55 * size[0] / 600)
    check = int(bthresh*shape[0])
    image = here_img[:]
    top, bottom = 0, shape[0] - 1
    # plt.imshow(image)
    # plt.show()

    # find the background color for empty column
    bg = np.repeat(thresh, shape[1])
    count = 0
    for row in range(1, shape[0]):
        if (np.equal(bg, image[row]).any()) == True:
            # print(count)
            count += 1
        else:
            count = 0
        if count >= check:
            top = row - check
            break

    bg = np.repeat(thresh, shape[1])
    count = 0
    rows = np.arange(1, shape[0])
    # print(rows)
    for row in rows[::-1]:
        if (np.equal(bg, image[row]).any()) == True:
            count += 1
        else:
            count = 0
        if count >= check:
            bottom = row + count
            break

    d1 = (top - 2) >= 0
    d2 = (bottom + 2) < shape[0]
    d = d1 and d2
    if(d):
        b = 2
    else:
        b = 0
#     if top>30:
#         top-=30
#         bottom+=30
#     if top>20:
#         top=top-20
#         bottom+=20
    if top > 10:
        top = top-10
        bottom += 10
    elif top > 5:
        top -= 5
        bottom += 5

    return (top, bottom, b)


def preprocess(bgr_img):  # gray image
    blur = cv2.GaussianBlur(bgr_img, (5, 5), 0)
    # converts black to white and inverse
    ret, th_img = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    rows, cols = th_img.shape
    bg_test = np.array([th_img[i][i] for i in range(5)])
    if bg_test.all() == 0:
        text_color = 255
    else:
        text_color = 0
#     plt.imshow(th_img)
    tb = borders(th_img, text_color)
    lr = borders(th_img.T, text_color)
    dummy = int(np.average((tb[2], lr[2]))) + 2
    template = th_img[tb[0]+dummy:tb[1]-dummy, lr[0]+dummy:lr[1]-dummy]

#     plt.imshow(template)
#     plt.show()
    return (template, tb, lr)


@app.route('/upload', methods=["POST"])
def upload():
    new_model = tf.keras.models.load_model('my_model.h5')
    characters = 'क,ख,ग,घ,ङ,च,छ,ज,झ,ञ,ट,ठ,ड,ढ,ण,त,थ,द,ध,न,प,फ,ब,भ,म,य,र,ल,व,श,ष,स,ह,क्ष,त्र,ज्ञ,का,कि,की,कु,कू,के,कै,को,कौ,कं,कः,०,१,२,३,४,५,६,७,८,९'
    characters = characters.split(',')
    if(request.method == "POST"):
        imagefile = request.files['image']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./uploadedimages/" + filename)
        img = cv2.imread("./uploadedimages/" + filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prepimg, tb, lr = preprocess(gray)
        (thresh, im_bw) = cv2.threshold(prepimg, 128,
                                        255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        resized_img = cv2.resize(im_bw, (32, 32))

        # image_file = new_image.convert('1')
        matrix = np.array(resized_img).astype(int)
        reshaped_matrix = matrix.reshape(1, 32, 32, 1)
        prediction = np.argmax(new_model.predict(reshaped_matrix))

        return jsonify({
            "message": characters[prediction]
        })


if __name__ == "__main__":
    app.run(debug=True, port=4000)

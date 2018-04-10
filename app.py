import boto3
import numpy as np
from PIL import Image
from lib import models, hash
from lshash.lshash import LSHash
from flask import Flask, request, render_template

s3 = boto3.client('s3')
bucket_name = 'vizlab-images'

MAX_RESULTS = 10
MAX_DISTANCE = 80
HASH_SIZE = 16 # must be power of 2

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

session = models.Session()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def build_index():
    lsh = LSHash(32, HASH_SIZE**2)
    q = session.query(models.Image.hash, models.Image.id)
    for h, id in q.all():
        h = hash.hex_to_hash(h).hash.flatten()
        bytearr = h.view(np.uint8)
        lsh.index(bytearr, extra_data=id)
    return lsh


def search(img):
    h = hash.dhash(img, hash_size=HASH_SIZE)
    query = h.hash.flatten().view(np.uint8)
    results = lsh.query(query, num_results=MAX_RESULTS, distance_func='hamming')
    ids_dists = [(r[0][1], r[1]) for r in results]
    return ids_dists


app = Flask(__name__)
print('Building index...')
lsh = build_index()
print('Done building index')


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        f = request.files['image']
        if f and f.filename and allowed_file(f.filename):
            img = Image.open(f.stream)
            results = search(img)
            ids_to_dists = {id: dist for id, dist in results}
            ids = list(ids_to_dists.keys())
            images = session.query(models.Image).filter(models.Image.id.in_(ids)).all()
            images = [
                (img, ids_to_dists[img.id], s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket_name, 'Key': img.key}, ExpiresIn=1800))
                for img in images]
            images = [im for im in images if im[1] <= MAX_DISTANCE]
            images = sorted(images, key=lambda i: i[1])
            return render_template('results.html', images=images)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
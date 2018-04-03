import boto3
import faiss
import numpy as np
from PIL import Image
from lib import models, hash
from flask import Flask, request, render_template

s3 = boto3.client('s3')
bucket_name = 'vizlab-images'

HASH_SIZE = 16 # must be power of 2
D = HASH_SIZE**2

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

session = models.Session()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def build_index():
    q = session.query(models.Image.hash, models.Image.id)
    hashes, ids = zip(*q.all())
    hashes = [hash.normalize(hash.hex_to_hash(h).hash).flatten() for h in hashes]
    hashes = np.stack(hashes)
    hashes = hashes.astype('float32')
    ids = np.array(ids)

    # <https://github.com/facebookresearch/faiss/wiki/Faiss-indexes>
    # idx_ = faiss.IndexLSH(d, D*2) # this gets better more bits
    # idx_ = faiss.IndexFlatL2(d) # best, but brute-force, and stores full vectors
    # idx_ = faiss.IndexHNSWFlat(D, 16)
    # see <https://github.com/facebookresearch/faiss/wiki/Getting-started-tutorial>
    idx_ = faiss.IndexFlatIP(D) # might be best option?
    idx = faiss.IndexIDMap(idx_) # so we can specify our own indices
    idx.add_with_ids(hashes, ids)

    # need to hold onto idx_ reference or will get a segfault
    return idx, idx_


app = Flask(__name__)
idx, idx_ = build_index()
print('Done building index')


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        f = request.files['image']
        if f and f.filename and allowed_file(f.filename):
            img = Image.open(f.stream)
            h = hash.dhash(img, hash_size=HASH_SIZE)
            query = hash.normalize(h.hash.flatten()).astype('float32')
            dists, ids = idx.search(np.array([query]), 10)
            dists, ids = dists[0].tolist(), ids[0].tolist()
            images = session.query(models.Image).filter(models.Image.id.in_(ids)).all()
            images = [
                (img, dist, s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket_name, 'Key': img.key}, ExpiresIn=1800))
                for img, dist in zip(images, dists)]
            return render_template('results.html', images=images)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
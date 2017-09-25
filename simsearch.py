"""
open questions:

- looks like this has to be maintained in memory.
- how to update if its in memory?
    - concurrent searches are supported, but not search/add or add/add
    - how to deal with this? might maintain two copies, one which updates, one
    which is live, they swap?

based on <https://github.com/facebookresearch/faiss/wiki/FAQ> looks like it has to be reset if new vectors are added:

>   No it is not. The states for the index are:
>
>       - is_trained = false, ntotal = 0, transition to 2 with train()
>       - is_trained = true, ntotal = 0, transition to 3 with add()
>       - is_trained = true, ntotal > 0, transition to 2 with reset()

also, it's optimized for batch queries, rather than searching one vector at a time

"""

import faiss
import imagehash
import numpy as np
from PIL import Image

HASH_SIZE = 16 # must be power of 2
D = HASH_SIZE**2

def create_index(fnames, ids):
    """create similarity index"""
    # <https://github.com/facebookresearch/faiss/wiki/Faiss-indexes>
    # idx_ = faiss.IndexLSH(d, 2048) # this gets better more bits
    # idx_ = faiss.IndexFlatL2(d) # best, but brute-force, and stores full vectors
    idx_ = faiss.IndexFlatIP(D) # might be best option?
    # see <https://github.com/facebookresearch/faiss/wiki/Getting-started-tutorial>

    idx = faiss.IndexIDMap(idx_) # so we can specify our own indices
    hashes = [imghash(fname) for fname in fnames]
    hashes = np.stack(hashes)

    idx.add_with_ids(hashes, np.array(ids))
    return idx


def imghash(fname):
    """generate perceptual hash for image.
    uses whash, but other options are available:
    <https://github.com/JohannesBuchner/imagehash>"""
    img = Image.open(fname)
    hash = imagehash.whash(img, hash_size=HASH_SIZE).hash.flatten()

    # faiss requires float32
    return hash.astype('float32')

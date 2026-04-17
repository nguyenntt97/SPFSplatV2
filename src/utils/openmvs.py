import struct
import numpy as np
from collections import namedtuple

# -------------------------------------------------------------------
# 1) Define bitmasks to match HeaderDepthDataRaw::HAS_*
HAS_DEPTH  = 1 << 0
HAS_NORMAL = 1 << 1
HAS_CONF   = 1 << 2
HAS_VIEWS  = 1 << 3

# 2) Mirror the C++ header layout exactly here:
#    e.g. if your C++ is:
#      struct HeaderDepthDataRaw {
#        char     name[8];
#        uint32_t type;
#        uint32_t depthWidth, depthHeight;
#        uint32_t imageWidth, imageHeight;
#        float    dMin, dMax;
#      };
#    then:
HEADER_FORMAT = '<HBBIIIIff'
HeaderMeta = namedtuple('Header',
                    ['name', 'type', "padding",
                     'depthWidth', 'depthHeight',
                     'imageWidth', 'imageHeight',
                     'dMin', 'dMax'])

def read_header(f):
    header_size = struct.calcsize(HEADER_FORMAT)
    data = f.read(header_size)
    if len(data) != header_size:
        raise EOFError("Cannot read full header")
    raw = struct.unpack(HEADER_FORMAT, data)
    # strip trailing NULs from name
    name = raw[0]
    return HeaderMeta(name, *raw[1:])

# -------------------------------------------------------------------
def import_depth_data_raw(path, flags):
    """
    Reads a .dmap-style binary file into numpy arrays.
    Returns a dict with keys:
      image_filename, IDs,
      image_size, K, R, C,
      dMin, dMax,
      depth_map, normal_map, confidence_map, views_map
    """
    with open(path, 'rb') as f:
        # --- header ---
        header = read_header(f)
        # you probably want to compare header.name to your expected magic:
        # if header.name != 'MyMAGIC': raise ValueError(...)
        if not (header.type & HAS_DEPTH):
            raise ValueError("File claims no depth data")
        if header.depthWidth <= 0 or header.depthHeight <= 0:
            raise ValueError("Invalid depth dims")
        if header.imageWidth < header.depthWidth or header.imageHeight < header.depthHeight:
            raise ValueError("Image smaller than depth")

        # --- image file name ---
        nfn = struct.unpack('<H', f.read(2))[0]
        image_filename = f.read(nfn).decode('latin-1')

        # --- neighbor IDs ---
        nIDs = struct.unpack('<I', f.read(4))[0]
        if not (0 < nIDs < 256):
            raise ValueError("Unreasonable number of neighbor IDs")
        IDs = np.frombuffer(f.read(4 * nIDs), dtype=np.uint32)

        # --- camera intrinsics/extrinsics ---
        #   assumes REAL==double in your build
        K = np.frombuffer(f.read(8 * 9), dtype=np.float64).reshape((3, 3))
        R = np.frombuffer(f.read(8 * 9), dtype=np.float64).reshape((3, 3))
        C = np.frombuffer(f.read(8 * 3), dtype=np.float64).reshape((3,))

        # --- depth range & image size ---
        dMin, dMax = header.dMin, header.dMax
        image_size = (header.imageWidth, header.imageHeight)

        # helper to skip bytes
        def skip(count):
            f.seek(count, 1)

        # --- depth map ---
        n_depth = header.depthWidth * header.depthHeight
        if flags & HAS_DEPTH:
            depth_buf = f.read(4 * n_depth)
            depth_map = np.frombuffer(depth_buf, dtype=np.float32)\
                          .reshape((header.depthHeight, header.depthWidth))
        else:
            skip(4 * n_depth)
            depth_map = None

        # --- normal map ---
        normal_map = None
        if header.type & HAS_NORMAL:
            if flags & HAS_NORMAL:
                normal_buf = f.read(4 * 3 * n_depth)
                normal_map = np.frombuffer(normal_buf, dtype=np.float32)\
                               .reshape((header.depthHeight,
                                         header.depthWidth, 3))
            else:
                skip(4 * 3 * n_depth)

        # --- confidence map ---
        conf_map = None
        if header.type & HAS_CONF:
            if flags & HAS_CONF:
                conf_buf = f.read(4 * n_depth)
                conf_map = np.frombuffer(conf_buf, dtype=np.float32)\
                               .reshape((header.depthHeight,
                                         header.depthWidth))
            else:
                skip(4 * n_depth)

        # --- views map ---
        views_map = None
        if header.type & HAS_VIEWS and (flags & HAS_VIEWS):
            # 4 bytes per pixel
            vbuf = f.read(4 * n_depth)
            views_map = np.frombuffer(vbuf, dtype=np.uint8)\
                             .reshape((header.depthHeight,
                                       header.depthWidth, 4))

    return {
        'image_filename': image_filename,
        'IDs':            IDs,
        'image_size':     image_size,
        'K':              K,
        'R':              R,
        'C':              C,
        'dMin':           dMin,
        'dMax':           dMax,
        'depth_map':      depth_map,
        'normal_map':     normal_map,
        'confidence_map': conf_map,
        'views_map':      views_map
    }

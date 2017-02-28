#!/usr/bin/env python

'''
Structure from motion
=====================

Example of using features2d framework for interactive video homography matching.
ORB features and FLANN matcher are used. The actual tracking is implemented by
PlaneTracker class in plane_tracker.py

Usage
-----
structure_from_motion.py [<video source>]

Keys:
   SPACE  -  pause video

Select a textured planar object to track by drawing a box with a mouse.
'''

# Python 2/3 compatibility
from __future__ import print_function

# built-in modules
from collections import namedtuple

import numpy as np
import cv2
from numpy import vstack, hstack

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# local modules
# import video
# from video import presets
import common
from common import getsize, draw_keypoints
# from plane_tracker import PlaneTracker

import sys,os
thisdir=os.path.dirname(os.path.abspath(__file__))

np.set_printoptions(suppress=False, formatter={'float_kind':lambda x: "%.5f" % x})

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

MIN_MATCH_COUNT = 10

'''
  image     - image to track
  rect      - tracked rectangle (x1, y1, x2, y2)
  keypoints - keypoints detected inside rect
  descrs    - their descriptors
  data      - some user-provided data
'''
PlanarTarget = namedtuple('PlaneTarget', 'image, rect, keypoints, descrs, data')

'''
  target - reference to PlanarTarget
  p0     - matched points coords in target image
  p1     - matched points coords in input frame
  H      - homography matrix from p0 to p1
  quad   - target bounary quad in input frame
'''
TrackedTarget = namedtuple('TrackedTarget', 'target, p0, p1, H, K, F, e1, e2, H1, H2, quad')

class PlaneTracker:
    def __init__(self):
        self.detector = cv2.ORB_create( nfeatures = 4000 )
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
        self.targets = []
        self.frame_points = []

    def add_target(self, image, rect, data=None):
        '''Add a new tracking target.'''
        x0, y0, x1, y1 = rect
        raw_points, raw_descrs = self.detect_features(image)
        points, descs = [], []
        for kp, desc in zip(raw_points, raw_descrs):
            x, y = kp.pt
            if x0 <= x <= x1 and y0 <= y <= y1:
                points.append(kp)
                descs.append(desc)
        descs = np.uint8(descs)
        self.matcher.add([descs])
        target = PlanarTarget(image = image, rect=rect, keypoints = points, descrs=descs, data=data)
        self.targets.append(target)

    def clear(self):
        '''Remove all targets'''
        self.targets = []
        self.matcher.clear()

    def track(self, frame):
        '''Returns a list of detected TrackedTarget objects'''
        self.frame_points, frame_descrs = self.detect_features(frame)
        if len(self.frame_points) < MIN_MATCH_COUNT:
            return []
        matches = self.matcher.knnMatch(frame_descrs, k = 2)
        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
        if len(matches) < MIN_MATCH_COUNT:
            return []
        matches_by_id = [[] for _ in xrange(len(self.targets))]
        for m in matches:
            matches_by_id[m.imgIdx].append(m)
        tracked = []
        for imgIdx, matches in enumerate(matches_by_id):
            if len(matches) < MIN_MATCH_COUNT:
                continue
            target = self.targets[imgIdx]
            p0 = [target.keypoints[m.trainIdx].pt for m in matches]
            p1 = [self.frame_points[m.queryIdx].pt for m in matches]
            p0, p1 = np.float32((p0, p1))

            H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
            # status = status.ravel() != 0
            # if status.sum() < MIN_MATCH_COUNT:
            #     continue
            # p0, p1 = p0[status], p1[status]

            # F, status = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, 3, .99, status)
            F, status = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, 3, .99)
            print(status.mean())
            status = status.ravel() != 0
            if status.sum() < MIN_MATCH_COUNT:
                continue
            p0, p1 = p0[status], p1[status]
            e1 = cv2.computeCorrespondEpilines(p0, 1, F)
            e2 = cv2.computeCorrespondEpilines(p1, 2, F)

            p0 = hstack((p0, np.ones((p0.shape[0],1)))).astype(np.float32)
            K = cv2.initCameraMatrix2D([p0], [p1], common.getsize(frame))
            print(K)
            p0 = p0[:,:2]
            
            w, h = common.getsize(frame)
            retval, H1, H2 = cv2.stereoRectifyUncalibrated(p0, p1, F, (w, h))

            projMat1 = np.hstack((H1, np.ones((3,1))))
            projMat2 = np.hstack((H2, np.ones((3,1))))
            points4D = cv2.triangulatePoints(projMat1, projMat2, p0.T, p1.T)
            print(points4D.T)
            
            x0, y0, x1, y1 = target.rect
            quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)

            track = TrackedTarget(target=target, p0=p0, p1=p1, H=H, K=K, 
                                  F=F, e1=e1, e2=e2, H1=H1, H2=H2, quad=quad)
            tracked.append(track)
        tracked.sort(key = lambda t: len(t.p0), reverse=True)
        return tracked

    def detect_features(self, frame):
        '''detect_features(self, frame) -> keypoints, descrs'''
        keypoints, descrs = self.detector.detectAndCompute(frame, None)
        if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
            descrs = []
        return keypoints, descrs

class App:
    def __init__(self, src):
        self.cap = None # video.create_capture(src, presets['book'])
        self.frame = None
        self.paused = False
        self.tracker = PlaneTracker()
        self.SGBM = 0
        self.maxDiff = 32
        self.blockSize = 21
        self.stereoMatcher = cv2.StereoBM_create(self.maxDiff, self.blockSize)

        # cv2.namedWindow('plane')
        # self.rect_sel = common.RectSelector('plane', self.on_rect)

    # def on_rect(self, rect):
    #     self.tracker.clear()
    #     self.tracker.add_target(self.frame, rect)

    def run(self):
        for imgidx in range(27):
            print(os.path.join(thisdir,'../data/castle/castle.%03d.jpg' % (imgidx,))+'\n'+
                  os.path.join(thisdir,'../data/castle/castle.%03d.jpg' % (imgidx+1,)))
            img1 = cv2.imread(os.path.join(thisdir,'../data/castle/castle.%03d.jpg' % (imgidx,)))
            img2 = cv2.imread(os.path.join(thisdir,'../data/castle/castle.%03d.jpg' % (imgidx+1,)))
            if img1==None or img2==None:
                raise Exception('Fail to open images.')
            w, h = getsize(img1)
            x0, y0, x1, y1 = (0, 0, w, h)
            self.tracker.clear()
            self.tracker.add_target(img1, (x0, y0, x1, y1))
            tracked = self.tracker.track(img2)

            rectified = np.zeros((h, w, 3), np.uint8)
            disparity = np.zeros((h, w, 3), np.float32)
            # vis[:h,:w] = img2
            if len(self.tracker.targets) > 0:
                target = self.tracker.targets[0]
                # vis[:,w:] = target.image
                # draw_keypoints(vis[:,w:], target.keypoints)
                x0, y0, x1, y1 = target.rect
                # cv2.rectangle(vis, (x0+w, y0), (x1+w, y1), (0, 255, 0), 2)
            if len(tracked) > 0:
                tracked = tracked[0]
                H, K, F, e1, e2, H1, H2 = (tracked.H, tracked.K, tracked.F, tracked.e1, tracked.e2, tracked.H1, tracked.H2)
                dist_coef = np.zeros(4)

                # retval, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
                # print(H)

                warpSize = (int(w*.9), int(h*.9))
                img1_warp = cv2.warpPerspective(img1, H1, warpSize)
                img2_warp = cv2.warpPerspective(img2, H2, warpSize)
                rectified = cv2.addWeighted(img1_warp, .5, img2_warp, .5, 0)

                # disparity = self.stereoMatcher.compute(cv2.cvtColor(img1_warp,cv2.COLOR_BGR2GRAY),
                #                                        cv2.cvtColor(img2_warp,cv2.COLOR_BGR2GRAY))
                # disparity, buf = cv2.filterSpeckles(disparity, -self.maxDiff, pow(20,2), self.maxDiff)
               
                # invH = np.linalg.inv(H)
                # print('invH = \n',invH)
                # img2_warp = cv2.warpPerspective(img2,invH,(w,h))
            rectified = cv2.addWeighted(img1_warp,.5,img2_warp,.5,0)
            cv2.imshow('rectified', rectified); [ exit(0) if cv2.waitKey()==27 else None ]
            # cv2.imshow('disparity', ((disparity+50)*.5).astype(np.uint8)); [ exit(0) if cv2.waitKey()==27 else None ]


if __name__ == '__main__':
    print(__doc__)

    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    App(video_src).run()



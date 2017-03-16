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
import sys,os
thisdir=os.path.dirname(os.path.abspath(__file__))

import numpy as np
import matplotlib.pyplot as plt
import cv2
np.set_printoptions(suppress=False, formatter={'float_kind':lambda x: "%.5f" % x})

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# local modules
# import visualization
import common
from common import getsize, draw_keypoints, draw_matches
# import sba

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


def sbaDriver(camname,ptsname,intrname=None,camoname=None,ptsoname=None):
    logging.debug("sbaDriver() {0} {1}".format(camname,ptsname))

    if intrname is None:
        cameras = sba.Cameras.fromTxt(camname) # Load cameras from file
    else:
        # Intrinsic file for all identical cameras not yet implemented
        logging.warning("Fixed intrinsic parameters not yet implemented")
        cameras = sba.Cameras.fromTxtWithIntr(camname,intrname)
    points = sba.Points.fromTxt(ptsname,cameras.ncameras) # Load points

    options = sba.Options.fromInput(cameras,points)
    #options.camera = sba.OPTS_CAMS_NODIST # only use this if passing in 5+3+3
    options.nccalib=sba.OPTS_FIX2_INTR # fix all intrinsics
    # If you wish to fix the intrinsics do so here by setting options
    options.ncdist = sba.OPTS_FIX5_DIST # fix all distortion coeffs

    newcameras,newpoints,info = sba.SparseBundleAdjust(cameras,points,options)
    info.printResults()

    if camoname:
        newcameras.toTxt(camoname)
    if ptsoname:
        newpoints.toTxt(ptsoname)

def get_P_prime_from_F(F):
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    Z = np.array([[0,-1,0],[1,0,0],[0,0,0]])
    U, D, V = np.linalg.svd(F)
    S = np.dot(np.dot(U,Z),U.T)
    M = np.dot(np.dot(np.dot(U,W.T),np.diag(D)),V)
    # import pdb; pdb.set_trace()
    assert(np.allclose(F,np.dot(np.dot(U,np.diag(D)),V)))
    assert(np.allclose(F,np.dot(S,M)))
    P_prime = np.hstack((M, U[:,-1].reshape((3,1))))
    return P_prime
        
class App:
    def __init__(self, src):
        self.cap = None
        self.frame = None
        self.paused = False
        self.SGBM = 0
        self.maxDiff = 32
        self.blockSize = 21
        self.stereoMatcher = cv2.StereoBM_create(self.maxDiff, self.blockSize)

        self.detector = cv2.ORB_create( nfeatures = 4000 )
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)

    def run(self):
        idxrange = range(1,27)
        img0 = cv2.imread(os.path.join(thisdir,'../data/castle/castle.%03d.jpg' % (idxrange[0],)))
        w, h = getsize(img0)
        prev_keypoints, prev_descrs = self.detector.detectAndCompute(img0.copy(), None)
        self.matcher.add([prev_descrs.astype(np.uint8)])
        
        for imgidx in idxrange:
            print('-------------------------------------------------------')
            print(os.path.join(thisdir,'../data/castle/castle.%03d.jpg' % (imgidx,))+'\n'+
                  os.path.join(thisdir,'../data/castle/castle.%03d.jpg' % (imgidx+1,)))
            img1 = cv2.imread(os.path.join(thisdir,'../data/castle/castle.%03d.jpg' % (imgidx,)))
            img2 = cv2.imread(os.path.join(thisdir,'../data/castle/castle.%03d.jpg' % (imgidx+1,)))
            if img1 is None or img2 is None:
                raise Exception('Fail to open images.')

            # Detect and match keypoints
            prev_keypoints, prev_descrs = self.detector.detectAndCompute(img1.copy(), None)
            curr_keypoints, curr_descrs = self.detector.detectAndCompute(img2.copy(), None)
            self.matcher.clear()
            self.matcher.add([prev_descrs.astype(np.uint8)])
            matches = self.matcher.knnMatch(curr_descrs, k = 2)
            matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
            print('%d matches.' % len(matches))
            p0 = [prev_keypoints[m.trainIdx].pt for m in matches]
            p1 = [curr_keypoints[m.queryIdx].pt for m in matches]
            p0, p1 = np.float32((p0, p1))

            # Skip first two frames
            # if imgidx<2: continue
            
            # Estimate homography
            H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 13.0)
            status = status.ravel() != 0
            print('inliner percentage: %.1f %%' % (status.mean().item(0)*100.,))
            if status.sum() < MIN_MATCH_COUNT:
                continue
            p0, p1 = p0[status], p1[status]

            # Display inliners
            imgpair = cv2.addWeighted(img2, .5, img1, .5, 0)
            draw_matches(imgpair, p0, p1)
            cv2.imshow('keypoint matches', imgpair)

            # Estimate fundamental matrix
            F, status = cv2.findFundamentalMat(p0, p1, cv2.FM_8POINT, 3, .99)

            # Estimate camera matrix
            p0 = np.hstack((p0, np.ones((p0.shape[0],1)))).astype(np.float32)
            K = cv2.initCameraMatrix2D([p0], [p1], (w, h))
            p0 = p0[:,:2]
            
            # Estimate essential matrix
            E, status = cv2.findEssentialMat(p0, p1, cameraMatrix=K)
            ret, R, t, status = cv2.recoverPose(E, p0, p1, cameraMatrix=K, mask = status)
            rvec, jacobian = cv2.Rodrigues(R)
            print('(R, t)=', rvec.T, '\n', t.T)

            # Dense rectification
            retval, H1, H2 = cv2.stereoRectifyUncalibrated(p0, p1, F, (w, h))

            # Triangulation
            projMat1 = np.hstack((np.eye(3), np.zeros((3,1))))
            projMat2 = get_P_prime_from_F(F)
            points4D = cv2.triangulatePoints(projMat1, projMat2, p0.T, p1.T)

            # Plot triangulation results
            objectPoints = points4D.T[:,:3].astype(np.float32)
            print(objectPoints)
            np.savetxt('pts.txt', objectPoints, fmt='%.5f')
            plt.close()
            fig, axarr = plt.subplots(2, 2)
            axarr[0,0].plot(p0[:,0], 480-p0[:,1], '.')
            axarr[0,1].plot(p1[:,0], 480-p1[:,1], '.')
            axarr[1,0].plot(-objectPoints[:,2], -objectPoints[:,1], '.')
            axarr[1,1].plot(-objectPoints[:,2], objectPoints[:,0], '.')
            plt.ion()
            plt.draw()
            plt.waitforbuttonpress(1)

            retval, rvec, tvec, inliners = cv2.solvePnPRansac(objectPoints, p1.astype(np.float32), K, np.zeros((1,4)))
            print('rvec, tvec = ', rvec.T, '\n', tvec.T)

            rectified = np.zeros((h, w, 3), np.uint8)
            disparity = np.zeros((h, w, 3), np.float32)

            warpSize = (int(w*.9), int(h*.9))
            img1_warp = cv2.warpPerspective(img1, H1, warpSize)
            img2_warp = cv2.warpPerspective(img2, H2, warpSize)
            rectified = cv2.addWeighted(img1_warp, .5, img2_warp, .5, 0)

            disparity = self.stereoMatcher.compute(cv2.cvtColor(img1_warp,cv2.COLOR_BGR2GRAY),
                                                   cv2.cvtColor(img2_warp,cv2.COLOR_BGR2GRAY))
            disparity, buf = cv2.filterSpeckles(disparity, -self.maxDiff, pow(20,2), self.maxDiff)
                
            rectified = cv2.addWeighted(img1_warp, .5, img2_warp, .5, 0)
            cv2.imshow('rectified', rectified)
            cv2.imshow('disparity', ((disparity+50)*.5).astype(np.uint8))

            [ exit(0) if cv2.waitKey()&0xff==27 else None ]

            prev_keypoints, prev_descrs = curr_keypoints, curr_descrs

if __name__ == '__main__':
    print(__doc__)

    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    App(video_src).run()



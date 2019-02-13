# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Run arguments: python run_wild.py -k detections -arc 3,3,3,3,3 -c checkpoint --evaluate d-pt-243.bin --viz-subject S1 --viz-action Directions --viz-video InTheWildData/vid1.mkv --viz-camera 0
# Run arguments: python run_wild.py -k detections -arc 3,3,3,3,3 -c checkpoint --evaluate d-pt-243.bin --viz-subject S1 --viz-action Directions --viz-video InTheWildData/vid1.mkv --viz-camera 0 --viz-output output_vid.mp4 --viz-size 5 --viz-downsample 1 --viz-skip 9


import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random
from common.visualization import render_animation_live

kps_left = [1, 3, 5, 7, 9, 11, 13, 15]
kps_right = [2, 4, 6, 8, 10, 12, 14, 16]
joints_left = list([4,5,6,11,12,13])
joints_right = list([1,2,3,14,15,16])
parents = [-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15]
parentsIndices = np.array([1, 0, 2, 1, 3, 2, 4, 0, 5, 4, 6, 5, 7, 0, 8, 7, 9, 8, 10, 9, 11, 8, 12, 11, 13, 12, 14, 8, 15, 14, 16, 15], dtype='float32')



skeletonJointsRight = [1,2,3,14,15,16]


def fetch(subjects, keypoints, stride, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    subject = 'S1'
    action = 'Directions 1'

                
    poses_2d = keypoints[subject][action]
    for i in range(len(poses_2d)): # Iterate across cameras
        out_poses_2d.append(poses_2d[i])


    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None
    
    # stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
    

    return out_camera_params, out_poses_3d, out_poses_2d

def evaluateLive(test_generator, model_pos, VBO, window, action=None, return_predictions=False):


    #fig1 = plt.figure(1)
    #ax = fig1.add_subplot(111, projection='3d')	
    #ax.set_xlim3d(0, 5)
    #ax.set_ylim3d(0, 5)
    #ax.set_zlim3d(0, 5)
    with torch.no_grad():
        model_pos.eval()
        predicted_3d_pos = torch.zeros([2, 2493, 17, 3])
        predicted_3d_pos = predicted_3d_pos.cuda()
		# batch_2d is a copy and flipped version of the input pose 2d its a array shape... 
		#(2, numFrames, numJoints, 3)
		#(L,R flip: numFrames: numJoints: x,y,confidence)
        for _, batch, batch_2d in test_generator.next_epoch():
            for i in range(243, batch_2d.shape[1]):
                myInputs = batch_2d[:,i-243:i,:,:]
                inputs_2d = torch.from_numpy(myInputs.astype('float32'))
                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                print('processing frame ', i)
                # Positional model
                tempPredictedPos = model_pos(inputs_2d)
                predicted_3d_pos[:,i-243:i,:,:] = tempPredictedPos
                #predicted_3d_pos = model_pos(inputs_2d)
                tempPredictedPos[1, :, :, 0] *= -1
                tempPredictedPos[1, :, joints_left + joints_right] = tempPredictedPos[1, :, joints_right + joints_left]
                tempPredictedPos = torch.mean(tempPredictedPos, dim=0, keepdim=True)

                cpuPred = tempPredictedPos.squeeze(0).cpu().numpy()
                cpuPred = camera_to_world(cpuPred, R=[ 0.0, 0.0, 0.0 , 0.0], t=[0 , 0, 0])
                # We don't have the trajectory, but at least we can rebase the height
                cpuPred[:, :, 2] -= np.min(cpuPred[:, :, 2])
                bufArray = cpuPred[0,:,:].flatten()

                glBindBuffer(GL_ARRAY_BUFFER, VBO)
                glBufferSubData(GL_ARRAY_BUFFER, 0, 17 * 3 * 4, bufArray)
                glfw.poll_events()
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glDrawElements(GL_LINES, 16, GL_UNSIGNED_INT, None)
                glfw.swap_buffers(window)

                #anim_output = {'Reconstruction': cpuPred}
                #plt.figure(1)	
                # https://stackoverflow.com/questions/22171558/what-does-enumerate-mean
                #for j, j_parent in enumerate(parents):
                #    if j_parent == -1:
                #        continue
                #    col = 'red' if j in skeletonJointsRight else 'black'
                    
                    
                #    ax.plot([cpuPred[0, j, 0], cpuPred[0, j_parent, 0]],[cpuPred[0, j, 1], cpuPred[0, j_parent, 1]], [cpuPred[0, j, 2], cpuPred[0, j_parent, 2]], zdir='z', c=col)
                #    ax.set_xlim3d(-2, 2)
                #    ax.set_ylim3d(-2, 2)
                #    ax.set_zlim3d(-2, 2)
                    

                #plt.pause(0.01)
                #plt.cla()
                #render_animation_live(cpuPred)				
                #print('shape ', predicted_3d_pos.shape)

                # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

                
            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()

def evaluate(test_generator, model_pos, action=None, return_predictions=False):
    with torch.no_grad():
        model_pos.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
                
            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()

def main():


    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "My OpenGL window", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    #        positions        colors
    cube = [-0.5, -0.5,  0.5, 1.0, 0.0, 0.0,
             0.5, -0.5,  0.5, 0.0, 1.0, 0.0,
             0.5,  0.5,  0.5, 0.0, 0.0, 1.0,
            -0.5,  0.5,  0.5, 1.0, 1.0, 1.0,
            
            -0.5, -0.5, -0.5, 1.0, 0.0, 0.0,
             0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
             0.5,  0.5, -0.5, 0.0, 0.0, 1.0,
            -0.5,  0.5, -0.5, 1.0, 1.0, 1.0]

    skeletonArray = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15, 16]
    skeletonArray = np.array(skeletonArray, dtype = np.float32)

    cube = np.array(cube, dtype = np.float32)

    indices = [0, 1, 2, 2, 3, 0,
               4, 5, 6, 6, 7, 4,
               4, 5, 1, 1, 0, 4,
               6, 7, 3, 3, 2, 6,
               5, 6, 2, 2, 1, 5,
               7, 4, 0, 0, 3, 7]

    indices = np.array(indices, dtype= np.uint32)

    vertex_shader = """
    #version 330
    in vec3 position;
    uniform mat4 transform;
    void main()
    {
        gl_Position = transform * vec4(position, 1.0f);
    }
    """

    fragment_shader = """
    #version 330
    out vec4 outColor;
    void main()
    {
        outColor = vec4(1.0f,1.0f,1.0f,1.0f);
    }
    """
    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 17 * 3 * 4, None, GL_DYNAMIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 128, parentsIndices, GL_STATIC_DRAW)

    position = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)




    glUseProgram(shader)

    glClearColor(0.2, 0.3, 0.2, 1.0)
    glEnable(GL_DEPTH_TEST)





    args = parse_args()
    print(args)

    try:
        # Create checkpoint directory if it does not exist
        os.makedirs(args.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)
    
    print('Loading 2D detections...')
    
    keypoints = np.load('data/data_2d_' + args.keypoints + '.npz')
    
    

    
    keypoints = keypoints['positions_2d'].item()
    
    subject = 'S1'
    
    action = 'Directions 1'
    
    width_of = 410
    height_of = 374

    for cam_idx, kps in enumerate(keypoints[subject][action]):
    
        # Normalize camera frame
        # cam = dataset.cameras()[subject][cam_idx]
        kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=width_of, h=height_of)
        keypoints[subject][action][cam_idx] = kps
    
    subjects_train = args.subjects_train.split(',')
    subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
    subjects_test = args.subjects_test.split(',')
    
    semi_supervised = len(subjects_semi) > 0
    if semi_supervised and not dataset.supports_semi_supervised():
        raise RuntimeError('Semi-supervised training is not implemented for this dataset')
            


    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        print('Selected actions:', action_filter)
        
    cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, keypoints, args.downsample, action_filter)
    
    filter_widths = [int(x) for x in args.architecture.split(',')]
    
    
    
    
    if args.viz_output:	
        model_pos = TemporalModel(poses_valid_2d[0].shape[1], poses_valid_2d[0].shape[2], 17, filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels, dense=args.dense)
    else:
    	model_pos = TemporalModelOptimized1f(poses_valid_2d[0].shape[1], poses_valid_2d[0].shape[2], 17, filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels)
    
    receptive_field = model_pos.receptive_field()
    print('INFO: Receptive field: {} frames'.format(receptive_field))
    pad = (receptive_field - 1) // 2 # Padding on each side
    if args.causal:
        print('INFO: Using causal convolutions')
        causal_shift = pad
    else:
        causal_shift = 0
    
    model_params = 0
    for parameter in model_pos.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    
    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
    #    model_pos_train = model_pos_train.cuda()
        
    if args.resume or args.evaluate:
        chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        print('This model was trained for {} epochs'.format(checkpoint['epoch']))
        model_pos.load_state_dict(checkpoint['model_pos'])
    

    if args.viz_output:	
    
        print('Rendering...')
        my_action = 'Directions 1'
    
        input_keypoints = keypoints[args.viz_subject][my_action][args.viz_camera].copy()
    
            
        gen = UnchunkedGenerator(None, None, [input_keypoints],
                                 pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                 kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

        prediction = evaluate(gen, model_pos, return_predictions=True)

        ground_truth = None

        # these values taken from a camera in the h36m dataset, would be good to get/determine values rom stereo calibration of the pip cameras
        prediction = camera_to_world(prediction, R=[ 0.14070565, -0.15007018, -0.7552408 ,  0.62232804], t=[1.841107 , 4.9552846, 0.5634454])
        # We don't have the trajectory, but at least we can rebase the height
        prediction[:, :, 2] -= np.min(prediction[:, :, 2])
        
        anim_output = {'Reconstruction': prediction}
        
        input_keypoints = image_coordinates(input_keypoints[..., :2], w=width_of, h=height_of)
    
        manual_fps = 25
    
        np.savez('out_3D_vp3d', anim_output['Reconstruction'])
        camAzimuth = 70.0
        from common.visualization import render_animation
        render_animation(input_keypoints, anim_output,
                         manual_fps, args.viz_bitrate, camAzimuth, args.viz_output,
                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=args.viz_video, viewport=(width_of, height_of),
                         input_video_skip=args.viz_skip)

    else:
        print('Rendering...')
        my_action = 'Directions 1'
    
        input_keypoints = keypoints[args.viz_subject][my_action][args.viz_camera].copy()
    
            
        gen = UnchunkedGenerator(None, None, [input_keypoints],
                                 pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                 kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

        prediction = evaluateLive(gen, model_pos, VBO, window, return_predictions=True)
        glfw.terminate()
    









if __name__ == "__main__":
    main()
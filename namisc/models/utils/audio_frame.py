'''
Author: LOTEAT
Date: 2023-06-30 16:15:16
'''
import torch


def enframe(wav_data, num_frame, frame_length, stride_length):
    frame_indices = torch.tile(torch.arange(0, frame_length).reshape(1, frame_length), [num_frame, 1])
    stride_indices = torch.transpose(torch.tile(torch.arange(0, num_frame * stride_length, stride_length).reshape(1, num_frame), [frame_length, 1]), 0, 1)
    frame_indices_combined = torch.add(frame_indices, stride_indices)  # index of each frame

    flat_indices = frame_indices_combined.reshape(num_frame * frame_length)
    frame_input = torch.gather(wav_data, 1, flat_indices.unsqueeze(0).repeat(wav_data.size(0), 1))
    frame_input = frame_input.reshape(frame_input.size(0), num_frame, frame_length)
    return frame_input


def deframe(frame_output, num_frame, frame_length, stride_length):
    wav1 = frame_output[:, 0 : num_frame - 1, 0 : stride_length].reshape(frame_output.size(0), (num_frame - 1) * stride_length)
    wav2 = frame_output[:, num_frame - 1, 0 : frame_length].reshape(frame_output.size(0), frame_length)
    wav_output = torch.cat([wav1, wav2], dim=1)

    return wav_output
